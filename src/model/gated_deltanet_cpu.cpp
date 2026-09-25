#include "orbi/streammoe/model/gated_deltanet_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/cpu/reference_ops.hpp"

namespace orbi::streammoe {
namespace {

float softplus(float x) noexcept {
  const float ax = std::fabs(x);
  return std::max(x, 0.0F) + std::log1p(std::exp(-ax));
}

bool checked_product(
    std::size_t a,
    std::size_t b,
    std::size_t* out) noexcept {
  if (a != 0U && b > std::numeric_limits<std::size_t>::max() / a) {
    return false;
  }
  *out = a * b;
  return true;
}

}  // namespace

QwenGatedDeltaNetStepResult run_qwen_gated_deltanet_cpu_step(
    std::span<const float> hidden,
    QwenGatedDeltaNetWeights weights,
    const QwenGatedDeltaNetConfig& config,
    QwenGatedDeltaNetState& state) noexcept {
  QwenGatedDeltaNetStepResult result;

  try {
    const auto D = config.hidden_size;
    const auto nk = config.num_key_heads;
    const auto nv = config.num_value_heads;
    const auto dk = config.key_head_dim;
    const auto dv = config.value_head_dim;
    const auto K = config.conv_kernel_size;

    if (D == 0U || nk == 0U || nv == 0U || dk == 0U || dv == 0U || K == 0U) {
      result.diagnostic = "Gated DeltaNet dimensions must be non-zero";
      return result;
    }
    if (nv % nk != 0U) {
      result.diagnostic = "Gated DeltaNet value heads must divide by key heads";
      return result;
    }
    if (!std::isfinite(config.rms_eps) || config.rms_eps < 0.0F) {
      result.diagnostic = "Gated DeltaNet RMS epsilon must be finite/non-negative";
      return result;
    }
    if (hidden.size() != D) {
      result.diagnostic = "Gated DeltaNet hidden size mismatch";
      return result;
    }

    const auto key_dim = config.key_dim();
    const auto value_dim = config.value_dim();
    const auto conv_dim = config.conv_dim();
    const auto rep = nv / nk;
    const auto qkvz_out = 2U * key_dim + 2U * value_dim;
    const auto ba_out = 2U * nv;

    std::size_t qkvz_weight_count = 0U;
    std::size_t ba_weight_count = 0U;
    std::size_t conv_weight_count = 0U;
    std::size_t out_weight_count = 0U;
    if (!checked_product(qkvz_out, D, &qkvz_weight_count) ||
        !checked_product(ba_out, D, &ba_weight_count) ||
        !checked_product(conv_dim, K, &conv_weight_count) ||
        !checked_product(D, value_dim, &out_weight_count)) {
      result.diagnostic = "Gated DeltaNet weight geometry overflow";
      return result;
    }

    if (weights.in_proj_qkvz.size() != qkvz_weight_count ||
        weights.in_proj_ba.size() != ba_weight_count ||
        weights.conv.size() != conv_weight_count ||
        weights.dt_bias.size() != nv ||
        weights.a_log.size() != nv ||
        weights.norm.size() != dv ||
        weights.out_proj.size() != out_weight_count) {
      result.diagnostic = "Gated DeltaNet weight geometry mismatch";
      return result;
    }

    const auto tail_size = (K - 1U) * conv_dim;
    const auto recurrent_size = nv * dv * dk;

    if (state.conv_tail.empty()) {
      state.conv_tail.assign(tail_size, 0.0F);
    } else if (state.conv_tail.size() != tail_size) {
      result.diagnostic = "Gated DeltaNet conv-tail state size mismatch";
      return result;
    }

    if (state.recurrent.empty()) {
      state.recurrent.assign(recurrent_size, 0.0F);
    } else if (state.recurrent.size() != recurrent_size) {
      result.diagnostic = "Gated DeltaNet recurrent state size mismatch";
      return result;
    }

    const auto qkvz = cpu::matvec_row_major(
        weights.in_proj_qkvz,
        qkvz_out,
        D,
        hidden);
    const auto ba = cpu::matvec_row_major(
        weights.in_proj_ba,
        ba_out,
        D,
        hidden);

    std::vector<float> mixed_qkv(conv_dim, 0.0F);
    std::vector<float> z(value_dim, 0.0F);
    std::vector<float> b(nv, 0.0F);
    std::vector<float> a(nv, 0.0F);

    // Qwen3-Next fused-interleaved ordering, per key head:
    // [q(dk), k(dk), v(rep*dv), z(rep*dv)]
    const auto chunk = 2U * dk + 2U * rep * dv;
    for (std::size_t hk = 0; hk < nk; ++hk) {
      const auto src = hk * chunk;

      for (std::size_t i = 0; i < dk; ++i) {
        mixed_qkv[hk * dk + i] = qkvz[src + i];
        mixed_qkv[key_dim + hk * dk + i] = qkvz[src + dk + i];
      }

      for (std::size_t ri = 0; ri < rep; ++ri) {
        const auto hv = hk * rep + ri;
        for (std::size_t i = 0; i < dv; ++i) {
          mixed_qkv[2U * key_dim + hv * dv + i] =
              qkvz[src + 2U * dk + ri * dv + i];
          z[hv * dv + i] =
              qkvz[src + 2U * dk + rep * dv + ri * dv + i];
        }
      }

      const auto ba_src = hk * 2U * rep;
      for (std::size_t ri = 0; ri < rep; ++ri) {
        b[hk * rep + ri] = ba[ba_src + ri];
        a[hk * rep + ri] = ba[ba_src + rep + ri];
      }
    }

    // One-token causal depthwise conv, continuing from the persistent tail.
    std::vector<float> padded;
    padded.reserve(K * conv_dim);
    padded.insert(
        padded.end(),
        state.conv_tail.begin(),
        state.conv_tail.end());
    padded.insert(
        padded.end(),
        mixed_qkv.begin(),
        mixed_qkv.end());

    std::vector<float> conv_out(conv_dim, 0.0F);
    for (std::size_t c = 0; c < conv_dim; ++c) {
      float acc = 0.0F;
      for (std::size_t j = 0; j < K; ++j) {
        acc +=
            weights.conv[c * K + j] *
            padded[j * conv_dim + c];
      }
      conv_out[c] = cpu::silu(acc);
    }

    if (tail_size != 0U) {
      state.conv_tail.assign(
          padded.end() - static_cast<std::ptrdiff_t>(tail_size),
          padded.end());
    } else {
      state.conv_tail.clear();
    }

    std::vector<float> qh(key_dim);
    std::vector<float> kh(key_dim);
    std::vector<float> vh(value_dim);
    std::copy_n(conv_out.begin(), key_dim, qh.begin());
    std::copy_n(conv_out.begin() + static_cast<std::ptrdiff_t>(key_dim),
                key_dim,
                kh.begin());
    std::copy_n(conv_out.begin() + static_cast<std::ptrdiff_t>(2U * key_dim),
                value_dim,
                vh.begin());

    cpu::rms_norm_inplace(qh, nk, dk, std::span<const float>{}, 1e-6F);
    cpu::rms_norm_inplace(kh, nk, dk, std::span<const float>{}, 1e-6F);

    const float inv_scale = 1.0F / std::sqrt(static_cast<float>(dk));
    for (auto& value : qh) value *= inv_scale * inv_scale;
    for (auto& value : kh) value *= inv_scale;

    std::vector<float> out(value_dim, 0.0F);
    for (std::size_t hv = 0; hv < nv; ++hv) {
      const auto hk = hv / rep;
      const float g = std::exp(
          -std::exp(weights.a_log[hv]) *
          softplus(a[hv] + weights.dt_bias[hv]));
      const float beta = cpu::sigmoid(b[hv]);

      const auto q_base = hk * dk;
      const auto k_base = hk * dk;
      const auto v_base = hv * dv;

      for (std::size_t dvi = 0; dvi < dv; ++dvi) {
        const auto st_base = (hv * dv + dvi) * dk;

        float kv_mem = 0.0F;
        for (std::size_t dki = 0; dki < dk; ++dki) {
          auto& slot = state.recurrent[st_base + dki];
          slot *= g;
          kv_mem += slot * kh[k_base + dki];
        }

        const float delta =
            (vh[v_base + dvi] - kv_mem) * beta;

        float y = 0.0F;
        for (std::size_t dki = 0; dki < dk; ++dki) {
          auto& slot = state.recurrent[st_base + dki];
          slot += kh[k_base + dki] * delta;
          y += slot * qh[q_base + dki];
        }
        out[v_base + dvi] = y;
      }
    }

    auto normed = out;
    cpu::rms_norm_inplace(
        normed,
        nv,
        dv,
        weights.norm,
        config.rms_eps);

    for (std::size_t i = 0; i < normed.size(); ++i) {
      normed[i] *= cpu::silu(z[i]);
    }

    result.values = cpu::matvec_row_major(
        weights.out_proj,
        D,
        value_dim,
        normed);
    result.executed = true;
    result.diagnostic =
        "Qwen3-Next Gated DeltaNet CPU decode step executed.";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("Gated DeltaNet CPU step failed: ") + e.what();
    result.values.clear();
    return result;
  } catch (...) {
    result.diagnostic =
        "Gated DeltaNet CPU step encountered an unknown exception";
    result.values.clear();
    return result;
  }
}

}  // namespace orbi::streammoe
