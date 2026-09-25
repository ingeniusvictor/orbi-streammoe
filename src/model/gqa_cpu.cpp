#include "orbi/streammoe/model/gqa_cpu.hpp"

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

void apply_partial_rope(
    std::span<float> values,
    std::size_t heads,
    std::size_t head_dim,
    std::size_t rotary_dims,
    std::size_t position,
    float theta) {
  if (rotary_dims == 0U) return;

  const auto half = rotary_dims / 2U;
  for (std::size_t head = 0; head < heads; ++head) {
    const auto base = head * head_dim;
    for (std::size_t j = 0; j < half; ++j) {
      const float exponent =
          -static_cast<float>(2U * j) /
          static_cast<float>(rotary_dims);
      const float inv_freq = std::pow(theta, exponent);
      const float angle =
          static_cast<float>(position) * inv_freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);

      const float a = values[base + j];
      const float b = values[base + half + j];
      values[base + j] = a * c - b * s;
      values[base + half + j] = b * c + a * s;
    }
  }
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

QwenGqaStepResult run_qwen_gqa_cpu_step(
    std::span<const float> hidden,
    QwenGqaWeights weights,
    const QwenGqaConfig& config,
    QwenGqaState& state) noexcept {
  QwenGqaStepResult result;

  try {
    const auto D = config.hidden_size;
    const auto H = config.num_attention_heads;
    const auto KVH = config.num_key_value_heads;
    const auto hd = config.head_dim;
    const auto rot = config.rotary_dims();

    if (D == 0U || H == 0U || KVH == 0U || hd == 0U) {
      result.diagnostic = "GQA dimensions must be non-zero";
      return result;
    }
    if (H % KVH != 0U) {
      result.diagnostic =
          "GQA attention-head count must be divisible by KV-head count";
      return result;
    }
    if (hidden.size() != D) {
      result.diagnostic = "GQA hidden size mismatch";
      return result;
    }
    if (!std::isfinite(config.partial_rotary_factor) ||
        config.partial_rotary_factor < 0.0F ||
        config.partial_rotary_factor > 1.0F ||
        rot > hd ||
        (rot % 2U) != 0U) {
      result.diagnostic = "GQA partial RoPE geometry is invalid";
      return result;
    }
    if (!std::isfinite(config.rope_theta) ||
        !(config.rope_theta > 0.0F)) {
      result.diagnostic = "GQA rope theta must be finite and positive";
      return result;
    }
    if (!std::isfinite(config.rms_eps) || config.rms_eps < 0.0F) {
      result.diagnostic = "GQA RMS epsilon must be finite/non-negative";
      return result;
    }
    if (config.max_position_embeddings == 0U ||
        state.position >= config.max_position_embeddings) {
      result.diagnostic = "GQA context window exhausted";
      return result;
    }

    std::size_t q_rows = 0U;
    std::size_t kv_rows = 0U;
    std::size_t q_count = 0U;
    std::size_t kv_count = 0U;
    std::size_t o_count = 0U;
    q_rows = H * 2U * hd;
    kv_rows = KVH * hd;

    if (!checked_product(q_rows, D, &q_count) ||
        !checked_product(kv_rows, D, &kv_count) ||
        !checked_product(D, H * hd, &o_count)) {
      result.diagnostic = "GQA weight geometry overflow";
      return result;
    }

    if (weights.q_proj.size() != q_count ||
        weights.k_proj.size() != kv_count ||
        weights.v_proj.size() != kv_count ||
        weights.o_proj.size() != o_count ||
        weights.q_norm.size() != hd ||
        weights.k_norm.size() != hd) {
      result.diagnostic = "GQA weight geometry mismatch";
      return result;
    }

    const auto expected_cache =
        state.position * KVH * hd;
    if (state.k_cache.size() != expected_cache ||
        state.v_cache.size() != expected_cache) {
      result.diagnostic = "GQA KV cache size disagrees with position";
      return result;
    }

    const auto q_out = cpu::matvec_row_major(
        weights.q_proj,
        q_rows,
        D,
        hidden);

    std::vector<float> q(H * hd, 0.0F);
    std::vector<float> gate(H * hd, 0.0F);
    for (std::size_t head = 0; head < H; ++head) {
      const auto src = head * 2U * hd;
      const auto dst = head * hd;
      for (std::size_t i = 0; i < hd; ++i) {
        q[dst + i] = q_out[src + i];
        gate[dst + i] = q_out[src + hd + i];
      }
    }

    auto k = cpu::matvec_row_major(
        weights.k_proj,
        kv_rows,
        D,
        hidden);
    auto v = cpu::matvec_row_major(
        weights.v_proj,
        kv_rows,
        D,
        hidden);

    cpu::rms_norm_inplace(
        q,
        H,
        hd,
        weights.q_norm,
        config.rms_eps);
    cpu::rms_norm_inplace(
        k,
        KVH,
        hd,
        weights.k_norm,
        config.rms_eps);

    apply_partial_rope(
        q,
        H,
        hd,
        rot,
        state.position,
        config.rope_theta);
    apply_partial_rope(
        k,
        KVH,
        hd,
        rot,
        state.position,
        config.rope_theta);

    state.k_cache.insert(
        state.k_cache.end(),
        k.begin(),
        k.end());
    state.v_cache.insert(
        state.v_cache.end(),
        v.begin(),
        v.end());

    const auto kv_len = state.position + 1U;
    const auto group = H / KVH;
    const float scale =
        1.0F / std::sqrt(static_cast<float>(hd));

    std::vector<float> attn_out(H * hd, 0.0F);
    std::vector<float> scores(kv_len, 0.0F);

    for (std::size_t head = 0; head < H; ++head) {
      const auto kv_head = head / group;
      const auto q_base = head * hd;

      for (std::size_t pos = 0; pos < kv_len; ++pos) {
        const auto k_base =
            (pos * KVH + kv_head) * hd;
        float dot = 0.0F;
        for (std::size_t i = 0; i < hd; ++i) {
          dot += q[q_base + i] *
                 state.k_cache[k_base + i];
        }
        scores[pos] = dot * scale;
      }

      cpu::softmax_inplace(scores);

      const auto out_base = head * hd;
      for (std::size_t pos = 0; pos < kv_len; ++pos) {
        const auto v_base =
            (pos * KVH + kv_head) * hd;
        const float probability = scores[pos];
        for (std::size_t i = 0; i < hd; ++i) {
          attn_out[out_base + i] +=
              probability *
              state.v_cache[v_base + i];
        }
      }
    }

    for (std::size_t i = 0; i < attn_out.size(); ++i) {
      attn_out[i] *= cpu::sigmoid(gate[i]);
    }

    result.values = cpu::matvec_row_major(
        weights.o_proj,
        D,
        H * hd,
        attn_out);

    ++state.position;
    result.executed = true;
    result.diagnostic =
        "Qwen3-Next gated GQA CPU decode step executed.";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("GQA CPU step failed: ") + e.what();
    result.values.clear();
    return result;
  } catch (...) {
    result.diagnostic =
        "GQA CPU step encountered an unknown exception";
    result.values.clear();
    return result;
  }
}

}  // namespace orbi::streammoe
