#include "orbi/streammoe/cpu/reference_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace orbi::streammoe::cpu {

std::vector<float> dequantize_affine_rows(
    std::span<const std::uint32_t> packed,
    std::size_t rows,
    std::size_t packed_cols,
    std::span<const float> scales,
    std::span<const float> biases,
    AffineQuantSpec spec) {
  if (spec.bits != 4 && spec.bits != 8) {
    throw std::invalid_argument("affine dequant: only 4-bit and 8-bit are supported");
  }
  if (spec.group_size == 0) {
    throw std::invalid_argument("affine dequant: group_size must be non-zero");
  }
  if (rows == 0 || packed_cols == 0) {
    throw std::invalid_argument("affine dequant: rows and packed_cols must be non-zero");
  }

  const std::size_t per_word = 32U / spec.bits;
  const std::size_t logical_cols = packed_cols * per_word;
  if ((logical_cols % spec.group_size) != 0U) {
    throw std::invalid_argument(
        "affine dequant: logical columns must be divisible by group_size");
  }

  const std::size_t groups_per_row = logical_cols / spec.group_size;
  if (packed.size() != rows * packed_cols) {
    throw std::invalid_argument("affine dequant: packed size mismatch");
  }
  if (scales.size() != rows * groups_per_row ||
      biases.size() != rows * groups_per_row) {
    throw std::invalid_argument("affine dequant: scale/bias size mismatch");
  }

  const std::uint32_t mask = (1U << spec.bits) - 1U;
  std::vector<float> out(rows * logical_cols, 0.0F);

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t word_index = 0; word_index < packed_cols; ++word_index) {
      std::uint32_t word = packed[row * packed_cols + word_index];
      const std::size_t col_base = word_index * per_word;

      for (std::size_t lane = 0; lane < per_word; ++lane) {
        const std::size_t col = col_base + lane;
        const std::size_t group = col / spec.group_size;
        const float q = static_cast<float>(word & mask);

        out[row * logical_cols + col] =
            scales[row * groups_per_row + group] * q +
            biases[row * groups_per_row + group];

        word >>= spec.bits;
      }
    }
  }

  return out;
}

std::vector<float> matvec_row_major(
    std::span<const float> weight,
    std::size_t out_dim,
    std::size_t in_dim,
    std::span<const float> x) {
  if (out_dim == 0 || in_dim == 0) {
    throw std::invalid_argument("matvec: dimensions must be non-zero");
  }
  if (weight.size() != out_dim * in_dim) {
    throw std::invalid_argument("matvec: weight size mismatch");
  }
  if (x.size() != in_dim) {
    throw std::invalid_argument("matvec: input size mismatch");
  }

  std::vector<float> out(out_dim, 0.0F);
  for (std::size_t row = 0; row < out_dim; ++row) {
    float acc = 0.0F;
    const auto base = row * in_dim;
    for (std::size_t col = 0; col < in_dim; ++col) {
      acc += weight[base + col] * x[col];
    }
    out[row] = acc;
  }
  return out;
}

void rms_norm_inplace(
    std::span<float> values,
    std::size_t rows,
    std::size_t dim,
    std::span<const float> weight,
    float eps) {
  if (rows == 0 || dim == 0 || values.size() != rows * dim) {
    throw std::invalid_argument("rms_norm: invalid shape");
  }
  if (!weight.empty() && weight.size() != dim) {
    throw std::invalid_argument("rms_norm: weight size mismatch");
  }
  if (eps < 0.0F) {
    throw std::invalid_argument("rms_norm: eps must be non-negative");
  }

  for (std::size_t row = 0; row < rows; ++row) {
    const auto base = row * dim;
    float sum_sq = 0.0F;
    for (std::size_t i = 0; i < dim; ++i) {
      const auto v = values[base + i];
      sum_sq += v * v;
    }

    const float scale =
        1.0F / std::sqrt(sum_sq / static_cast<float>(dim) + eps);

    for (std::size_t i = 0; i < dim; ++i) {
      const float w = weight.empty() ? 1.0F : weight[i];
      values[base + i] *= scale * w;
    }
  }
}

void softmax_inplace(std::span<float> values) {
  if (values.empty()) {
    throw std::invalid_argument("softmax: input must not be empty");
  }

  const float max_value =
      *std::max_element(values.begin(), values.end());

  float sum = 0.0F;
  for (auto& value : values) {
    value = std::exp(value - max_value);
    sum += value;
  }

  if (!(sum > 0.0F) || !std::isfinite(sum)) {
    throw std::runtime_error("softmax: invalid normalization sum");
  }

  for (auto& value : values) {
    value /= sum;
  }
}

std::vector<RoutingPick> route_top_k(
    std::span<const float> logits,
    std::size_t k,
    bool normalize_selected) {
  if (logits.empty()) {
    throw std::invalid_argument("router: logits must not be empty");
  }
  if (k == 0 || k > logits.size()) {
    throw std::invalid_argument("router: k must be within expert count");
  }

  std::vector<float> probabilities(logits.begin(), logits.end());
  softmax_inplace(probabilities);

  std::vector<RoutingPick> picks;
  picks.reserve(k);

  for (std::size_t expert = 0; expert < probabilities.size(); ++expert) {
    const RoutingPick candidate{
        static_cast<std::uint32_t>(expert),
        probabilities[expert],
    };

    if (picks.size() < k) {
      picks.push_back(candidate);
      std::stable_sort(
          picks.begin(),
          picks.end(),
          [](const RoutingPick& a, const RoutingPick& b) {
            return a.probability > b.probability;
          });
      continue;
    }

    if (candidate.probability > picks.back().probability) {
      picks.back() = candidate;
      std::stable_sort(
          picks.begin(),
          picks.end(),
          [](const RoutingPick& a, const RoutingPick& b) {
            return a.probability > b.probability;
          });
    }
  }

  if (normalize_selected) {
    float sum = 0.0F;
    for (const auto& pick : picks) {
      sum += pick.probability;
    }
    if (!(sum > 0.0F) || !std::isfinite(sum)) {
      throw std::runtime_error("router: selected probability sum is invalid");
    }
    for (auto& pick : picks) {
      pick.probability /= sum;
    }
  }

  return picks;
}

float sigmoid(float value) noexcept {
  return 1.0F / (1.0F + std::exp(-value));
}

float silu(float value) noexcept {
  return value * sigmoid(value);
}

void swiglu_inplace(
    std::span<float> gate,
    std::span<const float> up) {
  if (gate.size() != up.size()) {
    throw std::invalid_argument("swiglu: gate/up size mismatch");
  }

  for (std::size_t i = 0; i < gate.size(); ++i) {
    gate[i] = silu(gate[i]) * up[i];
  }
}

}  // namespace orbi::streammoe::cpu
