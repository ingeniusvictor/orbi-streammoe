#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace orbi::streammoe::cpu {

struct AffineQuantSpec {
  std::uint32_t bits{};
  std::uint32_t group_size{};
};

struct RoutingPick {
  std::uint32_t expert{};
  float probability{};
};

[[nodiscard]] std::vector<float> dequantize_affine_rows(
    std::span<const std::uint32_t> packed,
    std::size_t rows,
    std::size_t packed_cols,
    std::span<const float> scales,
    std::span<const float> biases,
    AffineQuantSpec spec);

[[nodiscard]] std::vector<float> matvec_row_major(
    std::span<const float> weight,
    std::size_t out_dim,
    std::size_t in_dim,
    std::span<const float> x);

void rms_norm_inplace(
    std::span<float> values,
    std::size_t rows,
    std::size_t dim,
    std::span<const float> weight,
    float eps);

void softmax_inplace(std::span<float> values);

[[nodiscard]] std::vector<RoutingPick> route_top_k(
    std::span<const float> logits,
    std::size_t k,
    bool normalize_selected);

[[nodiscard]] float sigmoid(float value) noexcept;
[[nodiscard]] float silu(float value) noexcept;

void swiglu_inplace(
    std::span<float> gate,
    std::span<const float> up);

}  // namespace orbi::streammoe::cpu
