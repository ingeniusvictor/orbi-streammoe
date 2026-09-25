#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace orbi::streammoe {

struct QwenGatedDeltaNetConfig {
  std::size_t hidden_size{};
  std::size_t num_key_heads{};
  std::size_t num_value_heads{};
  std::size_t key_head_dim{};
  std::size_t value_head_dim{};
  std::size_t conv_kernel_size{};
  float rms_eps{1e-6F};

  [[nodiscard]] std::size_t key_dim() const noexcept {
    return num_key_heads * key_head_dim;
  }

  [[nodiscard]] std::size_t value_dim() const noexcept {
    return num_value_heads * value_head_dim;
  }

  [[nodiscard]] std::size_t conv_dim() const noexcept {
    return 2U * key_dim() + value_dim();
  }
};

struct QwenGatedDeltaNetWeights {
  // Fused Qwen3-Next affine outputs.
  std::span<const float> in_proj_qkvz;
  std::span<const float> in_proj_ba;

  // Depthwise conv weight, row-major [conv_dim, kernel].
  std::span<const float> conv;

  std::span<const float> dt_bias;
  std::span<const float> a_log;

  // Per-value-head RMSNorm weight, length value_head_dim.
  std::span<const float> norm;

  // Row-major [hidden_size, value_dim].
  std::span<const float> out_proj;
};

struct QwenGatedDeltaNetState {
  // [kernel - 1, conv_dim], row-major. Empty means zero-initialized.
  std::vector<float> conv_tail;

  // [num_value_heads, value_head_dim, key_head_dim], row-major.
  // Empty means zero-initialized.
  std::vector<float> recurrent;
};

struct QwenGatedDeltaNetStepResult {
  bool executed{};
  std::vector<float> values;
  std::string diagnostic;
};

/// Correctness-first single-token CPU decode step for Qwen3-Next
/// Gated DeltaNet, matching the frozen Swiftlet/MLX fused-interleaved layout.
[[nodiscard]] QwenGatedDeltaNetStepResult run_qwen_gated_deltanet_cpu_step(
    std::span<const float> hidden,
    QwenGatedDeltaNetWeights weights,
    const QwenGatedDeltaNetConfig& config,
    QwenGatedDeltaNetState& state) noexcept;

}  // namespace orbi::streammoe
