#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace orbi::streammoe {

struct QwenGqaConfig {
  std::size_t hidden_size{};
  std::size_t num_attention_heads{};
  std::size_t num_key_value_heads{};
  std::size_t head_dim{};
  float partial_rotary_factor{0.25F};
  float rope_theta{10000000.0F};
  float rms_eps{1e-6F};
  std::size_t max_position_embeddings{};

  [[nodiscard]] std::size_t rotary_dims() const noexcept {
    return static_cast<std::size_t>(
        static_cast<double>(head_dim) *
        static_cast<double>(partial_rotary_factor));
  }
};

struct QwenGqaWeights {
  // Row-major affine weights.
  // q_proj: [num_attention_heads * 2 * head_dim, hidden_size]
  // k_proj: [num_key_value_heads * head_dim, hidden_size]
  // v_proj: [num_key_value_heads * head_dim, hidden_size]
  // o_proj: [hidden_size, num_attention_heads * head_dim]
  std::span<const float> q_proj;
  std::span<const float> k_proj;
  std::span<const float> v_proj;
  std::span<const float> o_proj;

  std::span<const float> q_norm;
  std::span<const float> k_norm;
};

struct QwenGqaState {
  std::size_t position{};
  // Flat [position, kv_head, head_dim].
  std::vector<float> k_cache;
  std::vector<float> v_cache;
};

struct QwenGqaStepResult {
  bool executed{};
  std::vector<float> values;
  std::string diagnostic;
};

/// Correctness-first single-token gated GQA decode step matching the frozen
/// Qwen3-Next/Swiftlet semantics.
[[nodiscard]] QwenGqaStepResult run_qwen_gqa_cpu_step(
    std::span<const float> hidden,
    QwenGqaWeights weights,
    const QwenGqaConfig& config,
    QwenGqaState& state) noexcept;

}  // namespace orbi::streammoe
