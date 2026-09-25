#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/container/qpack.hpp"

namespace orbi::streammoe {

struct Qwen3NextDenseConfig {
  std::size_t hidden_size{};
  std::size_t num_hidden_layers{};
  std::size_t full_attention_interval{};

  std::size_t num_attention_heads{};
  std::size_t num_key_value_heads{};
  std::size_t head_dim{};

  std::size_t linear_num_value_heads{};
  std::size_t linear_num_key_heads{};
  std::size_t linear_key_head_dim{};
  std::size_t linear_value_head_dim{};
  std::size_t linear_conv_kernel_dim{};

  std::size_t num_experts{};
  std::size_t num_experts_per_tok{};
  std::size_t moe_intermediate_size{};
  std::size_t shared_expert_intermediate_size{};
  bool norm_topk_prob{};

  [[nodiscard]] bool is_linear_layer(
      std::size_t layer_index) const noexcept {
    return ((layer_index + 1U) % full_attention_interval) != 0U;
  }

  [[nodiscard]] std::size_t key_dim() const noexcept {
    return linear_num_key_heads * linear_key_head_dim;
  }

  [[nodiscard]] std::size_t value_dim() const noexcept {
    return linear_num_value_heads * linear_value_head_dim;
  }

  [[nodiscard]] std::size_t conv_dim() const noexcept {
    return 2U * key_dim() + value_dim();
  }
};

struct QwenMoeDenseBinding {
  MlxAffineModule router;
  MlxAffineModule shared_gate;
  MlxAffineModule shared_up;
  MlxAffineModule shared_down;
  MlxAffineModule shared_expert_gate;
};

struct QwenDeltaDenseBinding {
  std::vector<float> conv;
  std::vector<float> dt_bias;
  std::vector<float> a_log;
  std::vector<float> norm;

  MlxAffineModule out_proj;
  MlxAffineModule in_proj_qkvz;
  MlxAffineModule in_proj_ba;
};

struct QwenAttentionDenseBinding {
  MlxAffineModule q_proj;
  MlxAffineModule k_proj;
  MlxAffineModule v_proj;
  MlxAffineModule o_proj;

  std::vector<float> q_norm;
  std::vector<float> k_norm;
};

struct QwenDenseLayerBinding {
  std::size_t layer_index{};
  bool is_linear{};

  std::vector<float> input_norm;
  std::vector<float> post_attention_norm;
  QwenMoeDenseBinding moe;

  std::optional<QwenDeltaDenseBinding> delta;
  std::optional<QwenAttentionDenseBinding> attention;
};

[[nodiscard]] Qwen3NextDenseConfig parse_qwen3_next_dense_config(
    const QpackReader& qpack);

[[nodiscard]] QwenDenseLayerBinding bind_qwen3_next_dense_layer(
    const QpackMlxCheckpoint& checkpoint,
    const Qwen3NextDenseConfig& config,
    std::size_t layer_index);

}  // namespace orbi::streammoe
