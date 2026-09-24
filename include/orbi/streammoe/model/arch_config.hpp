#pragma once

#include <cstddef>
#include <cstdint>

namespace orbi::streammoe {

struct ArchConfig {
  std::size_t hidden_size{};
  std::size_t layer_count{};
  std::size_t full_attention_interval{};
  std::size_t vocab_size{};

  std::size_t attention_heads{};
  std::size_t kv_heads{};
  std::size_t head_dim{};

  std::size_t linear_v_heads{};
  std::size_t linear_k_heads{};
  std::size_t linear_k_head_dim{};
  std::size_t linear_v_head_dim{};
  std::size_t conv_kernel_size{};

  std::size_t expert_count{};
  std::size_t expert_top_k{};
  std::size_t moe_intermediate_size{};
  std::size_t shared_expert_intermediate_size{};

  [[nodiscard]] constexpr std::size_t full_attention_layer_count() const noexcept {
    return layer_count / full_attention_interval;
  }

  [[nodiscard]] constexpr std::size_t linear_layer_count() const noexcept {
    return layer_count - full_attention_layer_count();
  }

  [[nodiscard]] constexpr bool is_linear_layer(std::size_t zero_based_index) const noexcept {
    return ((zero_based_index + 1U) % full_attention_interval) != 0U;
  }

  [[nodiscard]] constexpr std::size_t routed_fetches_per_token() const noexcept {
    return layer_count * expert_top_k;
  }
};

inline constexpr ArchConfig kQwen3Next80BA3B{
    .hidden_size = 2048,
    .layer_count = 48,
    .full_attention_interval = 4,
    .vocab_size = 151936,
    .attention_heads = 16,
    .kv_heads = 2,
    .head_dim = 256,
    .linear_v_heads = 32,
    .linear_k_heads = 16,
    .linear_k_head_dim = 128,
    .linear_v_head_dim = 128,
    .conv_kernel_size = 4,
    .expert_count = 512,
    .expert_top_k = 10,
    .moe_intermediate_size = 512,
    .shared_expert_intermediate_size = 512,
};

static_assert(kQwen3Next80BA3B.full_attention_layer_count() == 12);
static_assert(kQwen3Next80BA3B.linear_layer_count() == 36);
static_assert(kQwen3Next80BA3B.routed_fetches_per_token() == 480);

}  // namespace orbi::streammoe
