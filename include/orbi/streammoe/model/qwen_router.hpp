#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace orbi::streammoe {

struct QwenRouterConfig {
  std::size_t hidden_size{};
  std::size_t expert_count{};
  std::size_t top_k{};
  bool norm_topk_prob{};
};

struct RoutedExpert {
  std::uint32_t expert{};
  float weight{};
};

struct QwenRouterResult {
  std::vector<float> logits;
  std::vector<RoutedExpert> picks;

  // Sum of the globally-softmaxed probabilities of the selected experts
  // before optional top-k renormalization.
  float selected_probability_mass{};
};

/// CPU correctness/reference implementation of the Qwen sparse MoE router.
///
/// router_weight is row-major [expert_count, hidden_size].
///
/// Semantics:
///   logits = router_weight * hidden
///   probabilities = softmax(logits)
///   picks = top-k(probabilities)
///   if norm_topk_prob: pick weights /= selected probability mass
///
/// Equal-probability ties are deterministic: lower expert id wins.
[[nodiscard]] QwenRouterResult route_qwen_top_k_cpu(
    std::span<const float> hidden,
    std::span<const float> router_weight,
    QwenRouterConfig config);

}  // namespace orbi::streammoe
