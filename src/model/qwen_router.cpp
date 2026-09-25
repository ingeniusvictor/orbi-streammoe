#include "orbi/streammoe/model/qwen_router.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "orbi/streammoe/cpu/reference_ops.hpp"

namespace orbi::streammoe {

QwenRouterResult route_qwen_top_k_cpu(
    std::span<const float> hidden,
    std::span<const float> router_weight,
    QwenRouterConfig config) {
  if (config.hidden_size == 0U ||
      config.expert_count == 0U ||
      config.top_k == 0U) {
    throw std::invalid_argument(
        "qwen router: dimensions and top_k must be non-zero");
  }
  if (config.top_k > config.expert_count) {
    throw std::invalid_argument(
        "qwen router: top_k cannot exceed expert_count");
  }
  if (config.expert_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument(
        "qwen router: expert_count exceeds routed-expert id range");
  }
  if (hidden.size() != config.hidden_size) {
    throw std::invalid_argument(
        "qwen router: hidden vector size mismatch");
  }
  if (config.expert_count >
      std::numeric_limits<std::size_t>::max() / config.hidden_size) {
    throw std::invalid_argument(
        "qwen router: weight shape overflows size_t");
  }

  const auto required_weights =
      config.expert_count * config.hidden_size;
  if (router_weight.size() != required_weights) {
    throw std::invalid_argument(
        "qwen router: router weight size mismatch");
  }

  for (const auto value : hidden) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "qwen router: hidden vector contains non-finite value");
    }
  }
  for (const auto value : router_weight) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "qwen router: router weight contains non-finite value");
    }
  }

  QwenRouterResult result;
  result.logits = cpu::matvec_row_major(
      router_weight,
      config.expert_count,
      config.hidden_size,
      hidden);

  const auto raw_picks =
      cpu::route_top_k(result.logits, config.top_k, false);

  result.picks.reserve(raw_picks.size());
  float mass = 0.0F;
  for (const auto& pick : raw_picks) {
    mass += pick.probability;
    result.picks.push_back({
        .expert = pick.expert,
        .weight = pick.probability,
    });
  }

  if (!(mass > 0.0F) || !std::isfinite(mass)) {
    throw std::runtime_error(
        "qwen router: selected probability mass is invalid");
  }
  result.selected_probability_mass = mass;

  if (config.norm_topk_prob) {
    for (auto& pick : result.picks) {
      pick.weight /= mass;
    }
  }

  return result;
}

}  // namespace orbi::streammoe
