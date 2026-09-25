#include "orbi/streammoe/model/routed_moe.hpp"

#include <cmath>
#include <exception>
#include <string>

namespace orbi::streammoe {

RoutedMoeResult run_weighted_routed_moe(
    VulkanResidentExpertCache& expert_cache,
    std::uint32_t layer,
    std::span<const float> hidden,
    std::span<const float> router_weight,
    QwenRouterConfig router_config) noexcept {
  RoutedMoeResult result;

  try {
    result.routing =
        route_qwen_top_k_cpu(hidden, router_weight, router_config);

    if (result.routing.picks.empty()) {
      result.diagnostic = "routed MoE selected no experts";
      return result;
    }

    bool initialized = false;

    for (const auto& pick : result.routing.picks) {
      if (!std::isfinite(pick.weight)) {
        result.diagnostic =
            "routed MoE encountered a non-finite routing weight";
        return result;
      }

      const auto expert_result =
          expert_cache.run(layer, pick.expert, hidden);
      if (!expert_result.executed) {
        result.diagnostic =
            "routed MoE expert " + std::to_string(pick.expert) +
            " failed: " + expert_result.diagnostic;
        return result;
      }

      if (!initialized) {
        result.values.assign(expert_result.values.size(), 0.0F);
        initialized = true;
      } else if (expert_result.values.size() != result.values.size()) {
        result.diagnostic =
            "routed MoE expert output dimensions disagree";
        result.values.clear();
        return result;
      }

      for (std::size_t i = 0; i < result.values.size(); ++i) {
        result.values[i] += pick.weight * expert_result.values[i];
      }
    }

    if (!initialized || result.values.empty()) {
      result.diagnostic = "routed MoE produced no output";
      result.values.clear();
      return result;
    }

    for (const auto value : result.values) {
      if (!std::isfinite(value)) {
        result.diagnostic = "routed MoE produced non-finite output";
        result.values.clear();
        return result;
      }
    }

    result.executed = true;
    result.diagnostic =
        "weighted routed MoE executed through Vulkan resident expert cache";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("routed MoE exception: ") + e.what();
    result.values.clear();
    return result;
  } catch (...) {
    result.diagnostic =
        "routed MoE encountered an unknown exception";
    result.values.clear();
    return result;
  }
}

}  // namespace orbi::streammoe
