#include "orbi/streammoe/model/routed_moe.hpp"

#include <cmath>
#include <exception>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"
#include "orbi/streammoe/backend/vulkan_weighted_accumulate.hpp"

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


RoutedMoeResult run_weighted_routed_moe_vulkan_accum(
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::uint32_t layer,
    std::span<const float> hidden,
    std::span<const float> router_weight,
    QwenRouterConfig router_config) noexcept {
  RoutedMoeResult result;

  if (!context.valid()) {
    result.diagnostic =
        "Vulkan-accumulated routed MoE requires a valid Vulkan context";
    return result;
  }

  try {
    result.routing =
        route_qwen_top_k_cpu(hidden, router_weight, router_config);

    if (result.routing.picks.empty()) {
      result.diagnostic =
          "Vulkan-accumulated routed MoE selected no experts";
      return result;
    }

    std::string diagnostic;
    auto shared_input =
        VulkanFloatBuffer::create(context, hidden.size(), &diagnostic);
    if (!shared_input.has_value()) {
      result.diagnostic =
          "routed MoE shared input allocation failed: " + diagnostic;
      return result;
    }
    if (!shared_input->upload(hidden, &diagnostic)) {
      result.diagnostic =
          "routed MoE shared input upload failed: " + diagnostic;
      return result;
    }

    const auto& first_pick = result.routing.picks.front();
    auto* first = expert_cache.get(
        layer, first_pick.expert, &diagnostic);
    if (first == nullptr) {
      result.diagnostic =
          "routed MoE first resident expert load failed: " + diagnostic;
      return result;
    }
    if (first->input_dim() != hidden.size() ||
        first->output_dim() == 0U) {
      result.diagnostic =
          "routed MoE first expert geometry mismatch";
      return result;
    }

    auto accumulator =
        VulkanFloatBuffer::create(
            context, first->output_dim(), &diagnostic);
    if (!accumulator.has_value()) {
      result.diagnostic =
          "routed MoE accumulator allocation failed: " + diagnostic;
      return result;
    }

    std::vector<float> zeros(first->output_dim(), 0.0F);
    if (!accumulator->upload(zeros, &diagnostic)) {
      result.diagnostic =
          "routed MoE accumulator zero upload failed: " + diagnostic;
      return result;
    }

    auto execute_and_accumulate =
        [&](VulkanResidentExpert& resident,
            const RoutedExpert& pick) -> bool {
      if (!std::isfinite(pick.weight)) {
        result.diagnostic =
            "routed MoE encountered a non-finite routing weight";
        return false;
      }
      if (resident.input_dim() != shared_input->size() ||
          resident.output_dim() != accumulator->size()) {
        result.diagnostic =
            "routed MoE resident expert geometry mismatch";
        return false;
      }

      const auto dispatch =
          resident.run_from_buffer(context, *shared_input);
      if (!dispatch.executed) {
        result.diagnostic =
            "routed MoE expert " + std::to_string(pick.expert) +
            " execution failed: " + dispatch.diagnostic;
        return false;
      }

      const auto* source = resident.output_buffer();
      if (source == nullptr) {
        result.diagnostic =
            "routed MoE resident expert exposed no output buffer";
        return false;
      }

      const auto accumulate = run_vulkan_weighted_accumulate(
          context, *source, pick.weight, *accumulator);
      if (!accumulate.executed) {
        result.diagnostic =
            "routed MoE expert " + std::to_string(pick.expert) +
            " accumulation failed: " + accumulate.diagnostic;
        return false;
      }
      return true;
    };

    if (!execute_and_accumulate(*first, first_pick)) {
      return result;
    }

    for (std::size_t i = 1; i < result.routing.picks.size(); ++i) {
      const auto& pick = result.routing.picks[i];
      auto* resident =
          expert_cache.get(layer, pick.expert, &diagnostic);
      if (resident == nullptr) {
        result.diagnostic =
            "routed MoE resident expert " +
            std::to_string(pick.expert) +
            " load failed: " + diagnostic;
        return result;
      }

      if (!execute_and_accumulate(*resident, pick)) {
        return result;
      }
    }

    const auto values = accumulator->download(&diagnostic);
    if (!values.has_value()) {
      result.diagnostic =
          "routed MoE final accumulator download failed: " + diagnostic;
      return result;
    }

    result.executed = true;
    result.values = *values;
    result.diagnostic =
        "routed MoE used one hidden upload and Vulkan weighted accumulation";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("Vulkan-accumulated routed MoE exception: ") + e.what();
    result.values.clear();
    return result;
  } catch (...) {
    result.diagnostic =
        "Vulkan-accumulated routed MoE encountered an unknown exception";
    result.values.clear();
    return result;
  }
}

}  // namespace orbi::streammoe
