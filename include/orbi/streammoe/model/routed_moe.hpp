#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/backend/vulkan_resident_shared_expert.hpp"
#include "orbi/streammoe/model/qwen_router.hpp"

namespace orbi::streammoe {

struct RoutedMoeResult {
  bool executed{};
  QwenRouterResult routing;
  std::vector<float> values;
  std::string diagnostic;
};

struct QwenSparseMoeResult {
  bool executed{};
  QwenRouterResult routing;
  float shared_gate{};
  std::vector<float> values;
  std::string diagnostic;
};

/// Correctness-first sparse routed-MoE execution.
///
/// Routing remains on the CPU oracle path. Selected expert MLPs execute through
/// VulkanResidentExpertCache. Each expert result is downloaded, multiplied by
/// its routing weight, and accumulated on the host.
///
/// Shared-expert execution is intentionally excluded from OSM-23.
[[nodiscard]] RoutedMoeResult run_weighted_routed_moe(
    VulkanResidentExpertCache& expert_cache,
    std::uint32_t layer,
    std::span<const float> hidden,
    std::span<const float> router_weight,
    QwenRouterConfig router_config) noexcept;

/// Optimized routed path: upload hidden once, execute selected experts from the
/// shared Vulkan input, accumulate weighted outputs on Vulkan, then download
/// only the final routed result.
[[nodiscard]] RoutedMoeResult run_weighted_routed_moe_vulkan_accum(
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::uint32_t layer,
    std::span<const float> hidden,
    std::span<const float> router_weight,
    QwenRouterConfig router_config) noexcept;

/// Complete Qwen sparse-MoE semantic path for one token:
/// routed Top-K weighted sum + sigmoid-gated always-on shared expert.
/// The hidden vector is uploaded once and only the final combined output is
/// downloaded.
[[nodiscard]] QwenSparseMoeResult run_qwen_sparse_moe_vulkan_accum(
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    VulkanResidentSharedExpert& shared_expert,
    std::uint32_t layer,
    std::span<const float> hidden,
    std::span<const float> router_weight,
    QwenRouterConfig router_config) noexcept;

}  // namespace orbi::streammoe
