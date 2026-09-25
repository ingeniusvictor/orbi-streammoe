#pragma once

#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/container/qpack.hpp"

namespace orbi::streammoe {

struct VulkanQpackExpertMlpResult {
  bool executed{};
  std::vector<float> values;
  std::string diagnostic;
};

[[nodiscard]] VulkanQpackExpertMlpResult run_vulkan_qpack_q4_expert_mlp(
    VulkanComputeContext& context,
    const QpackReader& reader,
    const ExpertCacheEntry& entry,
    std::span<const float> x) noexcept;

}  // namespace orbi::streammoe
