#pragma once

#include <string>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"

namespace orbi::streammoe {

/// In-place device-side accumulation:
/// accumulator[i] += weight * source[i]
[[nodiscard]] VulkanBufferDispatchResult run_vulkan_weighted_accumulate(
    VulkanComputeContext& context,
    const VulkanFloatBuffer& source,
    float weight,
    VulkanFloatBuffer& accumulator) noexcept;

}  // namespace orbi::streammoe
