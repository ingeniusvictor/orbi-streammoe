#pragma once

#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"

namespace orbi::streammoe {

struct VulkanSwiGluResult {
  bool executed{};
  std::vector<float> values;
  std::string diagnostic;
};

/// Computes SiLU(gate) * up elementwise on Vulkan.
[[nodiscard]] VulkanSwiGluResult run_vulkan_swiglu(
    VulkanComputeContext& context,
    std::span<const float> gate,
    std::span<const float> up) noexcept;


/// Computes SiLU(gate) * up from persistent Vulkan buffers into output.
[[nodiscard]] VulkanBufferDispatchResult run_vulkan_swiglu_buffers(
    VulkanComputeContext& context,
    const VulkanFloatBuffer& gate,
    const VulkanFloatBuffer& up,
    VulkanFloatBuffer& output) noexcept;

}  // namespace orbi::streammoe
