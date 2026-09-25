#pragma once

#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"

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

}  // namespace orbi::streammoe
