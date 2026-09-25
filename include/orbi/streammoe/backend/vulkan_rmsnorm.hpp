#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"

namespace orbi::streammoe {

struct VulkanRmsNormResult {
  bool executed{};
  std::vector<float> values;
  std::string diagnostic;
};

/// Executes correctness-first RMSNorm on the active Vulkan compute context.
///
/// Semantics match cpu::rms_norm_inplace:
///   y = x * rsqrt(mean(x^2) + eps) * weight
///
/// Input is row-major [rows, dim]. The implementation intentionally favors
/// portability and numerical validation over throughput at this gate.
[[nodiscard]] VulkanRmsNormResult run_vulkan_rms_norm(
    VulkanComputeContext& context,
    std::span<const float> input,
    std::size_t rows,
    std::size_t dim,
    std::span<const float> weight,
    float eps) noexcept;

}  // namespace orbi::streammoe
