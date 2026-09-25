#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"

namespace orbi::streammoe {

struct VulkanQ4DequantResult {
  bool executed{};
  std::vector<float> values;
  std::string diagnostic;
};

/// Dequantizes MLX-style affine Q4 rows on Vulkan.
///
/// One uint32 stores eight 4-bit values. packed_cols is measured in uint32
/// words per row, so the logical column count is packed_cols * 8.
[[nodiscard]] VulkanQ4DequantResult run_vulkan_q4_dequant(
    VulkanComputeContext& context,
    std::span<const std::uint32_t> packed,
    std::size_t rows,
    std::size_t packed_cols,
    std::span<const float> scales,
    std::span<const float> biases,
    std::size_t group_size) noexcept;

}  // namespace orbi::streammoe
