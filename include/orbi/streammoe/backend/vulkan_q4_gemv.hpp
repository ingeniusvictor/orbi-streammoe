#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"

namespace orbi::streammoe {

struct VulkanQ4GemvResult {
  bool executed{};
  std::vector<float> values;
  std::string diagnostic;
};

/// Computes y = W*x directly from MLX-style affine Q4 packed rows.
///
/// One uint32 stores eight 4-bit values. packed_cols is measured in uint32
/// words per output row, so the logical input dimension is packed_cols * 8.
[[nodiscard]] VulkanQ4GemvResult run_vulkan_q4_gemv(
    VulkanComputeContext& context,
    std::span<const std::uint32_t> packed,
    std::size_t out_dim,
    std::size_t packed_cols,
    std::span<const float> scales,
    std::span<const float> biases,
    std::size_t group_size,
    std::span<const float> x) noexcept;


/// Computes y = W*x while keeping x/y in reusable Vulkan float buffers.
/// Packed Q4 weights and their small affine metadata remain caller-owned host
/// spans for this gate and are staged internally for the dispatch.
[[nodiscard]] VulkanBufferDispatchResult run_vulkan_q4_gemv_buffers(
    VulkanComputeContext& context,
    std::span<const std::uint32_t> packed,
    std::size_t out_dim,
    std::size_t packed_cols,
    std::span<const float> scales,
    std::span<const float> biases,
    std::size_t group_size,
    const VulkanFloatBuffer& x,
    VulkanFloatBuffer& y) noexcept;

}  // namespace orbi::streammoe
