#pragma once

#include <cstdint>
#include <string>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"

namespace orbi::streammoe {

struct VulkanRoundTripResult {
  bool executed{};
  std::uint32_t initial_value{};
  std::uint32_t output_value{};
  std::string diagnostic;
};

/// Executes the first real ORBI StreamMoE Vulkan compute dispatch.
///
/// The shader writes a fixed sentinel value into one host-visible storage
/// buffer. This intentionally exercises descriptor binding, a compute
/// pipeline, command recording/submission, synchronization, and readback
/// without introducing model math yet.
[[nodiscard]] VulkanRoundTripResult run_vulkan_compute_roundtrip(
    VulkanComputeContext& context,
    std::uint32_t initial_value = 0U) noexcept;

inline constexpr std::uint32_t kVulkanRoundTripSentinel = 0x12345678U;

}  // namespace orbi::streammoe
