#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_q4_gemv.hpp"
#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/container/qpack.hpp"

namespace orbi::streammoe {

struct QpackExpertQ4View {
  std::span<const std::uint32_t> packed;
  std::span<const float> scales;
  std::span<const float> biases;
  std::size_t out_dim{};
  std::size_t packed_cols{};
  std::size_t group_size{};
};

[[nodiscard]] QpackExpertQ4View bind_qpack_q4_projection(
    const QpackReader& reader,
    const ExpertCacheEntry& entry,
    std::string_view projection);

[[nodiscard]] VulkanQ4GemvResult run_vulkan_qpack_q4_projection(
    VulkanComputeContext& context,
    const QpackReader& reader,
    const ExpertCacheEntry& entry,
    std::string_view projection,
    std::span<const float> x) noexcept;

}  // namespace orbi::streammoe
