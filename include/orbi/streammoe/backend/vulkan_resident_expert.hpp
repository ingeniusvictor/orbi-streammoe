#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/container/qpack.hpp"

namespace orbi::streammoe {

struct VulkanResidentExpertResult {
  bool executed{};
  std::vector<float> values;
  std::string diagnostic;
};

/// One complete streamed MoE expert with gate/up/down Q4 projections resident
/// in reusable Vulkan buffers.
///
/// The source ExpertCacheEntry is only required during creation. Packed weights
/// and affine metadata are uploaded into Vulkan-owned resources before create()
/// returns.
class VulkanResidentExpert {
 public:
  VulkanResidentExpert();
  ~VulkanResidentExpert();

  VulkanResidentExpert(const VulkanResidentExpert&) = delete;
  VulkanResidentExpert& operator=(const VulkanResidentExpert&) = delete;

  VulkanResidentExpert(VulkanResidentExpert&&) noexcept;
  VulkanResidentExpert& operator=(VulkanResidentExpert&&) noexcept;

  [[nodiscard]] static std::optional<VulkanResidentExpert> create(
      VulkanComputeContext& context,
      const QpackReader& reader,
      const ExpertCacheEntry& entry,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t input_dim() const noexcept;
  [[nodiscard]] std::size_t intermediate_dim() const noexcept;
  [[nodiscard]] std::size_t output_dim() const noexcept;
  [[nodiscard]] std::size_t packed_weight_bytes() const noexcept;
  [[nodiscard]] std::uintptr_t native_device() const noexcept;

  /// Executes gate/up -> SwiGLU -> down using resident projection weights and
  /// reusable activation buffers. Only input upload and final output download
  /// cross the host/device boundary for each call.
  [[nodiscard]] VulkanResidentExpertResult run(
      VulkanComputeContext& context,
      std::span<const float> x) noexcept;

 private:
  struct Impl;
  explicit VulkanResidentExpert(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
