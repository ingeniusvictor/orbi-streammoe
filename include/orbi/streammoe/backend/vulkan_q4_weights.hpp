#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"

namespace orbi::streammoe {

/// Persistent Vulkan-side affine-Q4 projection weights.
///
/// Packed Q4 words plus decoded affine scale/bias vectors are uploaded once and
/// can be reused across multiple token/expert executions.
class VulkanQ4ProjectionWeights {
 public:
  VulkanQ4ProjectionWeights();
  ~VulkanQ4ProjectionWeights();

  VulkanQ4ProjectionWeights(const VulkanQ4ProjectionWeights&) = delete;
  VulkanQ4ProjectionWeights& operator=(const VulkanQ4ProjectionWeights&) = delete;

  VulkanQ4ProjectionWeights(VulkanQ4ProjectionWeights&&) noexcept;
  VulkanQ4ProjectionWeights& operator=(VulkanQ4ProjectionWeights&&) noexcept;

  [[nodiscard]] static std::optional<VulkanQ4ProjectionWeights> create(
      VulkanComputeContext& context,
      std::span<const std::uint32_t> packed,
      std::size_t out_dim,
      std::size_t packed_cols,
      std::span<const float> scales,
      std::span<const float> biases,
      std::size_t group_size,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t out_dim() const noexcept;
  [[nodiscard]] std::size_t packed_cols() const noexcept;
  [[nodiscard]] std::size_t in_dim() const noexcept;
  [[nodiscard]] std::size_t group_size() const noexcept;

  [[nodiscard]] std::uintptr_t native_device() const noexcept;
  [[nodiscard]] std::uintptr_t native_packed_buffer() const noexcept;
  [[nodiscard]] std::size_t packed_size_bytes() const noexcept;

  [[nodiscard]] const VulkanFloatBuffer& scales_buffer() const noexcept;
  [[nodiscard]] const VulkanFloatBuffer& biases_buffer() const noexcept;

 private:
  struct Impl;
  explicit VulkanQ4ProjectionWeights(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
