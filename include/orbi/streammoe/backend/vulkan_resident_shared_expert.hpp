#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"

namespace orbi::streammoe {

struct VulkanSharedExpertProjectionView {
  std::span<const std::uint32_t> packed;
  std::span<const float> scales;
  std::span<const float> biases;
  std::size_t out_dim{};
  std::size_t packed_cols{};
  std::size_t group_size{};
};

struct VulkanSharedExpertWeights {
  VulkanSharedExpertProjectionView gate;
  VulkanSharedExpertProjectionView up;
  VulkanSharedExpertProjectionView down;
  std::span<const float> scalar_gate;
};

/// Always-on Qwen shared expert.
///
/// gate/up/down projections are uploaded once into reusable Vulkan Q4 buffers.
/// The scalar gate vector remains host-resident in this gate because the router
/// path already has the token hidden vector on the host. That avoids an extra
/// device round-trip while preserving exact Qwen semantics.
class VulkanResidentSharedExpert {
 public:
  VulkanResidentSharedExpert();
  ~VulkanResidentSharedExpert();

  VulkanResidentSharedExpert(const VulkanResidentSharedExpert&) = delete;
  VulkanResidentSharedExpert& operator=(const VulkanResidentSharedExpert&) = delete;

  VulkanResidentSharedExpert(VulkanResidentSharedExpert&&) noexcept;
  VulkanResidentSharedExpert& operator=(VulkanResidentSharedExpert&&) noexcept;

  [[nodiscard]] static std::optional<VulkanResidentSharedExpert> create(
      VulkanComputeContext& context,
      VulkanSharedExpertWeights weights,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t input_dim() const noexcept;
  [[nodiscard]] std::size_t intermediate_dim() const noexcept;
  [[nodiscard]] std::size_t output_dim() const noexcept;
  [[nodiscard]] std::size_t accounted_vulkan_bytes() const noexcept;
  [[nodiscard]] std::uintptr_t native_device() const noexcept;

  [[nodiscard]] std::optional<float> scalar_scale(
      std::span<const float> hidden,
      std::string* diagnostic = nullptr) const noexcept;

  [[nodiscard]] VulkanBufferDispatchResult run_from_buffer(
      VulkanComputeContext& context,
      const VulkanFloatBuffer& hidden) noexcept;

  [[nodiscard]] const VulkanFloatBuffer* output_buffer() const noexcept;

 private:
  struct Impl;
  explicit VulkanResidentSharedExpert(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
