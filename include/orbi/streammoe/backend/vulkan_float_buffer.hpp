#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"

namespace orbi::streammoe {

struct VulkanBufferDispatchResult {
  bool executed{};
  std::string diagnostic;
};

/// Reusable host-visible Vulkan storage buffer for float activations.
///
/// The VulkanComputeContext used at creation must outlive this buffer.
class VulkanFloatBuffer {
 public:
  VulkanFloatBuffer();
  ~VulkanFloatBuffer();

  VulkanFloatBuffer(const VulkanFloatBuffer&) = delete;
  VulkanFloatBuffer& operator=(const VulkanFloatBuffer&) = delete;

  VulkanFloatBuffer(VulkanFloatBuffer&&) noexcept;
  VulkanFloatBuffer& operator=(VulkanFloatBuffer&&) noexcept;

  [[nodiscard]] static std::optional<VulkanFloatBuffer> create(
      VulkanComputeContext& context,
      std::size_t element_count,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t size_bytes() const noexcept;

  [[nodiscard]] bool upload(
      std::span<const float> values,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] std::optional<std::vector<float>> download(
      std::string* diagnostic = nullptr) const noexcept;

  [[nodiscard]] std::uintptr_t native_buffer() const noexcept;
  [[nodiscard]] std::uintptr_t native_device() const noexcept;

 private:
  struct Impl;
  explicit VulkanFloatBuffer(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
