#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace orbi::streammoe {

struct VulkanComputeContextInfo {
  std::string device_name;
  std::uint32_t vendor_id{};
  std::uint32_t device_id{};
  std::uint32_t api_version{};
  std::uint32_t queue_family_index{};
};

class VulkanComputeContext {
 public:
  VulkanComputeContext();
  ~VulkanComputeContext();

  VulkanComputeContext(const VulkanComputeContext&) = delete;
  VulkanComputeContext& operator=(const VulkanComputeContext&) = delete;

  VulkanComputeContext(VulkanComputeContext&&) noexcept;
  VulkanComputeContext& operator=(VulkanComputeContext&&) noexcept;

  [[nodiscard]] static std::optional<VulkanComputeContext> create(
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const VulkanComputeContextInfo& info() const noexcept;

  // Opaque native handles are exposed as integer-sized values so public
  // headers remain independent of Vulkan headers.
  [[nodiscard]] std::uintptr_t native_instance() const noexcept;
  [[nodiscard]] std::uintptr_t native_physical_device() const noexcept;
  [[nodiscard]] std::uintptr_t native_device() const noexcept;
  [[nodiscard]] std::uintptr_t native_queue() const noexcept;
  [[nodiscard]] std::uintptr_t native_command_pool() const noexcept;

 private:
  struct Impl;
  explicit VulkanComputeContext(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
