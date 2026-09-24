#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace orbi::streammoe {

struct VulkanDeviceInfo {
  std::string name;
  std::uint32_t vendor_id{};
  std::uint32_t device_id{};
  std::uint32_t api_version{};
  std::uint32_t device_type{};
  std::optional<std::uint32_t> compute_queue_family;
  std::uint32_t max_compute_workgroup_invocations{};
  std::uint32_t max_compute_shared_memory_size{};
  std::uint64_t max_storage_buffer_range{};
};

struct VulkanProbeResult {
  bool loader_available{};
  std::uint32_t loader_api_version{};
  bool instance_created{};
  std::vector<VulkanDeviceInfo> devices;
  std::string diagnostic;
};

[[nodiscard]] VulkanProbeResult probe_vulkan_runtime() noexcept;

}  // namespace orbi::streammoe
