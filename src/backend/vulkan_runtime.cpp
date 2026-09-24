#include "orbi/streammoe/backend/vulkan_runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <volk.h>

namespace orbi::streammoe {
namespace {

std::string vk_result_string(VkResult result) {
  return "VkResult(" + std::to_string(static_cast<int>(result)) + ")";
}

std::optional<std::uint32_t> first_compute_queue_family(
    VkPhysicalDevice device) {
  std::uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  if (count == 0) {
    return std::nullopt;
  }

  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

  for (std::uint32_t index = 0; index < count; ++index) {
    if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U &&
        families[index].queueCount > 0) {
      return index;
    }
  }

  return std::nullopt;
}

}  // namespace

VulkanProbeResult probe_vulkan_runtime() noexcept {
  VulkanProbeResult result;

  try {
    const VkResult init = volkInitialize();
    if (init != VK_SUCCESS) {
      result.diagnostic =
          "Vulkan loader unavailable: " + vk_result_string(init);
      return result;
    }

    result.loader_available = true;
    result.loader_api_version = volkGetInstanceVersion();
    if (result.loader_api_version == 0) {
      result.loader_api_version = VK_API_VERSION_1_0;
    }

    const auto requested_version =
        std::min(result.loader_api_version, VK_API_VERSION_1_3);

    VkApplicationInfo application{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "ORBI StreamMoE",
        .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .pEngineName = "ORBI StreamMoE",
        .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .apiVersion = requested_version,
    };

    VkInstanceCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &application,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
    };

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult create = vkCreateInstance(
        &create_info,
        nullptr,
        &instance);

    if (create != VK_SUCCESS) {
      result.diagnostic =
          "Vulkan loader found but instance creation failed: " +
          vk_result_string(create);
      volkFinalize();
      return result;
    }

    result.instance_created = true;
    volkLoadInstance(instance);

    std::uint32_t device_count = 0;
    VkResult enumerate = vkEnumeratePhysicalDevices(
        instance,
        &device_count,
        nullptr);

    if (enumerate != VK_SUCCESS) {
      result.diagnostic =
          "vkEnumeratePhysicalDevices(count) failed: " +
          vk_result_string(enumerate);
      vkDestroyInstance(instance, nullptr);
      volkFinalize();
      return result;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    if (device_count > 0) {
      enumerate = vkEnumeratePhysicalDevices(
          instance,
          &device_count,
          devices.data());

      if (enumerate != VK_SUCCESS &&
          enumerate != VK_INCOMPLETE) {
        result.diagnostic =
            "vkEnumeratePhysicalDevices(data) failed: " +
            vk_result_string(enumerate);
        vkDestroyInstance(instance, nullptr);
        volkFinalize();
        return result;
      }
      devices.resize(device_count);
    }

    result.devices.reserve(devices.size());

    for (const auto device : devices) {
      VkPhysicalDeviceProperties properties{};
      vkGetPhysicalDeviceProperties(device, &properties);

      const auto compute_family =
          first_compute_queue_family(device);

      result.devices.push_back({
          .name = properties.deviceName,
          .vendor_id = properties.vendorID,
          .device_id = properties.deviceID,
          .api_version = properties.apiVersion,
          .device_type = static_cast<std::uint32_t>(properties.deviceType),
          .compute_queue_family = compute_family,
          .max_compute_workgroup_invocations =
              properties.limits.maxComputeWorkGroupInvocations,
          .max_compute_shared_memory_size =
              properties.limits.maxComputeSharedMemorySize,
          .max_storage_buffer_range =
              static_cast<std::uint64_t>(
                  properties.limits.maxStorageBufferRange),
      });
    }

    if (result.devices.empty()) {
      result.diagnostic =
          "Vulkan loader and instance are available, but no physical device was enumerated.";
    } else {
      const auto compute_devices = std::count_if(
          result.devices.begin(),
          result.devices.end(),
          [](const VulkanDeviceInfo& device) {
            return device.compute_queue_family.has_value();
          });

      result.diagnostic =
          "Vulkan probe enumerated " +
          std::to_string(result.devices.size()) +
          " device(s), " +
          std::to_string(compute_devices) +
          " with a compute queue.";
    }

    vkDestroyInstance(instance, nullptr);
    volkFinalize();
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("Vulkan probe exception: ") + e.what();
    volkFinalize();
    return result;
  } catch (...) {
    result.diagnostic = "Vulkan probe encountered an unknown exception.";
    volkFinalize();
    return result;
  }
}

}  // namespace orbi::streammoe
