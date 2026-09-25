#include "orbi/streammoe/backend/vulkan_compute_context.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <volk.h>

namespace orbi::streammoe {
namespace {

std::string vk_result_string(VkResult result) {
  return "VkResult(" + std::to_string(static_cast<int>(result)) + ")";
}

struct Candidate {
  VkPhysicalDevice device{VK_NULL_HANDLE};
  VkPhysicalDeviceProperties properties{};
  std::uint32_t queue_family{};
  int score{};
};

std::optional<std::uint32_t> compute_queue_family(VkPhysicalDevice device) {
  std::uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  if (count == 0) return std::nullopt;

  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

  std::optional<std::uint32_t> fallback;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (families[i].queueCount == 0 ||
        (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0U) {
      continue;
    }

    // Prefer a compute-only queue when the driver exposes one.
    if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0U) {
      return i;
    }
    if (!fallback.has_value()) fallback = i;
  }
  return fallback;
}

int device_score(const VkPhysicalDeviceProperties& properties) {
  switch (properties.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return 400;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return 300;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return 200;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return 100;
    default:
      return 0;
  }
}

}  // namespace

struct VulkanComputeContext::Impl {
  VkInstance instance{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device{VK_NULL_HANDLE};
  VkDevice device{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};
  VkCommandPool command_pool{VK_NULL_HANDLE};
  VulkanComputeContextInfo info{};

  ~Impl() {
    if (device != VK_NULL_HANDLE) {
      if (command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, command_pool, nullptr);
      }
      vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) {
      vkDestroyInstance(instance, nullptr);
    }
    volkFinalize();
  }
};

VulkanComputeContext::VulkanComputeContext() = default;
VulkanComputeContext::~VulkanComputeContext() = default;
VulkanComputeContext::VulkanComputeContext(
    VulkanComputeContext&&) noexcept = default;
VulkanComputeContext& VulkanComputeContext::operator=(
    VulkanComputeContext&&) noexcept = default;

VulkanComputeContext::VulkanComputeContext(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<VulkanComputeContext> VulkanComputeContext::create(
    std::string* diagnostic) noexcept {
  auto set_diagnostic = [diagnostic](std::string message) {
    if (diagnostic != nullptr) *diagnostic = std::move(message);
  };

  try {
    const VkResult init = volkInitialize();
    if (init != VK_SUCCESS) {
      set_diagnostic("Vulkan loader unavailable: " + vk_result_string(init));
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();

    std::uint32_t loader_version = volkGetInstanceVersion();
    if (loader_version == 0U) loader_version = VK_API_VERSION_1_0;
    const std::uint32_t requested_version =
        std::min(loader_version, VK_API_VERSION_1_3);

    VkApplicationInfo app{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "ORBI StreamMoE",
        .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .pEngineName = "ORBI StreamMoE",
        .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .apiVersion = requested_version,
    };

    VkInstanceCreateInfo instance_ci{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &app,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
    };

    const VkResult create_instance =
        vkCreateInstance(&instance_ci, nullptr, &impl->instance);
    if (create_instance != VK_SUCCESS) {
      set_diagnostic(
          "Vulkan instance creation failed: " +
          vk_result_string(create_instance));
      return std::nullopt;
    }
    volkLoadInstance(impl->instance);

    std::uint32_t physical_count = 0;
    VkResult enumerate = vkEnumeratePhysicalDevices(
        impl->instance, &physical_count, nullptr);
    if (enumerate != VK_SUCCESS || physical_count == 0U) {
      set_diagnostic(
          physical_count == 0U
              ? "No Vulkan physical devices were enumerated."
              : "vkEnumeratePhysicalDevices failed: " +
                    vk_result_string(enumerate));
      return std::nullopt;
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    enumerate = vkEnumeratePhysicalDevices(
        impl->instance, &physical_count, physical_devices.data());
    if (enumerate != VK_SUCCESS && enumerate != VK_INCOMPLETE) {
      set_diagnostic(
          "vkEnumeratePhysicalDevices(data) failed: " +
          vk_result_string(enumerate));
      return std::nullopt;
    }
    physical_devices.resize(physical_count);

    std::optional<Candidate> selected;
    for (const auto physical : physical_devices) {
      const auto family = compute_queue_family(physical);
      if (!family.has_value()) continue;

      VkPhysicalDeviceProperties properties{};
      vkGetPhysicalDeviceProperties(physical, &properties);

      Candidate candidate{
          .device = physical,
          .properties = properties,
          .queue_family = *family,
          .score = device_score(properties),
      };

      if (!selected.has_value() || candidate.score > selected->score) {
        selected = candidate;
      }
    }

    if (!selected.has_value()) {
      set_diagnostic("No Vulkan device exposes a usable compute queue.");
      return std::nullopt;
    }

    impl->physical_device = selected->device;
    impl->info = {
        .device_name = selected->properties.deviceName,
        .vendor_id = selected->properties.vendorID,
        .device_id = selected->properties.deviceID,
        .api_version = selected->properties.apiVersion,
        .queue_family_index = selected->queue_family,
    };

    constexpr float queue_priority = 1.0F;
    VkDeviceQueueCreateInfo queue_ci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = selected->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    VkDeviceCreateInfo device_ci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_ci,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
        .pEnabledFeatures = nullptr,
    };

    const VkResult create_device = vkCreateDevice(
        impl->physical_device, &device_ci, nullptr, &impl->device);
    if (create_device != VK_SUCCESS) {
      set_diagnostic(
          "Vulkan logical device creation failed: " +
          vk_result_string(create_device));
      return std::nullopt;
    }

    volkLoadDevice(impl->device);
    vkGetDeviceQueue(
        impl->device, selected->queue_family, 0, &impl->queue);
    if (impl->queue == VK_NULL_HANDLE) {
      set_diagnostic("Vulkan compute queue acquisition returned null.");
      return std::nullopt;
    }

    VkCommandPoolCreateInfo pool_ci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = selected->queue_family,
    };

    const VkResult create_pool = vkCreateCommandPool(
        impl->device, &pool_ci, nullptr, &impl->command_pool);
    if (create_pool != VK_SUCCESS) {
      set_diagnostic(
          "Vulkan command pool creation failed: " +
          vk_result_string(create_pool));
      return std::nullopt;
    }

    set_diagnostic(
        "Vulkan compute context ready on " + impl->info.device_name + ".");
    return VulkanComputeContext(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        std::string("Vulkan compute context exception: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic("Vulkan compute context encountered an unknown exception.");
    return std::nullopt;
  }
}

bool VulkanComputeContext::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->instance != VK_NULL_HANDLE &&
         impl_->physical_device != VK_NULL_HANDLE &&
         impl_->device != VK_NULL_HANDLE &&
         impl_->queue != VK_NULL_HANDLE &&
         impl_->command_pool != VK_NULL_HANDLE;
}

const VulkanComputeContextInfo& VulkanComputeContext::info() const noexcept {
  static const VulkanComputeContextInfo empty{};
  return impl_ != nullptr ? impl_->info : empty;
}

std::uintptr_t VulkanComputeContext::native_instance() const noexcept {
  return impl_ != nullptr
      ? reinterpret_cast<std::uintptr_t>(impl_->instance)
      : 0U;
}

std::uintptr_t VulkanComputeContext::native_physical_device() const noexcept {
  return impl_ != nullptr
      ? reinterpret_cast<std::uintptr_t>(impl_->physical_device)
      : 0U;
}

std::uintptr_t VulkanComputeContext::native_device() const noexcept {
  return impl_ != nullptr
      ? reinterpret_cast<std::uintptr_t>(impl_->device)
      : 0U;
}

std::uintptr_t VulkanComputeContext::native_queue() const noexcept {
  return impl_ != nullptr
      ? reinterpret_cast<std::uintptr_t>(impl_->queue)
      : 0U;
}

std::uintptr_t VulkanComputeContext::native_command_pool() const noexcept {
  return impl_ != nullptr
      ? reinterpret_cast<std::uintptr_t>(impl_->command_pool)
      : 0U;
}

}  // namespace orbi::streammoe
