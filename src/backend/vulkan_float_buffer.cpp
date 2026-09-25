#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <volk.h>

namespace orbi::streammoe {
namespace {

std::string vk_result_string(VkResult result) {
  return "VkResult(" + std::to_string(static_cast<int>(result)) + ")";
}

std::uint32_t find_host_memory_type(
    VkPhysicalDevice physical_device,
    std::uint32_t type_bits,
    VkMemoryPropertyFlags* selected_flags) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

  std::uint32_t fallback = std::numeric_limits<std::uint32_t>::max();

  for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    if ((type_bits & (1U << i)) == 0U) continue;

    const auto flags = properties.memoryTypes[i].propertyFlags;
    if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0U) continue;

    if (fallback == std::numeric_limits<std::uint32_t>::max()) {
      fallback = i;
    }

    if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U) {
      if (selected_flags != nullptr) *selected_flags = flags;
      return i;
    }
  }

  if (fallback != std::numeric_limits<std::uint32_t>::max()) {
    if (selected_flags != nullptr) {
      *selected_flags = properties.memoryTypes[fallback].propertyFlags;
    }
    return fallback;
  }

  return std::numeric_limits<std::uint32_t>::max();
}

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
}

}  // namespace

struct VulkanFloatBuffer::Impl {
  VkDevice device{VK_NULL_HANDLE};
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  VkMemoryPropertyFlags memory_flags{};
  std::size_t element_count{};
  std::size_t byte_count{};

  ~Impl() {
    if (device == VK_NULL_HANDLE) return;
    if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, buffer, nullptr);
    if (memory != VK_NULL_HANDLE) vkFreeMemory(device, memory, nullptr);
  }
};

VulkanFloatBuffer::VulkanFloatBuffer() = default;
VulkanFloatBuffer::~VulkanFloatBuffer() = default;
VulkanFloatBuffer::VulkanFloatBuffer(VulkanFloatBuffer&&) noexcept = default;
VulkanFloatBuffer& VulkanFloatBuffer::operator=(
    VulkanFloatBuffer&&) noexcept = default;

VulkanFloatBuffer::VulkanFloatBuffer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<VulkanFloatBuffer> VulkanFloatBuffer::create(
    VulkanComputeContext& context,
    std::size_t element_count,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(diagnostic, "Vulkan compute context is not valid.");
    return std::nullopt;
  }
  if (element_count == 0U) {
    set_diagnostic(diagnostic, "Vulkan float buffer element count must be non-zero.");
    return std::nullopt;
  }
  if (element_count >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    set_diagnostic(diagnostic, "Vulkan float buffer byte size overflow.");
    return std::nullopt;
  }

  try {
    const auto physical_device = reinterpret_cast<VkPhysicalDevice>(
        context.native_physical_device());
    const auto device =
        reinterpret_cast<VkDevice>(context.native_device());

    if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE) {
      set_diagnostic(
          diagnostic,
          "Vulkan compute context exposed a null device handle.");
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->element_count = element_count;
    impl->byte_count = element_count * sizeof(float);

    VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = static_cast<VkDeviceSize>(impl->byte_count),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    VkResult status =
        vkCreateBuffer(device, &buffer_ci, nullptr, &impl->buffer);
    if (status != VK_SUCCESS) {
      set_diagnostic(
          diagnostic,
          "vkCreateBuffer failed: " + vk_result_string(status));
      return std::nullopt;
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, impl->buffer, &requirements);

    const auto memory_type = find_host_memory_type(
        physical_device,
        requirements.memoryTypeBits,
        &impl->memory_flags);

    if (memory_type == std::numeric_limits<std::uint32_t>::max()) {
      set_diagnostic(
          diagnostic,
          "No host-visible Vulkan memory type is available for float buffer.");
      return std::nullopt;
    }

    VkMemoryAllocateInfo memory_ai{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = requirements.size,
        .memoryTypeIndex = memory_type,
    };

    status = vkAllocateMemory(
        device, &memory_ai, nullptr, &impl->memory);
    if (status != VK_SUCCESS) {
      set_diagnostic(
          diagnostic,
          "vkAllocateMemory failed: " + vk_result_string(status));
      return std::nullopt;
    }

    status = vkBindBufferMemory(
        device, impl->buffer, impl->memory, 0);
    if (status != VK_SUCCESS) {
      set_diagnostic(
          diagnostic,
          "vkBindBufferMemory failed: " + vk_result_string(status));
      return std::nullopt;
    }

    set_diagnostic(
        diagnostic,
        "Reusable Vulkan float buffer created.");
    return VulkanFloatBuffer(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("Vulkan float buffer exception: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "Vulkan float buffer encountered an unknown exception.");
    return std::nullopt;
  }
}

bool VulkanFloatBuffer::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->device != VK_NULL_HANDLE &&
         impl_->buffer != VK_NULL_HANDLE &&
         impl_->memory != VK_NULL_HANDLE &&
         impl_->element_count != 0U;
}

std::size_t VulkanFloatBuffer::size() const noexcept {
  return impl_ != nullptr ? impl_->element_count : 0U;
}

std::size_t VulkanFloatBuffer::size_bytes() const noexcept {
  return impl_ != nullptr ? impl_->byte_count : 0U;
}

bool VulkanFloatBuffer::upload(
    std::span<const float> values,
    std::string* diagnostic) noexcept {
  if (!valid()) {
    set_diagnostic(diagnostic, "Vulkan float buffer is not valid.");
    return false;
  }
  if (values.size() != impl_->element_count) {
    set_diagnostic(
        diagnostic,
        "Upload element count does not match Vulkan float buffer.");
    return false;
  }

  void* mapped = nullptr;
  VkResult status = vkMapMemory(
      impl_->device,
      impl_->memory,
      0,
      VK_WHOLE_SIZE,
      0,
      &mapped);

  if (status != VK_SUCCESS || mapped == nullptr) {
    set_diagnostic(
        diagnostic,
        "vkMapMemory(upload) failed: " + vk_result_string(status));
    return false;
  }

  std::memcpy(mapped, values.data(), impl_->byte_count);

  if ((impl_->memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0U) {
    VkMappedMemoryRange range{
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext = nullptr,
        .memory = impl_->memory,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };

    status = vkFlushMappedMemoryRanges(
        impl_->device, 1, &range);
    if (status != VK_SUCCESS) {
      vkUnmapMemory(impl_->device, impl_->memory);
      set_diagnostic(
          diagnostic,
          "vkFlushMappedMemoryRanges failed: " +
              vk_result_string(status));
      return false;
    }
  }

  vkUnmapMemory(impl_->device, impl_->memory);
  set_diagnostic(diagnostic, "Vulkan float buffer upload completed.");
  return true;
}

std::optional<std::vector<float>> VulkanFloatBuffer::download(
    std::string* diagnostic) const noexcept {
  if (!valid()) {
    set_diagnostic(diagnostic, "Vulkan float buffer is not valid.");
    return std::nullopt;
  }

  void* mapped = nullptr;
  VkResult status = vkMapMemory(
      impl_->device,
      impl_->memory,
      0,
      VK_WHOLE_SIZE,
      0,
      &mapped);

  if (status != VK_SUCCESS || mapped == nullptr) {
    set_diagnostic(
        diagnostic,
        "vkMapMemory(download) failed: " + vk_result_string(status));
    return std::nullopt;
  }

  if ((impl_->memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0U) {
    VkMappedMemoryRange range{
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext = nullptr,
        .memory = impl_->memory,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };

    status = vkInvalidateMappedMemoryRanges(
        impl_->device, 1, &range);
    if (status != VK_SUCCESS) {
      vkUnmapMemory(impl_->device, impl_->memory);
      set_diagnostic(
          diagnostic,
          "vkInvalidateMappedMemoryRanges failed: " +
              vk_result_string(status));
      return std::nullopt;
    }
  }

  std::vector<float> values(impl_->element_count);
  std::memcpy(values.data(), mapped, impl_->byte_count);
  vkUnmapMemory(impl_->device, impl_->memory);

  set_diagnostic(diagnostic, "Vulkan float buffer download completed.");
  return values;
}

std::uintptr_t VulkanFloatBuffer::native_buffer() const noexcept {
  return impl_ != nullptr
      ? reinterpret_cast<std::uintptr_t>(impl_->buffer)
      : 0U;
}

}  // namespace orbi::streammoe
