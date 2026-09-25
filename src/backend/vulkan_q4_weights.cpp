#include "orbi/streammoe/backend/vulkan_q4_weights.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <volk.h>

namespace orbi::streammoe {
namespace {

std::string vk_result_string(VkResult result) {
  return "VkResult(" + std::to_string(static_cast<int>(result)) + ")";
}

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
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

}  // namespace

struct VulkanQ4ProjectionWeights::Impl {
  VkDevice device{VK_NULL_HANDLE};
  VkBuffer packed_buffer{VK_NULL_HANDLE};
  VkDeviceMemory packed_memory{VK_NULL_HANDLE};
  VkMemoryPropertyFlags packed_memory_flags{};
  std::size_t packed_bytes{};

  std::size_t out_dim{};
  std::size_t packed_cols{};
  std::size_t in_dim{};
  std::size_t group_size{};

  std::optional<VulkanFloatBuffer> scales;
  std::optional<VulkanFloatBuffer> biases;

  ~Impl() {
    if (device == VK_NULL_HANDLE) return;
    if (packed_buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, packed_buffer, nullptr);
    }
    if (packed_memory != VK_NULL_HANDLE) {
      vkFreeMemory(device, packed_memory, nullptr);
    }
  }
};

VulkanQ4ProjectionWeights::VulkanQ4ProjectionWeights() = default;
VulkanQ4ProjectionWeights::~VulkanQ4ProjectionWeights() = default;
VulkanQ4ProjectionWeights::VulkanQ4ProjectionWeights(
    VulkanQ4ProjectionWeights&&) noexcept = default;
VulkanQ4ProjectionWeights& VulkanQ4ProjectionWeights::operator=(
    VulkanQ4ProjectionWeights&&) noexcept = default;

VulkanQ4ProjectionWeights::VulkanQ4ProjectionWeights(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<VulkanQ4ProjectionWeights>
VulkanQ4ProjectionWeights::create(
    VulkanComputeContext& context,
    std::span<const std::uint32_t> packed,
    std::size_t out_dim,
    std::size_t packed_cols,
    std::span<const float> scales,
    std::span<const float> biases,
    std::size_t group_size,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(diagnostic, "Vulkan compute context is not valid.");
    return std::nullopt;
  }
  if (out_dim == 0U || packed_cols == 0U || group_size == 0U) {
    set_diagnostic(
        diagnostic,
        "Q4 projection dimensions and group size must be non-zero.");
    return std::nullopt;
  }
  if (packed_cols > std::numeric_limits<std::size_t>::max() / 8U) {
    set_diagnostic(diagnostic, "Q4 projection input dimension overflow.");
    return std::nullopt;
  }

  const auto in_dim = packed_cols * 8U;
  if (in_dim % group_size != 0U) {
    set_diagnostic(
        diagnostic,
        "Q4 projection input dimension must be divisible by group size.");
    return std::nullopt;
  }
  if (out_dim > std::numeric_limits<std::size_t>::max() / packed_cols ||
      packed.size() != out_dim * packed_cols) {
    set_diagnostic(diagnostic, "Q4 projection packed shape mismatch.");
    return std::nullopt;
  }

  const auto groups_per_row = in_dim / group_size;
  if (out_dim > std::numeric_limits<std::size_t>::max() / groups_per_row) {
    set_diagnostic(diagnostic, "Q4 projection metadata shape overflow.");
    return std::nullopt;
  }
  const auto group_values = out_dim * groups_per_row;
  if (scales.size() != group_values || biases.size() != group_values) {
    set_diagnostic(diagnostic, "Q4 projection scale/bias shape mismatch.");
    return std::nullopt;
  }

  try {
    const auto physical_device = reinterpret_cast<VkPhysicalDevice>(
        context.native_physical_device());
    const auto device = reinterpret_cast<VkDevice>(context.native_device());
    if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE) {
      set_diagnostic(
          diagnostic,
          "Vulkan compute context exposed a null device handle.");
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->out_dim = out_dim;
    impl->packed_cols = packed_cols;
    impl->in_dim = in_dim;
    impl->group_size = group_size;
    impl->packed_bytes = packed.size() * sizeof(std::uint32_t);

    VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = static_cast<VkDeviceSize>(impl->packed_bytes),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    VkResult status = vkCreateBuffer(
        device, &buffer_ci, nullptr, &impl->packed_buffer);
    if (status != VK_SUCCESS) {
      set_diagnostic(
          diagnostic,
          "vkCreateBuffer(packed Q4) failed: " + vk_result_string(status));
      return std::nullopt;
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, impl->packed_buffer, &requirements);

    const auto memory_type = find_host_memory_type(
        physical_device,
        requirements.memoryTypeBits,
        &impl->packed_memory_flags);
    if (memory_type == std::numeric_limits<std::uint32_t>::max()) {
      set_diagnostic(
          diagnostic,
          "No host-visible memory type for persistent Q4 weights.");
      return std::nullopt;
    }

    VkMemoryAllocateInfo memory_ai{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = requirements.size,
        .memoryTypeIndex = memory_type,
    };

    status = vkAllocateMemory(
        device, &memory_ai, nullptr, &impl->packed_memory);
    if (status != VK_SUCCESS) {
      set_diagnostic(
          diagnostic,
          "vkAllocateMemory(packed Q4) failed: " + vk_result_string(status));
      return std::nullopt;
    }

    status = vkBindBufferMemory(
        device, impl->packed_buffer, impl->packed_memory, 0);
    if (status != VK_SUCCESS) {
      set_diagnostic(
          diagnostic,
          "vkBindBufferMemory(packed Q4) failed: " +
              vk_result_string(status));
      return std::nullopt;
    }

    void* mapped = nullptr;
    status = vkMapMemory(
        device,
        impl->packed_memory,
        0,
        VK_WHOLE_SIZE,
        0,
        &mapped);
    if (status != VK_SUCCESS || mapped == nullptr) {
      set_diagnostic(
          diagnostic,
          "vkMapMemory(packed Q4) failed: " + vk_result_string(status));
      return std::nullopt;
    }

    std::memcpy(mapped, packed.data(), impl->packed_bytes);

    if ((impl->packed_memory_flags &
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0U) {
      VkMappedMemoryRange range{
          .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
          .pNext = nullptr,
          .memory = impl->packed_memory,
          .offset = 0,
          .size = VK_WHOLE_SIZE,
      };
      status = vkFlushMappedMemoryRanges(device, 1, &range);
      if (status != VK_SUCCESS) {
        vkUnmapMemory(device, impl->packed_memory);
        set_diagnostic(
            diagnostic,
            "vkFlushMappedMemoryRanges(packed Q4) failed: " +
                vk_result_string(status));
        return std::nullopt;
      }
    }
    vkUnmapMemory(device, impl->packed_memory);

    impl->scales =
        VulkanFloatBuffer::create(context, scales.size(), diagnostic);
    if (!impl->scales.has_value() ||
        !impl->scales->upload(scales, diagnostic)) {
      return std::nullopt;
    }

    impl->biases =
        VulkanFloatBuffer::create(context, biases.size(), diagnostic);
    if (!impl->biases.has_value() ||
        !impl->biases->upload(biases, diagnostic)) {
      return std::nullopt;
    }

    set_diagnostic(
        diagnostic,
        "Persistent Vulkan Q4 projection weights uploaded.");
    return VulkanQ4ProjectionWeights(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("Vulkan Q4 projection upload exception: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "Vulkan Q4 projection upload encountered an unknown exception.");
    return std::nullopt;
  }
}

bool VulkanQ4ProjectionWeights::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->device != VK_NULL_HANDLE &&
         impl_->packed_buffer != VK_NULL_HANDLE &&
         impl_->packed_memory != VK_NULL_HANDLE &&
         impl_->scales.has_value() &&
         impl_->biases.has_value() &&
         impl_->scales->valid() &&
         impl_->biases->valid();
}

std::size_t VulkanQ4ProjectionWeights::out_dim() const noexcept {
  return impl_ != nullptr ? impl_->out_dim : 0U;
}

std::size_t VulkanQ4ProjectionWeights::packed_cols() const noexcept {
  return impl_ != nullptr ? impl_->packed_cols : 0U;
}

std::size_t VulkanQ4ProjectionWeights::in_dim() const noexcept {
  return impl_ != nullptr ? impl_->in_dim : 0U;
}

std::size_t VulkanQ4ProjectionWeights::group_size() const noexcept {
  return impl_ != nullptr ? impl_->group_size : 0U;
}

std::uintptr_t VulkanQ4ProjectionWeights::native_device() const noexcept {
  return impl_ != nullptr
      ? reinterpret_cast<std::uintptr_t>(impl_->device)
      : 0U;
}

std::uintptr_t VulkanQ4ProjectionWeights::native_packed_buffer() const noexcept {
  return impl_ != nullptr
      ? reinterpret_cast<std::uintptr_t>(impl_->packed_buffer)
      : 0U;
}

std::size_t VulkanQ4ProjectionWeights::packed_size_bytes() const noexcept {
  return impl_ != nullptr ? impl_->packed_bytes : 0U;
}

const VulkanFloatBuffer&
VulkanQ4ProjectionWeights::scales_buffer() const noexcept {
  static const VulkanFloatBuffer empty;
  return impl_ != nullptr && impl_->scales.has_value()
      ? *impl_->scales
      : empty;
}

const VulkanFloatBuffer&
VulkanQ4ProjectionWeights::biases_buffer() const noexcept {
  static const VulkanFloatBuffer empty;
  return impl_ != nullptr && impl_->biases.has_value()
      ? *impl_->biases
      : empty;
}

}  // namespace orbi::streammoe
