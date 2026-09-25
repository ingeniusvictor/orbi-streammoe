#include "orbi/streammoe/backend/vulkan_q4_dequant.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <string>

#include <volk.h>

#include "orbi/streammoe/generated/q4_dequant_spv.hpp"

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

struct HostBuffer {
  VkDevice device{VK_NULL_HANDLE};
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  VkMemoryPropertyFlags memory_flags{};

  HostBuffer() = default;
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;

  ~HostBuffer() {
    if (device == VK_NULL_HANDLE) return;
    if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, buffer, nullptr);
    if (memory != VK_NULL_HANDLE) vkFreeMemory(device, memory, nullptr);
  }
};

bool create_host_storage_buffer(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkDeviceSize size,
    HostBuffer& out,
    std::string& diagnostic) {
  if (size == 0) {
    diagnostic = "Vulkan host buffer size must be non-zero.";
    return false;
  }

  out.device = device;

  VkBufferCreateInfo buffer_ci{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = size,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
  };

  VkResult status =
      vkCreateBuffer(device, &buffer_ci, nullptr, &out.buffer);
  if (status != VK_SUCCESS) {
    diagnostic = "vkCreateBuffer failed: " + vk_result_string(status);
    return false;
  }

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device, out.buffer, &requirements);

  const std::uint32_t memory_type = find_host_memory_type(
      physical_device,
      requirements.memoryTypeBits,
      &out.memory_flags);

  if (memory_type == std::numeric_limits<std::uint32_t>::max()) {
    diagnostic =
        "No host-visible Vulkan memory type is compatible with a Q4 buffer.";
    return false;
  }

  VkMemoryAllocateInfo memory_ai{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = nullptr,
      .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type,
  };

  status = vkAllocateMemory(device, &memory_ai, nullptr, &out.memory);
  if (status != VK_SUCCESS) {
    diagnostic = "vkAllocateMemory failed: " + vk_result_string(status);
    return false;
  }

  status = vkBindBufferMemory(device, out.buffer, out.memory, 0);
  if (status != VK_SUCCESS) {
    diagnostic = "vkBindBufferMemory failed: " + vk_result_string(status);
    return false;
  }

  return true;
}

bool write_host_buffer(
    const HostBuffer& buffer,
    const void* source,
    std::size_t bytes,
    std::string& diagnostic) {
  void* mapped = nullptr;
  VkResult status = vkMapMemory(
      buffer.device,
      buffer.memory,
      0,
      VK_WHOLE_SIZE,
      0,
      &mapped);

  if (status != VK_SUCCESS || mapped == nullptr) {
    diagnostic = "vkMapMemory(write) failed: " + vk_result_string(status);
    return false;
  }

  std::memcpy(mapped, source, bytes);

  if ((buffer.memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0U) {
    VkMappedMemoryRange range{
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext = nullptr,
        .memory = buffer.memory,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };

    status = vkFlushMappedMemoryRanges(buffer.device, 1, &range);
    if (status != VK_SUCCESS) {
      vkUnmapMemory(buffer.device, buffer.memory);
      diagnostic =
          "vkFlushMappedMemoryRanges failed: " + vk_result_string(status);
      return false;
    }
  }

  vkUnmapMemory(buffer.device, buffer.memory);
  return true;
}

bool read_host_buffer(
    const HostBuffer& buffer,
    void* destination,
    std::size_t bytes,
    std::string& diagnostic) {
  void* mapped = nullptr;
  VkResult status = vkMapMemory(
      buffer.device,
      buffer.memory,
      0,
      VK_WHOLE_SIZE,
      0,
      &mapped);

  if (status != VK_SUCCESS || mapped == nullptr) {
    diagnostic = "vkMapMemory(read) failed: " + vk_result_string(status);
    return false;
  }

  if ((buffer.memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0U) {
    VkMappedMemoryRange range{
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext = nullptr,
        .memory = buffer.memory,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };

    status = vkInvalidateMappedMemoryRanges(buffer.device, 1, &range);
    if (status != VK_SUCCESS) {
      vkUnmapMemory(buffer.device, buffer.memory);
      diagnostic =
          "vkInvalidateMappedMemoryRanges failed: " + vk_result_string(status);
      return false;
    }
  }

  std::memcpy(destination, mapped, bytes);
  vkUnmapMemory(buffer.device, buffer.memory);
  return true;
}

struct VulkanObjects {
  VkDevice device{VK_NULL_HANDLE};
  VkCommandPool command_pool{VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptor_set_layout{VK_NULL_HANDLE};
  VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
  VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
  VkShaderModule shader_module{VK_NULL_HANDLE};
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkCommandBuffer command_buffer{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};

  ~VulkanObjects() {
    if (device == VK_NULL_HANDLE) return;

    if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
    if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
    if (shader_module != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device, shader_module, nullptr);
    }
    if (pipeline_layout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    }
    if (descriptor_pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    }
    if (descriptor_set_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    }
    if (command_buffer != VK_NULL_HANDLE &&
        command_pool != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    }
  }
};

struct Q4PushConstants {
  std::uint32_t total_values{};
  std::uint32_t packed_cols{};
  std::uint32_t logical_cols{};
  std::uint32_t group_size{};
};
static_assert(sizeof(Q4PushConstants) == 16);

}  // namespace

VulkanQ4DequantResult run_vulkan_q4_dequant(
    VulkanComputeContext& context,
    std::span<const std::uint32_t> packed,
    std::size_t rows,
    std::size_t packed_cols,
    std::span<const float> scales,
    std::span<const float> biases,
    std::size_t group_size) noexcept {
  VulkanQ4DequantResult result;

  if (!context.valid()) {
    result.diagnostic = "Vulkan compute context is not valid.";
    return result;
  }

  if (rows == 0 || packed_cols == 0 || group_size == 0) {
    result.diagnostic =
        "Q4 rows, packed_cols, and group_size must be non-zero.";
    return result;
  }

  if (packed_cols > std::numeric_limits<std::size_t>::max() / 8U) {
    result.diagnostic = "Q4 logical column count overflow.";
    return result;
  }

  const std::size_t logical_cols = packed_cols * 8U;

  if (logical_cols % group_size != 0U) {
    result.diagnostic =
        "Q4 logical columns must be divisible by group_size.";
    return result;
  }

  if (rows > std::numeric_limits<std::size_t>::max() / packed_cols ||
      packed.size() != rows * packed_cols) {
    result.diagnostic = "Q4 packed size does not match rows * packed_cols.";
    return result;
  }

  const std::size_t groups_per_row = logical_cols / group_size;
  if (rows > std::numeric_limits<std::size_t>::max() / groups_per_row) {
    result.diagnostic = "Q4 scale/bias shape overflow.";
    return result;
  }

  const std::size_t group_values = rows * groups_per_row;
  if (scales.size() != group_values || biases.size() != group_values) {
    result.diagnostic =
        "Q4 scales/biases do not match rows * groups_per_row.";
    return result;
  }

  if (rows > std::numeric_limits<std::size_t>::max() / logical_cols) {
    result.diagnostic = "Q4 output shape overflow.";
    return result;
  }

  const std::size_t total_values = rows * logical_cols;

  if (total_values > std::numeric_limits<std::uint32_t>::max() ||
      packed_cols > std::numeric_limits<std::uint32_t>::max() ||
      logical_cols > std::numeric_limits<std::uint32_t>::max() ||
      group_size > std::numeric_limits<std::uint32_t>::max()) {
    result.diagnostic = "Q4 shape exceeds Vulkan bootstrap limits.";
    return result;
  }

  try {
    const auto physical_device =
        reinterpret_cast<VkPhysicalDevice>(
            context.native_physical_device());
    const auto device =
        reinterpret_cast<VkDevice>(context.native_device());
    const auto queue =
        reinterpret_cast<VkQueue>(context.native_queue());
    const auto command_pool =
        reinterpret_cast<VkCommandPool>(
            context.native_command_pool());

    if (physical_device == VK_NULL_HANDLE ||
        device == VK_NULL_HANDLE ||
        queue == VK_NULL_HANDLE ||
        command_pool == VK_NULL_HANDLE) {
      result.diagnostic =
          "Vulkan compute context exposed a null native handle.";
      return result;
    }

    constexpr std::uint32_t kLocalSize = 64U;
    const std::uint32_t workgroups =
        (static_cast<std::uint32_t>(total_values) + kLocalSize - 1U) /
        kLocalSize;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    if (workgroups > properties.limits.maxComputeWorkGroupCount[0]) {
      result.diagnostic =
          "Q4 dispatch exceeds maxComputeWorkGroupCount[0].";
      return result;
    }

    const VkDeviceSize packed_bytes =
        static_cast<VkDeviceSize>(packed.size() * sizeof(std::uint32_t));
    const VkDeviceSize scale_bytes =
        static_cast<VkDeviceSize>(scales.size() * sizeof(float));
    const VkDeviceSize bias_bytes =
        static_cast<VkDeviceSize>(biases.size() * sizeof(float));
    const VkDeviceSize output_bytes =
        static_cast<VkDeviceSize>(total_values * sizeof(float));

    HostBuffer packed_buffer;
    HostBuffer scale_buffer;
    HostBuffer bias_buffer;
    HostBuffer output_buffer;

    if (!create_host_storage_buffer(
            physical_device, device, packed_bytes, packed_buffer,
            result.diagnostic) ||
        !create_host_storage_buffer(
            physical_device, device, scale_bytes, scale_buffer,
            result.diagnostic) ||
        !create_host_storage_buffer(
            physical_device, device, bias_bytes, bias_buffer,
            result.diagnostic) ||
        !create_host_storage_buffer(
            physical_device, device, output_bytes, output_buffer,
            result.diagnostic)) {
      return result;
    }

    if (!write_host_buffer(
            packed_buffer, packed.data(),
            static_cast<std::size_t>(packed_bytes), result.diagnostic) ||
        !write_host_buffer(
            scale_buffer, scales.data(),
            static_cast<std::size_t>(scale_bytes), result.diagnostic) ||
        !write_host_buffer(
            bias_buffer, biases.data(),
            static_cast<std::size_t>(bias_bytes), result.diagnostic)) {
      return result;
    }

    VulkanObjects objects;
    objects.device = device;
    objects.command_pool = command_pool;

    VkDescriptorSetLayoutBinding bindings[4]{};
    for (std::uint32_t i = 0; i < 4; ++i) {
      bindings[i] = {
          .binding = i,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
          .pImmutableSamplers = nullptr,
      };
    }

    VkDescriptorSetLayoutCreateInfo descriptor_layout_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = 4,
        .pBindings = bindings,
    };

    VkResult status = vkCreateDescriptorSetLayout(
        device,
        &descriptor_layout_ci,
        nullptr,
        &objects.descriptor_set_layout);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkCreateDescriptorSetLayout failed: " +
          vk_result_string(status);
      return result;
    }

    VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(Q4PushConstants),
    };

    VkPipelineLayoutCreateInfo pipeline_layout_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &objects.descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };

    status = vkCreatePipelineLayout(
        device,
        &pipeline_layout_ci,
        nullptr,
        &objects.pipeline_layout);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkCreatePipelineLayout failed: " + vk_result_string(status);
      return result;
    }

    VkShaderModuleCreateInfo shader_ci{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize =
            generated::kQ4DequantSpirv.size() * sizeof(std::uint32_t),
        .pCode = generated::kQ4DequantSpirv.data(),
    };

    status = vkCreateShaderModule(
        device,
        &shader_ci,
        nullptr,
        &objects.shader_module);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkCreateShaderModule failed: " + vk_result_string(status);
      return result;
    }

    VkPipelineShaderStageCreateInfo stage_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = objects.shader_module,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };

    VkComputePipelineCreateInfo compute_ci{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = stage_ci,
        .layout = objects.pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    status = vkCreateComputePipelines(
        device,
        VK_NULL_HANDLE,
        1,
        &compute_ci,
        nullptr,
        &objects.pipeline);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkCreateComputePipelines failed: " +
          vk_result_string(status);
      return result;
    }

    VkDescriptorPoolSize pool_size{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4,
    };

    VkDescriptorPoolCreateInfo pool_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };

    status = vkCreateDescriptorPool(
        device,
        &pool_ci,
        nullptr,
        &objects.descriptor_pool);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkCreateDescriptorPool failed: " +
          vk_result_string(status);
      return result;
    }

    VkDescriptorSetAllocateInfo set_ai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = objects.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &objects.descriptor_set_layout,
    };

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    status = vkAllocateDescriptorSets(
        device,
        &set_ai,
        &descriptor_set);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkAllocateDescriptorSets failed: " +
          vk_result_string(status);
      return result;
    }

    const VkDescriptorBufferInfo infos[4]{
        {.buffer = packed_buffer.buffer, .offset = 0, .range = packed_bytes},
        {.buffer = scale_buffer.buffer, .offset = 0, .range = scale_bytes},
        {.buffer = bias_buffer.buffer, .offset = 0, .range = bias_bytes},
        {.buffer = output_buffer.buffer, .offset = 0, .range = output_bytes},
    };

    VkWriteDescriptorSet writes[4]{};
    for (std::uint32_t i = 0; i < 4; ++i) {
      writes[i] = {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .pNext = nullptr,
          .dstSet = descriptor_set,
          .dstBinding = i,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pImageInfo = nullptr,
          .pBufferInfo = &infos[i],
          .pTexelBufferView = nullptr,
      };
    }
    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

    VkCommandBufferAllocateInfo command_ai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    status = vkAllocateCommandBuffers(
        device,
        &command_ai,
        &objects.command_buffer);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkAllocateCommandBuffers failed: " +
          vk_result_string(status);
      return result;
    }

    VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };

    status = vkBeginCommandBuffer(objects.command_buffer, &begin_info);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkBeginCommandBuffer failed: " + vk_result_string(status);
      return result;
    }

    const Q4PushConstants params{
        .total_values = static_cast<std::uint32_t>(total_values),
        .packed_cols = static_cast<std::uint32_t>(packed_cols),
        .logical_cols = static_cast<std::uint32_t>(logical_cols),
        .group_size = static_cast<std::uint32_t>(group_size),
    };

    vkCmdBindPipeline(
        objects.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        objects.pipeline);

    vkCmdBindDescriptorSets(
        objects.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        objects.pipeline_layout,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);

    vkCmdPushConstants(
        objects.command_buffer,
        objects.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(params),
        &params);

    vkCmdDispatch(objects.command_buffer, workgroups, 1, 1);

    status = vkEndCommandBuffer(objects.command_buffer);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkEndCommandBuffer failed: " + vk_result_string(status);
      return result;
    }

    VkFenceCreateInfo fence_ci{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };

    status = vkCreateFence(
        device,
        &fence_ci,
        nullptr,
        &objects.fence);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkCreateFence failed: " + vk_result_string(status);
      return result;
    }

    VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &objects.command_buffer,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };

    status = vkQueueSubmit(queue, 1, &submit_info, objects.fence);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkQueueSubmit failed: " + vk_result_string(status);
      return result;
    }

    constexpr std::uint64_t kTimeoutNanoseconds =
        5ULL * 1000ULL * 1000ULL * 1000ULL;

    status = vkWaitForFences(
        device,
        1,
        &objects.fence,
        VK_TRUE,
        kTimeoutNanoseconds);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkWaitForFences failed: " + vk_result_string(status);
      return result;
    }

    result.values.resize(total_values);
    if (!read_host_buffer(
            output_buffer,
            result.values.data(),
            static_cast<std::size_t>(output_bytes),
            result.diagnostic)) {
      result.values.clear();
      return result;
    }

    result.executed = true;
    result.diagnostic =
        "Vulkan affine Q4 dequantization completed.";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("Vulkan Q4 dequant exception: ") + e.what();
    result.values.clear();
    return result;
  } catch (...) {
    result.diagnostic =
        "Vulkan Q4 dequant encountered an unknown exception.";
    result.values.clear();
    return result;
  }
}

}  // namespace orbi::streammoe
