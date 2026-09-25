#include "orbi/streammoe/backend/vulkan_compute_roundtrip.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <volk.h>

namespace orbi::streammoe {
namespace {

constexpr std::uint32_t instruction(
    std::uint16_t word_count,
    std::uint16_t opcode) noexcept {
  return (static_cast<std::uint32_t>(word_count) << 16U) |
         static_cast<std::uint32_t>(opcode);
}

std::vector<std::uint32_t> make_write_sentinel_spirv() {
  // Minimal SPIR-V 1.0 compute module equivalent to:
  //
  // layout(set=0,binding=0) buffer B { uint value; } b;
  // void main() { b.value = 0x12345678u; }
  //
  // It deliberately uses the SPIR-V 1.0 Uniform + BufferBlock representation
  // of a Vulkan storage buffer so the bootstrap remains usable on Vulkan 1.0
  // class drivers as well as modern Windows/Android implementations.
  enum : std::uint32_t {
    kVoid = 1,
    kFunctionType = 2,
    kUint = 3,
    kBufferStruct = 4,
    kPtrUniformStruct = 5,
    kBufferVariable = 6,
    kZero = 7,
    kSentinel = 8,
    kPtrUniformUint = 9,
    kMain = 10,
    kLabel = 11,
    kElementPointer = 12,
    kBound = 13,
  };

  return {
      // SPIR-V header: magic, version 1.0, generator, bound, schema.
      0x07230203U, 0x00010000U, 0U, kBound, 0U,

      // OpCapability Shader
      instruction(2, 17), 1U,

      // OpMemoryModel Logical GLSL450
      instruction(3, 14), 0U, 1U,

      // OpEntryPoint GLCompute %main "main"
      instruction(5, 15), 5U, kMain, 0x6e69616dU, 0U,

      // OpExecutionMode %main LocalSize 1 1 1
      instruction(6, 16), kMain, 17U, 1U, 1U, 1U,

      // OpDecorate %Buffer BufferBlock
      instruction(3, 71), kBufferStruct, 3U,

      // OpMemberDecorate %Buffer 0 Offset 0
      instruction(5, 72), kBufferStruct, 0U, 35U, 0U,

      // OpDecorate %buffer DescriptorSet 0
      instruction(4, 71), kBufferVariable, 34U, 0U,

      // OpDecorate %buffer Binding 0
      instruction(4, 71), kBufferVariable, 33U, 0U,

      // %void = OpTypeVoid
      instruction(2, 19), kVoid,

      // %fn = OpTypeFunction %void
      instruction(3, 33), kFunctionType, kVoid,

      // %uint = OpTypeInt 32 0
      instruction(4, 21), kUint, 32U, 0U,

      // %Buffer = OpTypeStruct %uint
      instruction(3, 30), kBufferStruct, kUint,

      // %ptrBuffer = OpTypePointer Uniform %Buffer
      instruction(4, 32), kPtrUniformStruct, 2U, kBufferStruct,

      // %buffer = OpVariable %ptrBuffer Uniform
      instruction(4, 59), kPtrUniformStruct, kBufferVariable, 2U,

      // %zero = OpConstant %uint 0
      instruction(4, 43), kUint, kZero, 0U,

      // %sentinel = OpConstant %uint 0x12345678
      instruction(4, 43), kUint, kSentinel, kVulkanRoundTripSentinel,

      // %ptrUint = OpTypePointer Uniform %uint
      instruction(4, 32), kPtrUniformUint, 2U, kUint,

      // %main = OpFunction %void None %fn
      instruction(5, 54), kVoid, kMain, 0U, kFunctionType,

      // %label = OpLabel
      instruction(2, 248), kLabel,

      // %element = OpAccessChain %ptrUint %buffer %zero
      instruction(5, 65),
      kPtrUniformUint,
      kElementPointer,
      kBufferVariable,
      kZero,

      // OpStore %element %sentinel
      instruction(3, 62), kElementPointer, kSentinel,

      // OpReturn
      instruction(1, 253),

      // OpFunctionEnd
      instruction(1, 56),
  };
}

bool structurally_valid_spirv(
    const std::vector<std::uint32_t>& words) noexcept {
  if (words.size() < 6U ||
      words[0] != 0x07230203U ||
      words[1] != 0x00010000U ||
      words[3] == 0U) {
    return false;
  }

  std::size_t cursor = 5U;
  while (cursor < words.size()) {
    const auto word_count = static_cast<std::uint16_t>(words[cursor] >> 16U);
    if (word_count == 0U || cursor + word_count > words.size()) {
      return false;
    }
    cursor += word_count;
  }
  return cursor == words.size();
}

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

struct VulkanObjects {
  VkDevice device{VK_NULL_HANDLE};
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptor_set_layout{VK_NULL_HANDLE};
  VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
  VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
  VkShaderModule shader_module{VK_NULL_HANDLE};
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};
  VkCommandPool command_pool{VK_NULL_HANDLE};
  VkCommandBuffer command_buffer{VK_NULL_HANDLE};

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
    if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, buffer, nullptr);
    if (memory != VK_NULL_HANDLE) vkFreeMemory(device, memory, nullptr);
  }
};

}  // namespace

VulkanRoundTripResult run_vulkan_compute_roundtrip(
    VulkanComputeContext& context,
    std::uint32_t initial_value) noexcept {
  VulkanRoundTripResult result;
  result.initial_value = initial_value;

  if (!context.valid()) {
    result.diagnostic = "Vulkan compute context is not valid.";
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
      result.diagnostic = "Vulkan compute context exposed a null native handle.";
      return result;
    }

    VulkanObjects objects;
    objects.device = device;
    objects.command_pool = command_pool;

    VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = sizeof(std::uint32_t),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    VkResult status =
        vkCreateBuffer(device, &buffer_ci, nullptr, &objects.buffer);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkCreateBuffer failed: " + vk_result_string(status);
      return result;
    }

    VkMemoryRequirements memory_requirements{};
    vkGetBufferMemoryRequirements(
        device, objects.buffer, &memory_requirements);

    VkMemoryPropertyFlags memory_flags = 0;
    const std::uint32_t memory_type = find_host_memory_type(
        physical_device,
        memory_requirements.memoryTypeBits,
        &memory_flags);
    if (memory_type == std::numeric_limits<std::uint32_t>::max()) {
      result.diagnostic =
          "No host-visible Vulkan memory type is compatible with the test buffer.";
      return result;
    }

    VkMemoryAllocateInfo memory_ai{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = memory_type,
    };

    status = vkAllocateMemory(
        device, &memory_ai, nullptr, &objects.memory);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkAllocateMemory failed: " + vk_result_string(status);
      return result;
    }

    status = vkBindBufferMemory(
        device, objects.buffer, objects.memory, 0);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkBindBufferMemory failed: " + vk_result_string(status);
      return result;
    }

    void* mapped = nullptr;
    status = vkMapMemory(
        device,
        objects.memory,
        0,
        VK_WHOLE_SIZE,
        0,
        &mapped);
    if (status != VK_SUCCESS || mapped == nullptr) {
      result.diagnostic =
          "vkMapMemory(initial) failed: " + vk_result_string(status);
      return result;
    }

    std::memcpy(mapped, &initial_value, sizeof(initial_value));
    if ((memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0U) {
      VkMappedMemoryRange range{
          .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
          .pNext = nullptr,
          .memory = objects.memory,
          .offset = 0,
          .size = VK_WHOLE_SIZE,
      };
      status = vkFlushMappedMemoryRanges(device, 1, &range);
      if (status != VK_SUCCESS) {
        vkUnmapMemory(device, objects.memory);
        result.diagnostic =
            "vkFlushMappedMemoryRanges failed: " +
            vk_result_string(status);
        return result;
      }
    }
    vkUnmapMemory(device, objects.memory);

    VkDescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };

    VkDescriptorSetLayoutCreateInfo descriptor_layout_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = 1,
        .pBindings = &binding,
    };

    status = vkCreateDescriptorSetLayout(
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

    VkPipelineLayoutCreateInfo pipeline_layout_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &objects.descriptor_set_layout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };

    status = vkCreatePipelineLayout(
        device,
        &pipeline_layout_ci,
        nullptr,
        &objects.pipeline_layout);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkCreatePipelineLayout failed: " +
          vk_result_string(status);
      return result;
    }

    const auto shader_words = make_write_sentinel_spirv();
    if (!structurally_valid_spirv(shader_words)) {
      result.diagnostic = "Embedded OSM-09 SPIR-V failed structural validation.";
      return result;
    }

    VkShaderModuleCreateInfo shader_ci{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = shader_words.size() * sizeof(std::uint32_t),
        .pCode = shader_words.data(),
    };

    status = vkCreateShaderModule(
        device, &shader_ci, nullptr, &objects.shader_module);
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
        .descriptorCount = 1,
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
        device, &pool_ci, nullptr, &objects.descriptor_pool);
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
        device, &set_ai, &descriptor_set);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkAllocateDescriptorSets failed: " +
          vk_result_string(status);
      return result;
    }

    VkDescriptorBufferInfo buffer_info{
        .buffer = objects.buffer,
        .offset = 0,
        .range = sizeof(std::uint32_t),
    };

    VkWriteDescriptorSet descriptor_write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &buffer_info,
        .pTexelBufferView = nullptr,
    };
    vkUpdateDescriptorSets(device, 1, &descriptor_write, 0, nullptr);

    VkCommandBufferAllocateInfo command_ai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    status = vkAllocateCommandBuffers(
        device, &command_ai, &objects.command_buffer);
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
          "vkBeginCommandBuffer failed: " +
          vk_result_string(status);
      return result;
    }

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
    vkCmdDispatch(objects.command_buffer, 1, 1, 1);

    status = vkEndCommandBuffer(objects.command_buffer);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkEndCommandBuffer failed: " +
          vk_result_string(status);
      return result;
    }

    VkFenceCreateInfo fence_ci{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };

    status = vkCreateFence(
        device, &fence_ci, nullptr, &objects.fence);
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
        device, 1, &objects.fence, VK_TRUE, kTimeoutNanoseconds);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkWaitForFences failed: " + vk_result_string(status);
      return result;
    }

    mapped = nullptr;
    status = vkMapMemory(
        device,
        objects.memory,
        0,
        VK_WHOLE_SIZE,
        0,
        &mapped);
    if (status != VK_SUCCESS || mapped == nullptr) {
      result.diagnostic =
          "vkMapMemory(readback) failed: " +
          vk_result_string(status);
      return result;
    }

    if ((memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0U) {
      VkMappedMemoryRange range{
          .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
          .pNext = nullptr,
          .memory = objects.memory,
          .offset = 0,
          .size = VK_WHOLE_SIZE,
      };
      status = vkInvalidateMappedMemoryRanges(device, 1, &range);
      if (status != VK_SUCCESS) {
        vkUnmapMemory(device, objects.memory);
        result.diagnostic =
            "vkInvalidateMappedMemoryRanges failed: " +
            vk_result_string(status);
        return result;
      }
    }

    std::memcpy(
        &result.output_value,
        mapped,
        sizeof(result.output_value));
    vkUnmapMemory(device, objects.memory);

    result.executed = true;
    result.diagnostic =
        result.output_value == kVulkanRoundTripSentinel
            ? "Vulkan compute round-trip wrote the expected sentinel."
            : "Vulkan compute dispatch completed but readback did not match the sentinel.";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("Vulkan round-trip exception: ") + e.what();
    return result;
  } catch (...) {
    result.diagnostic =
        "Vulkan round-trip encountered an unknown exception.";
    return result;
  }
}

}  // namespace orbi::streammoe
