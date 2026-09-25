#include "orbi/streammoe/backend/vulkan_weighted_accumulate.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>

#include <volk.h>

#include "orbi/streammoe/generated/weighted_accumulate_spv.hpp"

namespace orbi::streammoe {
namespace {

std::string vk_result_string(VkResult result) {
  return "VkResult(" + std::to_string(static_cast<int>(result)) + ")";
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
    if (command_buffer != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    }
  }
};

struct WeightedAccumulatePushConstants {
  std::uint32_t count{};
  float weight{};
};
static_assert(sizeof(WeightedAccumulatePushConstants) == 8);

}  // namespace

VulkanBufferDispatchResult run_vulkan_weighted_accumulate(
    VulkanComputeContext& context,
    const VulkanFloatBuffer& source,
    float weight,
    VulkanFloatBuffer& accumulator) noexcept {
  VulkanBufferDispatchResult result;

  if (!context.valid()) {
    result.diagnostic = "Vulkan compute context is not valid.";
    return result;
  }
  if (!source.valid() || !accumulator.valid()) {
    result.diagnostic =
        "weighted accumulate requires valid Vulkan buffers.";
    return result;
  }
  if (source.size() == 0U || source.size() != accumulator.size()) {
    result.diagnostic =
        "weighted accumulate buffers must have equal non-zero sizes.";
    return result;
  }
  if (source.size() > std::numeric_limits<std::uint32_t>::max()) {
    result.diagnostic =
        "weighted accumulate input exceeds Vulkan bootstrap limits.";
    return result;
  }
  if (!std::isfinite(weight)) {
    result.diagnostic =
        "weighted accumulate routing weight must be finite.";
    return result;
  }
  if (source.native_device() != context.native_device() ||
      accumulator.native_device() != context.native_device()) {
    result.diagnostic =
        "weighted accumulate buffers belong to a different Vulkan device.";
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

    const auto source_buffer =
        reinterpret_cast<VkBuffer>(source.native_buffer());
    const auto accumulator_buffer =
        reinterpret_cast<VkBuffer>(accumulator.native_buffer());

    if (source_buffer == VK_NULL_HANDLE ||
        accumulator_buffer == VK_NULL_HANDLE) {
      result.diagnostic =
          "weighted accumulate Vulkan buffer handle is null.";
      return result;
    }

    constexpr std::uint32_t kLocalSize = 64U;
    const auto count = static_cast<std::uint32_t>(source.size());
    const std::uint32_t workgroups =
        (count + kLocalSize - 1U) / kLocalSize;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    if (workgroups > properties.limits.maxComputeWorkGroupCount[0]) {
      result.diagnostic =
          "weighted accumulate dispatch exceeds maxComputeWorkGroupCount[0].";
      return result;
    }

    VulkanObjects objects;
    objects.device = device;
    objects.command_pool = command_pool;

    VkDescriptorSetLayoutBinding bindings[2]{};
    for (std::uint32_t i = 0; i < 2; ++i) {
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
        .bindingCount = 2,
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
        .size = sizeof(WeightedAccumulatePushConstants),
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
          "vkCreatePipelineLayout failed: " +
          vk_result_string(status);
      return result;
    }

    VkShaderModuleCreateInfo shader_ci{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize =
            generated::kWeightedAccumulateSpirv.size() *
            sizeof(std::uint32_t),
        .pCode = generated::kWeightedAccumulateSpirv.data(),
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
        .descriptorCount = 2,
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
        device, &set_ai, &descriptor_set);
    if (status != VK_SUCCESS) {
      result.diagnostic =
          "vkAllocateDescriptorSets failed: " +
          vk_result_string(status);
      return result;
    }

    const VkDescriptorBufferInfo infos[2]{
        {
            .buffer = source_buffer,
            .offset = 0,
            .range = static_cast<VkDeviceSize>(source.size_bytes()),
        },
        {
            .buffer = accumulator_buffer,
            .offset = 0,
            .range = static_cast<VkDeviceSize>(accumulator.size_bytes()),
        },
    };

    VkWriteDescriptorSet writes[2]{};
    for (std::uint32_t i = 0; i < 2; ++i) {
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
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

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
          "vkBeginCommandBuffer failed: " + vk_result_string(status);
      return result;
    }

    // Make earlier host uploads / compute writes visible to this compute read
    // and the accumulator read-modify-write.
    VkMemoryBarrier pre_barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask =
            VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(
        objects.command_buffer,
        VK_PIPELINE_STAGE_HOST_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &pre_barrier,
        0,
        nullptr,
        0,
        nullptr);

    const WeightedAccumulatePushConstants params{
        .count = count,
        .weight = weight,
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

    VkMemoryBarrier post_barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(
        objects.command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &post_barrier,
        0,
        nullptr,
        0,
        nullptr);

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

    status = vkQueueSubmit(
        queue, 1, &submit_info, objects.fence);
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

    result.executed = true;
    result.diagnostic =
        "Vulkan weighted accumulation completed in-place.";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("Vulkan weighted accumulate exception: ") + e.what();
    return result;
  } catch (...) {
    result.diagnostic =
        "Vulkan weighted accumulate encountered an unknown exception.";
    return result;
  }
}

}  // namespace orbi::streammoe
