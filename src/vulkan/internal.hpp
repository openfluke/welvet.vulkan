#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>
#include <string>

#include "welvet/status.hpp"
#include "welvet/vulkan/device.hpp"

namespace welvet::vulkan::detail {

struct Runtime {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue compute_queue = VK_NULL_HANDLE;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSetLayout saxpy_set_layout = VK_NULL_HANDLE;
  VkPipelineLayout saxpy_pipeline_layout = VK_NULL_HANDLE;
  VkPipeline saxpy_pipeline = VK_NULL_HANDLE;
  VkShaderModule saxpy_module = VK_NULL_HANDLE;
  std::uint32_t compute_queue_family = 0;
  std::uint32_t host_visible_mem_type = UINT32_MAX;
  DeviceInfo info{};
  std::string identity = "vulkan:unavailable";
  bool ready = false;
};

std::mutex& runtime_mu();
Runtime& runtime();

Status ensure_runtime();
Status vk_check(VkResult r, const char* what);
std::uint32_t find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags props);
DeviceInfo info_from(VkPhysicalDevice pd, int index);

} // namespace welvet::vulkan::detail
