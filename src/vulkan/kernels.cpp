#include "welvet/vulkan/kernels.hpp"

#include "vulkan/internal.hpp"

#include <cstring>

namespace welvet::vulkan {
namespace {

struct SaxpyPush {
  float a;
  float b;
  std::uint32_t n;
};

} // namespace

Status launch_smoke_saxpy(const Buffer& x, Buffer* y, float a, float b, std::size_t n) {
  if (!y) return Status::Err(StatusCode::InvalidArgument, "null y buffer");
  if (!x.handle || !y->handle) return Status::Err(StatusCode::InvalidArgument, "null buffer handle");
  if (n == 0) return Status::Err(StatusCode::InvalidArgument, "zero-length saxpy");
  if (x.bytes < n * sizeof(float) || y->bytes < n * sizeof(float)) {
    return Status::Err(StatusCode::InvalidArgument, "buffer too small for saxpy");
  }

  auto st = detail::ensure_runtime();
  if (!st) return st;

  auto& rt = detail::runtime();
  std::lock_guard<std::mutex> lock(detail::runtime_mu());

  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = rt.descriptor_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &rt.saxpy_set_layout;
  VkDescriptorSet set = VK_NULL_HANDLE;
  st = detail::vk_check(vkAllocateDescriptorSets(rt.device, &dsai, &set), "vkAllocateDescriptorSets");
  if (!st) return st;

  VkDescriptorBufferInfo xinfo{};
  xinfo.buffer = static_cast<VkBuffer>(x.handle);
  xinfo.offset = 0;
  xinfo.range = n * sizeof(float);
  VkDescriptorBufferInfo yinfo{};
  yinfo.buffer = static_cast<VkBuffer>(y->handle);
  yinfo.offset = 0;
  yinfo.range = n * sizeof(float);

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = set;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &xinfo;
  writes[1] = writes[0];
  writes[1].dstBinding = 1;
  writes[1].pBufferInfo = &yinfo;
  vkUpdateDescriptorSets(rt.device, 2, writes, 0, nullptr);

  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = rt.command_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  st = detail::vk_check(vkAllocateCommandBuffers(rt.device, &cbai, &cmd), "vkAllocateCommandBuffers");
  if (!st) {
    vkFreeDescriptorSets(rt.device, rt.descriptor_pool, 1, &set);
    return st;
  }

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  st = detail::vk_check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
  if (!st) {
    vkFreeCommandBuffers(rt.device, rt.command_pool, 1, &cmd);
    vkFreeDescriptorSets(rt.device, rt.descriptor_pool, 1, &set);
    return st;
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rt.saxpy_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rt.saxpy_pipeline_layout, 0, 1, &set,
                          0, nullptr);
  SaxpyPush push{a, b, static_cast<std::uint32_t>(n)};
  vkCmdPushConstants(cmd, rt.saxpy_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                     &push);
  const std::uint32_t groups = static_cast<std::uint32_t>((n + 63) / 64);
  vkCmdDispatch(cmd, groups, 1, 1);
  st = detail::vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
  if (!st) {
    vkFreeCommandBuffers(rt.device, rt.command_pool, 1, &cmd);
    vkFreeDescriptorSets(rt.device, rt.descriptor_pool, 1, &set);
    return st;
  }

  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  st = detail::vk_check(vkQueueSubmit(rt.compute_queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
  if (st) {
    st = detail::vk_check(vkQueueWaitIdle(rt.compute_queue), "vkQueueWaitIdle");
  }

  vkFreeCommandBuffers(rt.device, rt.command_pool, 1, &cmd);
  vkFreeDescriptorSets(rt.device, rt.descriptor_pool, 1, &set);
  return st;
}

} // namespace welvet::vulkan
