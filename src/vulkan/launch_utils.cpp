#include "vulkan/launch_utils.hpp"

#include "vulkan/internal.hpp"

namespace welvet::vulkan::detail {

VkDescriptorBufferInfo buf_info(const Buffer& b, std::size_t bytes) {
  VkDescriptorBufferInfo info{};
  info.buffer = static_cast<VkBuffer>(b.handle);
  info.offset = 0;
  info.range = bytes > 0 ? bytes : b.bytes;
  return info;
}

Status sync_queue() {
  auto st = ensure_runtime();
  if (!st) return st;
  std::lock_guard<std::mutex> lock(runtime_mu());
  return vk_check(vkQueueWaitIdle(runtime().compute_queue), "vkQueueWaitIdle");
}

} // namespace welvet::vulkan::detail
