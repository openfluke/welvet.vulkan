#include "welvet/vulkan/buffer.hpp"

#include "vulkan/internal.hpp"

#include <cstring>

namespace welvet::vulkan {

Status buffer_alloc(std::size_t bytes, Buffer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null Buffer");
  if (bytes == 0) return Status::Err(StatusCode::InvalidArgument, "zero-size buffer");

  auto st = detail::ensure_runtime();
  if (!st) return st;

  auto& rt = detail::runtime();
  std::lock_guard<std::mutex> lock(detail::runtime_mu());

  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = bytes;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer = VK_NULL_HANDLE;
  st = detail::vk_check(vkCreateBuffer(rt.device, &bci, nullptr, &buffer), "vkCreateBuffer");
  if (!st) return st;

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(rt.device, buffer, &req);
  const std::uint32_t mem_type = detail::find_memory_type(
      req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (mem_type == UINT32_MAX) {
    vkDestroyBuffer(rt.device, buffer, nullptr);
    return Status::Err(StatusCode::Unsupported, "no host-visible memory for buffer");
  }

  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = mem_type;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  st = detail::vk_check(vkAllocateMemory(rt.device, &mai, nullptr, &memory), "vkAllocateMemory");
  if (!st) {
    vkDestroyBuffer(rt.device, buffer, nullptr);
    return st;
  }
  st = detail::vk_check(vkBindBufferMemory(rt.device, buffer, memory, 0), "vkBindBufferMemory");
  if (!st) {
    vkFreeMemory(rt.device, memory, nullptr);
    vkDestroyBuffer(rt.device, buffer, nullptr);
    return st;
  }

  void* mapped = nullptr;
  st = detail::vk_check(vkMapMemory(rt.device, memory, 0, bytes, 0, &mapped), "vkMapMemory");
  if (!st) {
    vkFreeMemory(rt.device, memory, nullptr);
    vkDestroyBuffer(rt.device, buffer, nullptr);
    return st;
  }

  out->handle = buffer;
  out->memory = memory;
  out->mapped = mapped;
  out->bytes = bytes;
  out->device_owned = true;
  return Status::Ok();
}

Status buffer_upload(Buffer* buf, const void* host, std::size_t bytes) {
  if (!buf || !buf->handle || !buf->mapped) {
    return Status::Err(StatusCode::InvalidArgument, "null buffer");
  }
  if (!host) return Status::Err(StatusCode::InvalidArgument, "null host");
  if (bytes > buf->bytes) return Status::Err(StatusCode::InvalidArgument, "upload overflow");
  std::memcpy(buf->mapped, host, bytes);
  return Status::Ok();
}

Status buffer_download(const Buffer& buf, void* host, std::size_t bytes) {
  if (!buf.handle || !buf.mapped) {
    return Status::Err(StatusCode::InvalidArgument, "null buffer");
  }
  if (!host) return Status::Err(StatusCode::InvalidArgument, "null host");
  if (bytes > buf.bytes) return Status::Err(StatusCode::InvalidArgument, "download overflow");
  std::memcpy(host, buf.mapped, bytes);
  return Status::Ok();
}

void buffer_release(Buffer* buf) {
  if (!buf || !buf->handle) return;
  auto st = detail::ensure_runtime();
  if (!st) {
    buf->handle = nullptr;
    buf->memory = nullptr;
    buf->mapped = nullptr;
    buf->bytes = 0;
    return;
  }
  auto& rt = detail::runtime();
  std::lock_guard<std::mutex> lock(detail::runtime_mu());
  if (buf->device_owned) {
    if (buf->mapped && buf->memory) {
      vkUnmapMemory(rt.device, static_cast<VkDeviceMemory>(buf->memory));
    }
    if (buf->handle) {
      vkDestroyBuffer(rt.device, static_cast<VkBuffer>(buf->handle), nullptr);
    }
    if (buf->memory) {
      vkFreeMemory(rt.device, static_cast<VkDeviceMemory>(buf->memory), nullptr);
    }
  }
  buf->handle = nullptr;
  buf->memory = nullptr;
  buf->mapped = nullptr;
  buf->bytes = 0;
}

} // namespace welvet::vulkan
