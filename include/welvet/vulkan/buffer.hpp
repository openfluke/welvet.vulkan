#pragma once

#include <cstddef>
#include <cstdint>

#include "welvet/status.hpp"

namespace welvet::vulkan {

// Opaque device buffer owned by the engine (host-visible for M0 staging).
struct Buffer {
  void* handle = nullptr;   // VkBuffer (opaque)
  void* memory = nullptr;   // VkDeviceMemory (opaque)
  void* mapped = nullptr;   // persistent host mapping when available
  std::size_t bytes = 0;
  bool device_owned = true;
};

Status buffer_alloc(std::size_t bytes, Buffer* out);
Status buffer_upload(Buffer* buf, const void* host, std::size_t bytes);
Status buffer_download(const Buffer& buf, void* host, std::size_t bytes);
void buffer_release(Buffer* buf);

} // namespace welvet::vulkan
