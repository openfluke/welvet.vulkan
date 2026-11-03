#pragma once

#include <vulkan/vulkan.h>

#include "welvet/vulkan/buffer.hpp"
#include "welvet/status.hpp"

namespace welvet::vulkan::detail {

VkDescriptorBufferInfo buf_info(const Buffer& b, std::size_t bytes);
Status sync_queue();

} // namespace welvet::vulkan::detail
