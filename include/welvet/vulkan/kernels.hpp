#pragma once

#include <cstddef>

#include "welvet/status.hpp"
#include "welvet/vulkan/buffer.hpp"

namespace welvet::vulkan {

// M0 smoke: y[i] = a * x[i] + b  (device storage buffers).
Status launch_smoke_saxpy(const Buffer& x, Buffer* y, float a, float b, std::size_t n);

} // namespace welvet::vulkan
