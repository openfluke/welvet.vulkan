#include "welvet/vulkan/kernels.hpp"

#include "vulkan/internal.hpp"
#include "vulkan/launch_utils.hpp"
#include "vulkan/pipelines.hpp"

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

  VkDescriptorBufferInfo bufs[2] = {detail::buf_info(x, n * sizeof(float)),
                                    detail::buf_info(*y, n * sizeof(float))};
  SaxpyPush push{a, b, static_cast<std::uint32_t>(n)};
  const std::uint32_t groups = static_cast<std::uint32_t>((n + 63) / 64);
  return detail::dispatch_compute(detail::PipelineId::SmokeSaxpy, bufs, 2, &push, sizeof(push),
                                  groups, 1, 1);
}

} // namespace welvet::vulkan
