#include "welvet/gaps.hpp"
#include "welvet/vulkan/buffer.hpp"
#include "welvet/vulkan/device.hpp"
#include "welvet/vulkan/kernels.hpp"
#include "welvet_c.h"

#include <cstdio>
#include <iostream>
#include <vector>

int main() {
  std::printf("welvet.vulkan %s\n", welvet_version());

  auto st = welvet::vulkan::init_default_device();
  if (!st) {
    std::fprintf(stderr, "init: %s\n", st.message.c_str());
    return 1;
  }
  std::printf("runtime: %s\n", welvet::vulkan::runtime_identity().c_str());

  std::cout << "devices:\n";
  for (const auto& d : welvet::vulkan::list_devices()) {
    std::cout << "  [" << d.index << "] " << d.name << " api=" << ((d.api_version >> 22) & 0x7f)
              << "." << ((d.api_version >> 12) & 0x3ff) << "." << (d.api_version & 0xfff)
              << " type=" << d.device_type << " mem=" << d.total_device_mem << "\n";
  }

  constexpr std::size_t n = 4;
  welvet::vulkan::Buffer x_buf;
  welvet::vulkan::Buffer y_buf;
  st = welvet::vulkan::buffer_alloc(n * sizeof(float), &x_buf);
  if (!st) {
    std::fprintf(stderr, "buffer x: %s\n", st.message.c_str());
    return 1;
  }
  st = welvet::vulkan::buffer_alloc(n * sizeof(float), &y_buf);
  if (!st) {
    welvet::vulkan::buffer_release(&x_buf);
    std::fprintf(stderr, "buffer y: %s\n", st.message.c_str());
    return 1;
  }

  const float host_in[n] = {1, 2, 3, 4};
  float host_out[n] = {};
  st = welvet::vulkan::buffer_upload(&x_buf, host_in, sizeof(host_in));
  if (!st) {
    std::fprintf(stderr, "upload: %s\n", st.message.c_str());
    return 1;
  }

  // y = 2*x + 1 → [3, 5, 7, 9]
  st = welvet::vulkan::launch_smoke_saxpy(x_buf, &y_buf, 2.0f, 1.0f, n);
  if (!st) {
    std::fprintf(stderr, "saxpy: %s\n", st.message.c_str());
    welvet::vulkan::buffer_release(&x_buf);
    welvet::vulkan::buffer_release(&y_buf);
    return 1;
  }

  st = welvet::vulkan::buffer_download(y_buf, host_out, sizeof(host_out));
  welvet::vulkan::buffer_release(&x_buf);
  welvet::vulkan::buffer_release(&y_buf);
  if (!st) {
    std::fprintf(stderr, "download: %s\n", st.message.c_str());
    return 1;
  }
  std::printf("smoke saxpy: [%g, %g, %g, %g] (expect 3,5,7,9)\n", host_out[0], host_out[1],
              host_out[2], host_out[3]);

  std::cout << "\n";
  welvet::gaps::print_census(std::cout);
  const bool ok =
      host_out[0] == 3.f && host_out[1] == 5.f && host_out[2] == 7.f && host_out[3] == 9.f;
  return ok ? 0 : 1;
}
