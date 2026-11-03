#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "welvet/status.hpp"

namespace welvet::vulkan {

struct DeviceInfo {
  int index = -1;
  std::string name;
  std::uint32_t api_version = 0;
  std::uint32_t driver_version = 0;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  std::uint32_t device_type = 0; // VkPhysicalDeviceType
  std::size_t total_device_mem = 0;
  std::uint32_t compute_queue_family = 0;
};

// Process-wide default instance + compute device.
Status init_default_device();
bool available();
Status default_device_info(DeviceInfo* out);
std::vector<DeviceInfo> list_devices();

// Device + loader identity for provenance in w2a logs.
std::string runtime_identity();

} // namespace welvet::vulkan
