#include "welvet/vulkan/device.hpp"

#include "vulkan/internal.hpp"

#include "vulkan/pipelines.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <vector>

namespace welvet::vulkan::detail {
namespace {

std::mutex g_mu;
Runtime g_rt;

bool device_has_compute(VkPhysicalDevice pd, std::uint32_t* family_out) {
  std::uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
  std::vector<VkQueueFamilyProperties> props(count);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, props.data());
  for (std::uint32_t i = 0; i < count; ++i) {
    if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      if (family_out) *family_out = i;
      return true;
    }
  }
  return false;
}

std::size_t heap_bytes(VkPhysicalDevice pd) {
  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(pd, &mem);
  std::size_t total = 0;
  for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
    if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total += static_cast<std::size_t>(mem.memoryHeaps[i].size);
    }
  }
  return total;
}

DeviceInfo make_info(VkPhysicalDevice pd, int index) {
  DeviceInfo info;
  info.index = index;
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(pd, &props);
  info.name = props.deviceName;
  info.api_version = props.apiVersion;
  info.driver_version = props.driverVersion;
  info.vendor_id = props.vendorID;
  info.device_id = props.deviceID;
  info.device_type = props.deviceType;
  info.total_device_mem = heap_bytes(pd);
  (void)device_has_compute(pd, &info.compute_queue_family);
  return info;
}

Status create_instance(Runtime* rt) {
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "welvet.vulkan";
  app.applicationVersion = VK_MAKE_VERSION(0, 0, 0);
  app.pEngineName = "welvet";
  app.engineVersion = VK_MAKE_VERSION(0, 0, 0);
  app.apiVersion = VK_API_VERSION_1_2;

  const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
  const bool want_validation = [] {
    const char* e = std::getenv("WELVET_VULKAN_VALIDATION");
    return e && e[0] == '1';
  }();

  VkInstanceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app;
  if (want_validation) {
    ci.enabledLayerCount = 1;
    ci.ppEnabledLayerNames = layers;
  }

  return vk_check(vkCreateInstance(&ci, nullptr, &rt->instance), "vkCreateInstance");
}

int device_rank(VkPhysicalDeviceType t) {
  switch (t) {
  case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
    return 0;
  case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
    return 1;
  case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
    return 2;
  case VK_PHYSICAL_DEVICE_TYPE_CPU:
    return 3;
  default:
    return 4;
  }
}

Status pick_physical(Runtime* rt) {
  std::uint32_t count = 0;
  auto st = vk_check(vkEnumeratePhysicalDevices(rt->instance, &count, nullptr),
                     "vkEnumeratePhysicalDevices");
  if (!st) return st;
  if (count == 0) {
    return Status::Err(StatusCode::Unsupported, "no Vulkan physical devices");
  }
  std::vector<VkPhysicalDevice> devices(count);
  st = vk_check(vkEnumeratePhysicalDevices(rt->instance, &count, devices.data()),
                "vkEnumeratePhysicalDevices");
  if (!st) return st;

  struct Cand {
    VkPhysicalDevice pd;
    int index;
    int rank;
    DeviceInfo info;
  };
  std::vector<Cand> cands;
  for (std::uint32_t i = 0; i < count; ++i) {
    std::uint32_t family = 0;
    if (!device_has_compute(devices[i], &family)) continue;
    Cand c;
    c.pd = devices[i];
    c.index = static_cast<int>(i);
    c.info = make_info(devices[i], c.index);
    c.rank = device_rank(static_cast<VkPhysicalDeviceType>(c.info.device_type));
    cands.push_back(c);
  }
  if (cands.empty()) {
    return Status::Err(StatusCode::Unsupported, "no Vulkan compute queue family");
  }
  std::stable_sort(cands.begin(), cands.end(),
                   [](const Cand& a, const Cand& b) { return a.rank < b.rank; });

  rt->physical = cands.front().pd;
  rt->info = cands.front().info;
  rt->compute_queue_family = rt->info.compute_queue_family;
  return Status::Ok();
}

Status create_device(Runtime* rt) {
  float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = rt->compute_queue_family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;

  VkPhysicalDeviceFeatures supported{};
  vkGetPhysicalDeviceFeatures(rt->physical, &supported);
  if (!supported.shaderFloat64) {
    return Status::Err(StatusCode::Unsupported,
                       "shaderFloat64 unavailable — required for Welvet f64-accumulate Dense path");
  }
  rt->shader_float64 = true;

  VkPhysicalDeviceFeatures enabled{};
  enabled.shaderFloat64 = VK_TRUE;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pEnabledFeatures = &enabled;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;

  auto st = vk_check(vkCreateDevice(rt->physical, &dci, nullptr, &rt->device), "vkCreateDevice");
  if (!st) return st;
  vkGetDeviceQueue(rt->device, rt->compute_queue_family, 0, &rt->compute_queue);

  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(rt->physical, &mem);
  for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    const auto flags = mem.memoryTypes[i].propertyFlags;
    if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      rt->host_visible_mem_type = i;
      break;
    }
  }
  if (rt->host_visible_mem_type == UINT32_MAX) {
    return Status::Err(StatusCode::Unsupported, "no host-visible coherent memory type");
  }

  VkCommandPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = rt->compute_queue_family;
  st = vk_check(vkCreateCommandPool(rt->device, &pci, nullptr, &rt->command_pool),
                "vkCreateCommandPool");
  if (!st) return st;

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 4096;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dpci.maxSets = 512;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_size;
  return vk_check(vkCreateDescriptorPool(rt->device, &dpci, nullptr, &rt->descriptor_pool),
                  "vkCreateDescriptorPool");
}

void build_identity(Runtime* rt) {
  std::ostringstream os;
  os << "vulkan:" << rt->info.name << ":idx=" << rt->info.index
     << ":api=" << VK_VERSION_MAJOR(rt->info.api_version) << "."
     << VK_VERSION_MINOR(rt->info.api_version) << "." << VK_VERSION_PATCH(rt->info.api_version)
     << ":type=" << rt->info.device_type << ":mem=" << rt->info.total_device_mem
     << ":f64=" << (rt->shader_float64 ? 1 : 0);
  rt->identity = os.str();
}

} // namespace

std::mutex& runtime_mu() { return g_mu; }
Runtime& runtime() { return g_rt; }

Status vk_check(VkResult r, const char* what) {
  if (r == VK_SUCCESS) return Status::Ok();
  return Status::Err(StatusCode::Internal, std::string(what) + ": VkResult=" + std::to_string(r));
}

std::uint32_t find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(runtime().physical, &mem);
  for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) == 0) continue;
    if ((mem.memoryTypes[i].propertyFlags & props) == props) return i;
  }
  return UINT32_MAX;
}

DeviceInfo info_from(VkPhysicalDevice pd, int index) {
  // Re-probe properties without needing the anonymous helpers.
  DeviceInfo info;
  info.index = index;
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(pd, &props);
  info.name = props.deviceName;
  info.api_version = props.apiVersion;
  info.driver_version = props.driverVersion;
  info.vendor_id = props.vendorID;
  info.device_id = props.deviceID;
  info.device_type = props.deviceType;
  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(pd, &mem);
  for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
    if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      info.total_device_mem += static_cast<std::size_t>(mem.memoryHeaps[i].size);
    }
  }
  std::uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(count);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, qprops.data());
  for (std::uint32_t i = 0; i < count; ++i) {
    if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      info.compute_queue_family = i;
      break;
    }
  }
  return info;
}

Status ensure_runtime() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_rt.ready) return Status::Ok();

  auto st = create_instance(&g_rt);
  if (!st) return st;
  st = pick_physical(&g_rt);
  if (!st) return st;
  st = create_device(&g_rt);
  if (!st) return st;
  build_identity(&g_rt);
  g_rt.ready = true;
  // Pipelines compile lazily on first dispatch (avoid long init / lock issues).
  (void)warm_pipelines_after_device;
  return Status::Ok();
}

} // namespace welvet::vulkan::detail

namespace welvet::vulkan {

Status init_default_device() { return detail::ensure_runtime(); }

bool available() {
  std::uint32_t count = 0;
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "welvet.vulkan";
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app;
  VkInstance inst = VK_NULL_HANDLE;
  if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) return false;
  const bool ok = vkEnumeratePhysicalDevices(inst, &count, nullptr) == VK_SUCCESS && count > 0;
  vkDestroyInstance(inst, nullptr);
  return ok;
}

Status default_device_info(DeviceInfo* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null DeviceInfo");
  auto st = init_default_device();
  if (!st) return st;
  std::lock_guard<std::mutex> lock(detail::runtime_mu());
  *out = detail::runtime().info;
  return Status::Ok();
}

std::vector<DeviceInfo> list_devices() {
  std::vector<DeviceInfo> out;
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "welvet.vulkan";
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app;
  VkInstance inst = VK_NULL_HANDLE;
  if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) return out;
  std::uint32_t count = 0;
  if (vkEnumeratePhysicalDevices(inst, &count, nullptr) != VK_SUCCESS || count == 0) {
    vkDestroyInstance(inst, nullptr);
    return out;
  }
  std::vector<VkPhysicalDevice> devices(count);
  if (vkEnumeratePhysicalDevices(inst, &count, devices.data()) != VK_SUCCESS) {
    vkDestroyInstance(inst, nullptr);
    return out;
  }
  out.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    out.push_back(detail::info_from(devices[i], static_cast<int>(i)));
  }
  vkDestroyInstance(inst, nullptr);
  return out;
}

std::string runtime_identity() {
  (void)init_default_device();
  std::lock_guard<std::mutex> lock(detail::runtime_mu());
  return detail::runtime().identity;
}

} // namespace welvet::vulkan
