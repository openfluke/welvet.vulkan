#include "welvet_c.h"

#include "welvet/gaps.hpp"
#include "welvet/status.hpp"
#include "welvet/vulkan/device.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

welvet_status to_c(const welvet::Status& st) {
  welvet_status out{};
  out.code = static_cast<int32_t>(st.code);
  if (!st.message.empty()) {
    std::snprintf(out.message, sizeof(out.message), "%s", st.message.c_str());
  } else {
    out.message[0] = '\0';
  }
  return out;
}

} // namespace

extern "C" {

const char* welvet_version(void) {
  // M0 bedrock — Vulkan host scaffold; engine board 0 / 100.
  return "0.00";
}

const char* welvet_runtime_identity(void) {
  static thread_local std::string id;
  id = welvet::vulkan::runtime_identity();
  return id.c_str();
}

welvet_status welvet_vulkan_init(void) { return to_c(welvet::vulkan::init_default_device()); }

int32_t welvet_vulkan_available(void) { return welvet::vulkan::available() ? 1 : 0; }

int32_t welvet_gaps_count(void) { return static_cast<int32_t>(welvet::gaps::count_gaps()); }

int32_t welvet_gaps_done(void) { return static_cast<int32_t>(welvet::gaps::count_done()); }

} // extern "C"
