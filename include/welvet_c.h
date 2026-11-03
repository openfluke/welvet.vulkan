#pragma once

/* Stable C ABI for welvet.vulkan — used by future w2a go_v_vulkan and NAC. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(WELVET_BUILD_SHARED)
#    ifdef WELVET_EXPORTS
#      define WELVET_API __declspec(dllexport)
#    else
#      define WELVET_API __declspec(dllimport)
#    endif
#  else
#    define WELVET_API
#  endif
#else
#  define WELVET_API __attribute__((visibility("default")))
#endif

typedef struct welvet_status {
  int32_t code; /* 0=ok, 1=invalid, 2=unsupported, 3=gap, 4=internal */
  char message[256];
} welvet_status;

WELVET_API const char* welvet_version(void);
WELVET_API const char* welvet_runtime_identity(void);

WELVET_API welvet_status welvet_vulkan_init(void);
WELVET_API int32_t welvet_vulkan_available(void);
WELVET_API int32_t welvet_gaps_count(void);
WELVET_API int32_t welvet_gaps_done(void);

#ifdef __cplusplus
}
#endif
