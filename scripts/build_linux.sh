#!/usr/bin/env bash
# Linux Vulkan host build for welvet.vulkan — C++ + glslc SPIR-V.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "welvet.vulkan build_linux.sh expects Linux (got $(uname -s))" >&2
  exit 1
fi

# Prefer LunarG SDK when present (headers/tools), else system Vulkan.
if [[ -z "${VULKAN_SDK:-}" ]]; then
  for cand in \
    "${HOME}/VulkanSDK/"*/x86_64 \
    /usr
  do
    if [[ -x "${cand}/bin/glslc" || -x "${cand}/bin/glslangValidator" ]]; then
      # Expand first matching SDK tree.
      if [[ -d "${cand}" ]]; then
        export VULKAN_SDK="${cand}"
        break
      fi
    fi
  done
  # Glob may leave literal; pick newest SDK if needed.
  if [[ -z "${VULKAN_SDK:-}" && -d "${HOME}/VulkanSDK" ]]; then
    newest="$(ls -1d "${HOME}/VulkanSDK/"*/x86_64 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -n "${newest}" && -d "${newest}" ]]; then
      export VULKAN_SDK="${newest}"
    fi
  fi
fi

if [[ -n "${VULKAN_SDK:-}" ]]; then
  export PATH="${VULKAN_SDK}/bin:${PATH}"
  export LD_LIBRARY_PATH="${VULKAN_SDK}/lib:${LD_LIBRARY_PATH:-}"
  # Prefer SDK include/lib for CMake find_package(Vulkan).
  export CMAKE_PREFIX_PATH="${VULKAN_SDK}:${CMAKE_PREFIX_PATH:-}"
fi

if ! command -v glslc >/dev/null 2>&1; then
  echo "glslc not found. Install LunarG Vulkan SDK or glslang, or set VULKAN_SDK." >&2
  exit 1
fi

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD" -j"$(nproc 2>/dev/null || echo 4)"

echo "built: $BUILD"
echo "  lib:  $BUILD/libwelvet_vulkan.so"
echo "  tool: $BUILD/welvet_dense_fwd"
echo "  ex:   $BUILD/example_vulkan_smoke"
echo
echo "smoke:  $BUILD/example_vulkan_smoke"
echo "gaps:   $BUILD/welvet_dense_fwd --gaps"
