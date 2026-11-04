#!/usr/bin/env bash
# Cross-compile welvet.vulkan for Android arm64-v8a and/or x86_64 (NDK + host glslc).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<'HELP'
Usage: ./scripts/build_android.sh [arm64|x86_64|all]
  ANDROID_NDK / ANDROID_HOME   NDK root (default: newest under $ANDROID_HOME/ndk)
  ANDROID_PLATFORM             default android-28
  BUILD_DIR_PREFIX             default $ROOT/build-android
HELP
}

ABI_ARG="${1:-all}"
case "$ABI_ARG" in
  -h|--help) usage; exit 0 ;;
  arm64|arm64-v8a) ABIS=(arm64-v8a) ;;
  x86_64|x86-64|amd64) ABIS=(x86_64) ;;
  all) ABIS=(arm64-v8a x86_64) ;;
  *) echo "Unknown ABI: $ABI_ARG" >&2; usage >&2; exit 2 ;;
esac

if [[ -z "${ANDROID_NDK:-}" ]]; then
  if [[ -n "${ANDROID_HOME:-}" && -d "${ANDROID_HOME}/ndk" ]]; then
    ANDROID_NDK="$(ls -1d "${ANDROID_HOME}/ndk/"* 2>/dev/null | sort -V | tail -1 || true)"
  fi
  if [[ -z "${ANDROID_NDK:-}" && -d "${HOME}/Android/Sdk/ndk" ]]; then
    ANDROID_NDK="$(ls -1d "${HOME}/Android/Sdk/ndk/"* 2>/dev/null | sort -V | tail -1 || true)"
  fi
fi
[[ -n "${ANDROID_NDK:-}" && -f "${ANDROID_NDK}/build/cmake/android.toolchain.cmake" ]] \
  || { echo "ANDROID_NDK not found (set ANDROID_NDK or ANDROID_HOME)" >&2; exit 1; }

# Host glslc for SPIR-V embed (cross toolchain has no glslc).
if [[ -z "${VULKAN_SDK:-}" ]]; then
  newest="$(ls -1d "${HOME}/VulkanSDK/"*/x86_64 2>/dev/null | sort -V | tail -1 || true)"
  [[ -n "$newest" ]] && export VULKAN_SDK="$newest"
fi
if [[ -n "${VULKAN_SDK:-}" ]]; then
  export PATH="${VULKAN_SDK}/bin:${PATH}"
fi
command -v glslc >/dev/null 2>&1 || { echo "host glslc required (Vulkan SDK)" >&2; exit 1; }

PLATFORM="${ANDROID_PLATFORM:-android-28}"
PREFIX="${BUILD_DIR_PREFIX:-$ROOT/build-android}"
JOBS="$(nproc 2>/dev/null || echo 4)"
TOOLCHAIN="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"

echo "NDK:      $ANDROID_NDK"
echo "Platform: $PLATFORM"
echo "glslc:    $(command -v glslc)"

for abi in "${ABIS[@]}"; do
  BUILD="${PREFIX}-${abi}"
  echo "=== building ${abi} → ${BUILD} ==="
  cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI="$abi" \
    -DANDROID_PLATFORM="$PLATFORM" \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DWELVET_BUILD_SHARED=ON \
    -DWELVET_BUILD_TOOLS=OFF \
    -DWELVET_BUILD_EXAMPLES=OFF
  cmake --build "$BUILD" --parallel "$JOBS"
  echo "built: $BUILD/libwelvet_vulkan.so"
done
