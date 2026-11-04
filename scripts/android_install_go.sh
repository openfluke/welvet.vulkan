#!/usr/bin/env bash
# Build Android arm64-v8a + x86_64 libs and install into a local Go Welvet checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET_GO_ROOT="${WELVET_GO_ROOT:-$ROOT/../welvet}"
if [[ ! -f "$WELVET_GO_ROOT/go.mod" ]]; then
  echo "Go Welvet checkout not found: $WELVET_GO_ROOT" >&2
  exit 1
fi

if [[ -z "${ANDROID_NDK:-}" ]]; then
  if [[ -n "${ANDROID_HOME:-}" && -d "${ANDROID_HOME}/ndk" ]]; then
    ANDROID_NDK="$(ls -1d "${ANDROID_HOME}/ndk/"* 2>/dev/null | sort -V | tail -1 || true)"
  fi
  if [[ -z "${ANDROID_NDK:-}" && -d "${HOME}/Android/Sdk/ndk" ]]; then
    ANDROID_NDK="$(ls -1d "${HOME}/Android/Sdk/ndk/"* 2>/dev/null | sort -V | tail -1 || true)"
  fi
fi
export ANDROID_NDK

"$ROOT/scripts/build_android.sh" all

PREFIX="${BUILD_DIR_PREFIX:-$ROOT/build-android}"
INCLUDE_DIR="$WELVET_GO_ROOT/native/vulkan/include"
mkdir -p "$INCLUDE_DIR"
cp "$ROOT/include/welvet_c.h" "$INCLUDE_DIR/welvet_vulkan.h"

VERSION="0.79.1"
if [[ -x "$ROOT/build/welvet_dense_fwd" ]]; then
  VERSION="$("$ROOT/build/welvet_dense_fwd" --version)"
fi

STRIP_BIN=""
for cand in \
  "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" \
  "$(command -v llvm-strip 2>/dev/null || true)" \
  "$(command -v strip || true)"
do
  if [[ -n "$cand" && -x "$cand" ]]; then
    STRIP_BIN="$cand"
    break
  fi
done

install_abi() {
  local abi="$1" dest_name="$2"
  local src="${PREFIX}-${abi}/libwelvet_vulkan.so"
  local dest_dir="$WELVET_GO_ROOT/native/vulkan/${dest_name}"
  [[ -f "$src" ]] || { echo "missing $src" >&2; exit 1; }
  mkdir -p "$dest_dir"
  if [[ -n "$STRIP_BIN" ]]; then
    "$STRIP_BIN" --strip-unneeded -o "$dest_dir/libwelvet_vulkan.so" "$src"
  else
    cp "$src" "$dest_dir/libwelvet_vulkan.so"
  fi
  (
    cd "$dest_dir"
    sha256sum libwelvet_vulkan.so > SHA256SUMS
  )
  cat > "$dest_dir/README.md" <<EOF
# ${dest_name}

Shipped \`libwelvet_vulkan.so\` for Android (${abi}, API 28+). Artifacts for
\`jniLibs\` / NDK packaging. Go \`nativevulkan\` can dlopen this path on
\`GOOS=android\` when cgo is enabled.

\`\`\`bash
cd welvet.vulkan
./scripts/android_install_go.sh
\`\`\`

Engine pin: see \`../MANIFEST.txt\`.
EOF
  echo "Installed $dest_dir/libwelvet_vulkan.so"
}

install_abi arm64-v8a android_arm64
install_abi x86_64 android_x86_64

{
  echo "engine=welvet.vulkan"
  echo "version=$VERSION"
  echo "model_abi=1"
  echo "segment_extension=1"
  echo "platforms=linux_x86_64,android_arm64,android_x86_64"
  echo "source_checkout=$ROOT"
  if [[ -f "$WELVET_GO_ROOT/native/vulkan/linux_x86_64/libwelvet_vulkan.so" ]]; then
    sha256sum "$WELVET_GO_ROOT/native/vulkan/linux_x86_64/libwelvet_vulkan.so"
  fi
  sha256sum "$WELVET_GO_ROOT/native/vulkan/android_arm64/libwelvet_vulkan.so"
  sha256sum "$WELVET_GO_ROOT/native/vulkan/android_x86_64/libwelvet_vulkan.so"
} > "$WELVET_GO_ROOT/native/vulkan/MANIFEST.txt"

echo "Go: drop into jniLibs or set WELVET_VULKAN_LIBRARY; nativevulkan discovers android_* on GOOS=android"
