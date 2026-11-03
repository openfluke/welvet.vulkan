#!/usr/bin/env bash
# Build and install this independent backend into a local Go Welvet checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET_GO_ROOT="${WELVET_GO_ROOT:-$ROOT/../welvet}"
if [[ ! -f "$WELVET_GO_ROOT/go.mod" ]]; then
  echo "Go Welvet checkout not found: $WELVET_GO_ROOT" >&2
  exit 1
fi
"$ROOT/scripts/build_linux.sh"
DEST="$WELVET_GO_ROOT/native/vulkan/linux_x86_64"
mkdir -p "$DEST" "$WELVET_GO_ROOT/native/vulkan/include"
cp "$ROOT/build/libwelvet_vulkan.so" "$DEST/libwelvet_vulkan.so"
cp "$ROOT/include/welvet_c.h" "$WELVET_GO_ROOT/native/vulkan/include/welvet_vulkan.h"
VERSION="$("$ROOT/build/welvet_dense_fwd" --version)"
{
  echo "engine=welvet.vulkan"
  echo "version=$VERSION"
  echo "model_abi=1"
  echo "segment_extension=1"
  echo "platform=linux_x86_64"
  echo "source_checkout=$ROOT"
  sha256sum "$DEST/libwelvet_vulkan.so"
} > "$WELVET_GO_ROOT/native/vulkan/MANIFEST.txt"
(
  cd "$DEST"
  sha256sum libwelvet_vulkan.so > SHA256SUMS
)
echo "Installed $DEST/libwelvet_vulkan.so"
echo "Go build: go build -tags welvet_vulkan ./..."
