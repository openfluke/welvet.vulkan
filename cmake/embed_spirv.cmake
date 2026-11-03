# Embed a SPIR-V binary as a C++ byte array header.
if(NOT DEFINED SPV_PATH OR NOT DEFINED OUT_PATH)
  message(FATAL_ERROR "embed_spirv.cmake requires SPV_PATH and OUT_PATH")
endif()

file(READ "${SPV_PATH}" SPV_HEX HEX)
string(LENGTH "${SPV_HEX}" SPV_HEX_LEN)
math(EXPR SPV_BYTES "${SPV_HEX_LEN} / 2")
if(SPV_BYTES EQUAL 0)
  message(FATAL_ERROR "empty SPIR-V at ${SPV_PATH}")
endif()
math(EXPR SPV_MOD "${SPV_BYTES} % 4")
if(NOT SPV_MOD EQUAL 0)
  message(FATAL_ERROR "SPIR-V size not multiple of 4: ${SPV_BYTES}")
endif()

set(BYTES "")
set(COL 0)
set(I 0)
while(I LESS SPV_HEX_LEN)
  string(SUBSTRING "${SPV_HEX}" ${I} 2 BYTE)
  string(APPEND BYTES "0x${BYTE},")
  math(EXPR COL "${COL} + 1")
  if(COL EQUAL 12)
    string(APPEND BYTES "\n  ")
    set(COL 0)
  endif()
  math(EXPR I "${I} + 2")
endwhile()

file(WRITE "${OUT_PATH}"
"#pragma once
#include <cstddef>
#include <cstdint>
namespace welvet::vulkan::spirv {
inline constexpr std::uint8_t kSmokeSaxpy[] = {
  ${BYTES}
};
inline constexpr std::size_t kSmokeSaxpySize = ${SPV_BYTES};
} // namespace welvet::vulkan::spirv
")
