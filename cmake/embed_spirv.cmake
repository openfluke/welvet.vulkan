# Embed all *.spv under SPV_DIR as C++ byte arrays in OUT_PATH.
if(NOT DEFINED SPV_DIR OR NOT DEFINED OUT_PATH)
  message(FATAL_ERROR "embed_spirv.cmake requires SPV_DIR and OUT_PATH")
endif()

file(GLOB SPV_FILES "${SPV_DIR}/*.spv")
if(NOT SPV_FILES)
  message(FATAL_ERROR "no SPIR-V files in ${SPV_DIR}")
endif()

set(BODY "")
foreach(SPV_PATH IN LISTS SPV_FILES)
  get_filename_component(NAME_WE "${SPV_PATH}" NAME_WE)
  string(REGEX REPLACE "^welvet_" "" BASE "${NAME_WE}")
  string(REPLACE "_" ";" PARTS "${BASE}")
  set(SYM "k")
  set(FUN "")
  foreach(P IN LISTS PARTS)
    string(SUBSTRING "${P}" 0 1 FIRST)
    string(TOUPPER "${FIRST}" FIRST)
    string(SUBSTRING "${P}" 1 -1 REST)
    string(APPEND SYM "${FIRST}${REST}")
    string(APPEND FUN "${FIRST}${REST}")
  endforeach()

  file(READ "${SPV_PATH}" SPV_HEX HEX)
  string(LENGTH "${SPV_HEX}" SPV_HEX_LEN)
  math(EXPR SPV_BYTES "${SPV_HEX_LEN} / 2")
  if(SPV_BYTES EQUAL 0)
    message(FATAL_ERROR "empty SPIR-V at ${SPV_PATH}")
  endif()
  math(EXPR SPV_MOD "${SPV_BYTES} % 4")
  if(NOT SPV_MOD EQUAL 0)
    message(FATAL_ERROR "SPIR-V size not multiple of 4: ${SPV_PATH}")
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

  string(APPEND BODY
"inline constexpr std::uint8_t ${SYM}[] = {
  ${BYTES}
};
inline constexpr std::size_t ${SYM}Size = ${SPV_BYTES};
inline Blob ${FUN}() { return {${SYM}, ${SYM}Size}; }

")
endforeach()

file(WRITE "${OUT_PATH}"
"#pragma once
#include <cstddef>
#include <cstdint>
namespace welvet::vulkan::spirv {

struct Blob {
  const std::uint8_t* data;
  std::size_t size;
};

${BODY}} // namespace welvet::vulkan::spirv
")
