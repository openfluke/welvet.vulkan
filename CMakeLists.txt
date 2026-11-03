cmake_minimum_required(VERSION 3.20)
project(welvet_vulkan VERSION 0.0 LANGUAGES CXX)

# Cross-vendor Welvet engine: C++ host + Vulkan compute.
# Primary host: Linux x86_64. Windows later.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WELVET_BUILD_SHARED "Build shared library (C ABI / NAC)" ON)
option(WELVET_BUILD_TOOLS "Build CLI tools for w2a go_v_vulkan" ON)
option(WELVET_BUILD_EXAMPLES "Build standalone Vulkan smoke examples" ON)

find_package(Vulkan REQUIRED)

find_program(WELVET_GLSLC glslc
  HINTS
    "$ENV{VULKAN_SDK}/bin"
    "$ENV{HOME}/VulkanSDK/*/x86_64/bin"
  REQUIRED
)

include(GNUInstallDirs)

set(WELVET_SMOKE_COMP "${CMAKE_CURRENT_SOURCE_DIR}/shaders/welvet_smoke.comp")
set(WELVET_SMOKE_SPV "${CMAKE_CURRENT_BINARY_DIR}/generated/welvet_smoke.spv")
set(WELVET_EMBEDDED_SPIRV "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_spirv.hpp")

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
add_custom_command(
  OUTPUT "${WELVET_SMOKE_SPV}"
  COMMAND "${WELVET_GLSLC}" -fshader-stage=compute -O "${WELVET_SMOKE_COMP}" -o "${WELVET_SMOKE_SPV}"
  DEPENDS "${WELVET_SMOKE_COMP}"
  COMMENT "Compile welvet_smoke.comp → SPIR-V"
  VERBATIM
)

add_custom_command(
  OUTPUT "${WELVET_EMBEDDED_SPIRV}"
  COMMAND "${CMAKE_COMMAND}"
    "-DSPV_PATH=${WELVET_SMOKE_SPV}"
    "-DOUT_PATH=${WELVET_EMBEDDED_SPIRV}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
  DEPENDS "${WELVET_SMOKE_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
  COMMENT "Embed SPIR-V into embedded_spirv.hpp"
  VERBATIM
)

set(WELVET_CORE_SOURCES
  src/core/status.cpp
  src/vulkan/device.cpp
  src/vulkan/buffer.cpp
  src/vulkan/kernels.cpp
  src/gaps/registry.cpp
  src/interop/c_abi.cpp
)

add_custom_target(welvet_embed_spirv DEPENDS "${WELVET_EMBEDDED_SPIRV}")

add_library(welvet_vulkan_obj OBJECT ${WELVET_CORE_SOURCES})
add_dependencies(welvet_vulkan_obj welvet_embed_spirv)
set_target_properties(welvet_vulkan_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(welvet_vulkan_obj
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_definitions(welvet_vulkan_obj PRIVATE WELVET_VULKAN=1)
if(WELVET_BUILD_SHARED)
  target_compile_definitions(welvet_vulkan_obj PRIVATE WELVET_BUILD_SHARED WELVET_EXPORTS)
endif()
target_link_libraries(welvet_vulkan_obj PUBLIC Vulkan::Vulkan)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(welvet_vulkan_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
elseif(MSVC)
  target_compile_options(welvet_vulkan_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/fp:precise>)
endif()

if(WELVET_BUILD_SHARED)
  add_library(welvet SHARED $<TARGET_OBJECTS:welvet_vulkan_obj>)
else()
  add_library(welvet STATIC $<TARGET_OBJECTS:welvet_vulkan_obj>)
endif()
add_library(welvet::welvet ALIAS welvet)
target_include_directories(welvet
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(welvet PUBLIC Vulkan::Vulkan)
set_target_properties(welvet PROPERTIES
  OUTPUT_NAME welvet_vulkan
  POSITION_INDEPENDENT_CODE ON
)
if(WIN32)
  set_target_properties(welvet PROPERTIES PREFIX "lib")
  if(WELVET_BUILD_SHARED)
    target_compile_definitions(welvet PRIVATE WELVET_BUILD_SHARED WELVET_EXPORTS)
  endif()
endif()

function(welvet_link_engine target)
  target_link_libraries(${target} PRIVATE welvet)
endfunction()

if(WELVET_BUILD_TOOLS)
  add_executable(welvet_dense_fwd tools/dense_fwd/main.cpp)
  welvet_link_engine(welvet_dense_fwd)
endif()

if(WELVET_BUILD_EXAMPLES)
  add_executable(example_vulkan_smoke examples/vulkan_smoke/main.cpp)
  welvet_link_engine(example_vulkan_smoke)
endif()

install(TARGETS welvet
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY shaders/ DESTINATION ${CMAKE_INSTALL_DATADIR}/welvet/shaders)
if(WELVET_BUILD_TOOLS)
  install(TARGETS welvet_dense_fwd RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

message(STATUS "welvet.vulkan platform: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Vulkan: ${Vulkan_VERSION}")
message(STATUS "glslc: ${WELVET_GLSLC}")
message(STATUS "CXX: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
