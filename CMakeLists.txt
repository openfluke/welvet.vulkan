cmake_minimum_required(VERSION 3.20)
project(welvet_vulkan VERSION 0.79.1 LANGUAGES CXX)

# Cross-vendor Welvet engine: C++ host + Vulkan compute.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WELVET_BUILD_SHARED "Build shared library (C ABI / NAC)" ON)
option(WELVET_BUILD_TOOLS "Build CLI tools for w2a go_v_vulkan" ON)
option(WELVET_BUILD_EXAMPLES "Build standalone Vulkan smoke examples" ON)

# Desktop: LunarG / system Vulkan. Android NDK: sysroot headers + stub libvulkan.so.
if(ANDROID)
  if(NOT Vulkan_INCLUDE_DIR)
    find_path(Vulkan_INCLUDE_DIR NAMES vulkan/vulkan.h
      PATHS "${CMAKE_SYSROOT}/usr/include"
      NO_DEFAULT_PATH
      REQUIRED)
  endif()
  if(NOT Vulkan_LIBRARY)
    find_library(Vulkan_LIBRARY NAMES vulkan REQUIRED)
  endif()
  if(NOT TARGET Vulkan::Vulkan)
    add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
    set_target_properties(Vulkan::Vulkan PROPERTIES
      IMPORTED_LOCATION "${Vulkan_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}")
  endif()
else()
  find_package(Vulkan REQUIRED)
endif()

# SPIR-V compile always uses the host glslc (even when cross-compiling for Android).
find_program(WELVET_GLSLC glslc
  HINTS
    "$ENV{VULKAN_SDK}/bin"
    "$ENV{HOME}/VulkanSDK/*/x86_64/bin"
  REQUIRED
)

include(GNUInstallDirs)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Compile + embed every shaders/*.comp into generated/embedded_spirv.hpp
file(GLOB WELVET_COMP_SHADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp")
file(GLOB WELVET_SHADER_INCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.inc")
set(WELVET_SPV_FILES "")
foreach(COMP ${WELVET_COMP_SHADERS})
  get_filename_component(NAME_WE "${COMP}" NAME_WE)
  set(SPV "${CMAKE_CURRENT_BINARY_DIR}/generated/${NAME_WE}.spv")
  add_custom_command(
    OUTPUT "${SPV}"
    COMMAND "${WELVET_GLSLC}" -fshader-stage=compute -O --target-env=vulkan1.2
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "${COMP}" -o "${SPV}"
    DEPENDS "${COMP}" ${WELVET_SHADER_INCS}
    COMMENT "Compile ${NAME_WE}.comp → SPIR-V"
    VERBATIM
  )
  list(APPEND WELVET_SPV_FILES "${SPV}")
endforeach()

set(WELVET_EMBEDDED_SPIRV "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_spirv.hpp")
add_custom_command(
  OUTPUT "${WELVET_EMBEDDED_SPIRV}"
  COMMAND "${CMAKE_COMMAND}"
    "-DSPV_DIR=${CMAKE_CURRENT_BINARY_DIR}/generated"
    "-DOUT_PATH=${WELVET_EMBEDDED_SPIRV}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
  DEPENDS ${WELVET_SPV_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
  COMMENT "Embed SPIR-V blobs into embedded_spirv.hpp"
  VERBATIM
)
add_custom_target(welvet_embed_spirv DEPENDS "${WELVET_EMBEDDED_SPIRV}")

set(WELVET_CORE_SOURCES
  src/core/status.cpp
  src/core/dtype.cpp
  src/core/activate.cpp
  src/core/tensor.cpp
  src/core/float_codecs.cpp
  src/vulkan/device.cpp
  src/vulkan/buffer.cpp
  src/vulkan/kernels.cpp
  src/weights/store.cpp
  src/weights/native_codec.cpp
  src/layers/dense/dense.cpp
  src/layers/dense/session.cpp
  src/layers/rmsnorm/rmsnorm.cpp
  src/layers/layernorm/layernorm.cpp
  src/layers/softmax/softmax.cpp
  src/layers/swiglu/swiglu.cpp
  src/layers/embedding/embedding.cpp
  src/layers/mha/mha.cpp
  src/layers/cnn1/cnn1.cpp
  src/layers/cnn2/cnn2.cpp
  src/layers/cnn3/cnn3.cpp
  src/layers/convt1/convt1.cpp
  src/layers/convt2/convt2.cpp
  src/layers/convt3/convt3.cpp
  src/layers/rnn/rnn.cpp
  src/layers/lstm/lstm.cpp
  src/layers/sequential/sequential.cpp
  src/layers/residual/residual.cpp
  src/layers/stack/stack.cpp
  src/layers/mamba/mamba.cpp
  src/layers/metacognition/metacognition.cpp
  src/layers/gdn/gdn.cpp
  src/layers/kmeans/kmeans.cpp
  src/layers/parallel/parallel.cpp
  src/architecture/grid.cpp
  src/runtime/forward.cpp
  src/runtime/backward.cpp
  src/runtime/training.cpp
  src/runtime/line.cpp
  src/runtime/step.cpp
  src/runtime/tween_train.cpp
  src/runtime/tween_chain_train.cpp
  src/runtime/tween_split_train.cpp
  src/runtime/tween_alt_train.cpp
  src/runtime/cameral_train.cpp
  src/systems/tween.cpp
  src/systems/tween_chain.cpp
  src/systems/tween_split.cpp
  src/systems/tween_alt.cpp
  src/systems/cam_sync.cpp
  src/gaps/registry.cpp
  src/interop/c_abi.cpp
  src/interop/model_session.cpp
)

# Optional ops translation units added by the Vulkan kernel port
foreach(extra IN ITEMS
    src/vulkan/matvec.cpp
    src/vulkan/pipelines.cpp
    src/vulkan/launch_utils.cpp
    src/vulkan/launches.cpp
  )
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${extra}")
    list(APPEND WELVET_CORE_SOURCES "${extra}")
  endif()
endforeach()

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
# OBJECT libs don't propagate object files via ALIAS alone — link objects explicitly
set_target_properties(welvet PROPERTIES
  OUTPUT_NAME welvet_vulkan
  POSITION_INDEPENDENT_CODE ON
)

function(welvet_link_engine target)
  target_link_libraries(${target} PRIVATE welvet)
endfunction()

if(WELVET_BUILD_TOOLS)
  add_executable(welvet_dense_fwd
    tools/dense_fwd/main.cpp
    tools/dense_fwd/volume_fwd.cpp
    tools/dense_fwd/stepbp_wires.cpp
  )
  target_include_directories(welvet_dense_fwd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/dense_fwd)
  welvet_link_engine(welvet_dense_fwd)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(welvet_dense_fwd PRIVATE -ffp-contract=off)
  endif()
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
