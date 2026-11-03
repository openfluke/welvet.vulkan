#pragma once

#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan.h>

#include "welvet/status.hpp"

namespace welvet::vulkan::detail {

enum class PipelineId : std::uint32_t {
  SmokeSaxpy,
  DenseMatvec,
  DenseBiasAct,
  DenseActDeriv,
  DenseMatvect,
  DenseOuter,
  NormForward,
  NormBackward,
  NormParamGrad,
  SoftmaxForward,
  SoftmaxBackward,
  SwigluProduct,
  SwigluBwd,
  EmbeddingGather,
  EmbeddingScatter,
  MhaForward,
  MhaBackward,
  Convt1,
  Convt2,
  Convt3,
  ProjectGap,
  CamShadow,
  CamGateGap,
  CamBlend,
  CamBiasBlend,
  Count
};

struct ComputePipeline {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VkShaderModule module = VK_NULL_HANDLE;
  std::uint32_t binding_count = 0;
  std::uint32_t push_size = 0;
};

Status ensure_pipeline(PipelineId id, ComputePipeline* out);
Status dispatch_compute(PipelineId id, const VkDescriptorBufferInfo* buffers,
                        std::uint32_t binding_count, const void* push, std::size_t push_size,
                        std::uint32_t groups_x, std::uint32_t groups_y = 1,
                        std::uint32_t groups_z = 1);

} // namespace welvet::vulkan::detail
