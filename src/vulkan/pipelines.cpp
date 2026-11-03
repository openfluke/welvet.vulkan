#include "vulkan/pipelines.hpp"

#include "generated/embedded_spirv.hpp"
#include "vulkan/internal.hpp"

#include <array>
#include <mutex>
#include <vector>

namespace welvet::vulkan::detail {
namespace {

struct PipelineSpec {
  PipelineId id;
  spirv::Blob (*spirv)();
  std::uint32_t bindings;
  std::uint32_t push_size;
};

constexpr PipelineSpec kSpecs[] = {
    {PipelineId::SmokeSaxpy, spirv::SmokeSaxpy, 2, sizeof(float) * 2 + sizeof(std::uint32_t)},
    {PipelineId::DenseMatvec, spirv::DenseMatvec, 3, sizeof(int) * 3},
    {PipelineId::DenseBiasAct, spirv::DenseBiasAct, 3, sizeof(int) * 5},
    {PipelineId::DenseActDeriv, spirv::DenseActDeriv, 3, sizeof(int) * 2},
    {PipelineId::DenseMatvect, spirv::DenseMatvect, 3, sizeof(int) * 3},
    {PipelineId::DenseOuter, spirv::DenseOuter, 3, sizeof(int) * 3},
    {PipelineId::NormForward, spirv::NormForward, 5, 24},
    // C++ NormBwdPush is 24 (alignas double); SPIR-V block is 20 without trailing pad.
    {PipelineId::NormBackward, spirv::NormBackward, 5, 24},
    {PipelineId::NormParamGrad, spirv::NormParamGrad, 3, sizeof(int) * 3},
    {PipelineId::SoftmaxForward, spirv::SoftmaxForward, 5, 40},
    {PipelineId::SoftmaxBackward, spirv::SoftmaxBackward, 4, 28},
    {PipelineId::SwigluProduct, spirv::SwigluProduct, 3, sizeof(int)},
    {PipelineId::SwigluBwd, spirv::SwigluBwd, 5, sizeof(int)},
    {PipelineId::EmbeddingGather, spirv::EmbeddingGather, 3, sizeof(int) * 5},
    {PipelineId::EmbeddingScatter, spirv::EmbeddingScatter, 3, sizeof(int) * 4},
    {PipelineId::MhaForward, spirv::MhaForward, 9, sizeof(int) * 12 + sizeof(double) * 2},
    {PipelineId::MhaBackward, spirv::MhaBackward, 17, sizeof(int) * 12 + sizeof(double) * 2},
    {PipelineId::Convt1, spirv::Convt1, 4, sizeof(int) * 9},
    {PipelineId::Convt2, spirv::Convt2, 4, sizeof(int) * 11},
    {PipelineId::Convt3, spirv::Convt3, 4, sizeof(int) * 13},
    {PipelineId::ProjectGap, spirv::ProjectGap, 2, sizeof(std::uint32_t) * 2},
    {PipelineId::CamShadow, spirv::CamShadow, 3, 16},
    {PipelineId::CamGateGap, spirv::CamGateGap, 2, sizeof(std::uint32_t) * 2},
    {PipelineId::CamBlend, spirv::CamBlend, 3, sizeof(std::uint32_t) * 2},
    {PipelineId::CamBiasBlend, spirv::CamBiasBlend, 3, sizeof(std::uint32_t) * 2},
};

static_assert(sizeof(kSpecs) / sizeof(kSpecs[0]) ==
              static_cast<std::size_t>(PipelineId::Count));

ComputePipeline g_pipelines[static_cast<std::size_t>(PipelineId::Count)]{};
const PipelineSpec& spec(PipelineId id) { return kSpecs[static_cast<std::size_t>(id)]; }

Status create_one(PipelineId id) {
  auto& slot = g_pipelines[static_cast<std::size_t>(id)];
  if (slot.pipeline) return Status::Ok();
  const auto& s = spec(id);
  const spirv::Blob blob = s.spirv();
  auto& rt = runtime();

  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = blob.size;
  smci.pCode = reinterpret_cast<const std::uint32_t*>(blob.data);
  auto st = vk_check(vkCreateShaderModule(rt.device, &smci, nullptr, &slot.module),
                     "vkCreateShaderModule");
  if (!st) return st;

  std::vector<VkDescriptorSetLayoutBinding> binds(s.bindings);
  for (std::uint32_t i = 0; i < s.bindings; ++i) {
    binds[i].binding = i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = s.bindings;
  dslci.pBindings = binds.data();
  st = vk_check(vkCreateDescriptorSetLayout(rt.device, &dslci, nullptr, &slot.set_layout),
                "vkCreateDescriptorSetLayout");
  if (!st) return st;

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = s.push_size;

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &slot.set_layout;
  plci.pushConstantRangeCount = s.push_size ? 1u : 0u;
  plci.pPushConstantRanges = s.push_size ? &pcr : nullptr;
  st = vk_check(vkCreatePipelineLayout(rt.device, &plci, nullptr, &slot.layout),
                "vkCreatePipelineLayout");
  if (!st) return st;

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = slot.module;
  cpci.stage.pName = "main";
  cpci.layout = slot.layout;
  st = vk_check(vkCreateComputePipelines(rt.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &slot.pipeline),
                "vkCreateComputePipelines");
  if (!st) return st;

  slot.binding_count = s.bindings;
  slot.push_size = s.push_size;
  return Status::Ok();
}

} // namespace

Status ensure_pipeline(PipelineId id, ComputePipeline* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null pipeline out");
  auto st = ensure_runtime();
  if (!st) return st;
  std::lock_guard<std::mutex> lock(runtime_mu());
  st = create_one(id);
  if (!st) return st;
  *out = g_pipelines[static_cast<std::size_t>(id)];
  return Status::Ok();
}

Status dispatch_compute(PipelineId id, const VkDescriptorBufferInfo* buffers,
                        std::uint32_t binding_count, const void* push, std::size_t push_size,
                        std::uint32_t groups_x, std::uint32_t groups_y, std::uint32_t groups_z) {
  auto st = ensure_runtime();
  if (!st) return st;

  auto& rt = runtime();
  std::lock_guard<std::mutex> lock(runtime_mu());
  st = create_one(id);
  if (!st) return st;
  ComputePipeline pipe = g_pipelines[static_cast<std::size_t>(id)];
  if (binding_count != pipe.binding_count) {
    return Status::Err(StatusCode::Internal, "dispatch binding count mismatch");
  }
  if (push_size != pipe.push_size) {
    return Status::Err(StatusCode::Internal, "dispatch push constant size mismatch");
  }

  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = rt.descriptor_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &pipe.set_layout;
  VkDescriptorSet set = VK_NULL_HANDLE;
  st = vk_check(vkAllocateDescriptorSets(rt.device, &dsai, &set), "vkAllocateDescriptorSets");
  if (!st) return st;

  std::vector<VkWriteDescriptorSet> writes(binding_count);
  for (std::uint32_t i = 0; i < binding_count; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &buffers[i];
  }
  vkUpdateDescriptorSets(rt.device, binding_count, writes.data(), 0, nullptr);

  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = rt.command_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  st = vk_check(vkAllocateCommandBuffers(rt.device, &cbai, &cmd), "vkAllocateCommandBuffers");
  if (!st) {
    vkFreeDescriptorSets(rt.device, rt.descriptor_pool, 1, &set);
    return st;
  }

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  st = vk_check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
  if (!st) {
    vkFreeCommandBuffers(rt.device, rt.command_pool, 1, &cmd);
    vkFreeDescriptorSets(rt.device, rt.descriptor_pool, 1, &set);
    return st;
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &set, 0, nullptr);
  if (push_size) {
    vkCmdPushConstants(cmd, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<std::uint32_t>(push_size), push);
  }
  vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
  st = vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
  if (!st) {
    vkFreeCommandBuffers(rt.device, rt.command_pool, 1, &cmd);
    vkFreeDescriptorSets(rt.device, rt.descriptor_pool, 1, &set);
    return st;
  }

  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  st = vk_check(vkQueueSubmit(rt.compute_queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
  if (st) st = vk_check(vkQueueWaitIdle(rt.compute_queue), "vkQueueWaitIdle");

  vkFreeCommandBuffers(rt.device, rt.command_pool, 1, &cmd);
  vkFreeDescriptorSets(rt.device, rt.descriptor_pool, 1, &set);
  return st;
}

} // namespace welvet::vulkan::detail

namespace welvet::vulkan::detail {

void warm_pipelines_after_device() {
  // Caller must hold runtime_mu() (ensure_runtime does).
  for (std::size_t i = 0; i < static_cast<std::size_t>(PipelineId::Count); ++i) {
    (void)create_one(static_cast<PipelineId>(i));
  }
}

} // namespace welvet::vulkan::detail
