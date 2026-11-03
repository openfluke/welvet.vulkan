#include "welvet/gaps.hpp"

#include <iostream>

namespace welvet::gaps {
namespace {

const std::vector<Entry>& build() {
  static const std::vector<Entry> k = {
      // --- M0 Vulkan host (this scaffold) ---
      {"vulkan.device.init", "vulkan", "VkInstance/device probe + runtime identity", true},
      {"vulkan.shaders.compile", "vulkan", "glslc embed SPIR-V for welvet_smoke_saxpy", true},
      {"vulkan.buffer.host_visible", "vulkan", "host-visible buffer alloc/upload/download", true},
      {"vulkan.kernel.smoke", "vulkan", "Dispatch welvet_smoke_saxpy compute", true},
      {"abi.version", "interop", "welvet_version + welvet_vulkan_init C ABI", true},
      {"platform.linux", "platform", "Linux x86_64 Vulkan host build", true},

      // --- Core / weights (port semantics from siblings; compute on Vulkan) ---
      {"core.dtype.all34", "core", "gap? all 34 DType names + FormatNone Dense forward gates", false},
      {"core.activate", "core", "gap? Activate/ActivateDeriv helpers (GPU act in Dense kernel)",
       false},
      {"core.fp_codecs", "core", "gap? FP8/FP4/NF4/FP6 pack-unpack for FormatNone storage", false},
      {"weights.formatnone.x34", "weights", "gap? FormatNone pack/decode ×34 for Dense forward",
       false},
      {"weights.matvec.vulkan", "weights", "gap? resident Dense MatVec + bias/act on Vulkan", false},
      {"weights.resident", "weights", "gap? upload-once W/bias; warm execute reuses device buffers",
       false},
      {"vulkan.no_fallback", "vulkan",
       "gap? CPU numerical layer paths rejected; no runtime opt-out", false},

      // --- Layers (same 23 kinds as welvet.cpp / Go / welvet.cuda) ---
      {"dense.forward.vulkan", "layers", "gap? Dense FormatNone×34 forward", false},
      {"dense.normalbp.vulkan", "layers", "gap? Dense NormalBP dX/dW + FormatNone SGD", false},
      {"mha.forward.vulkan", "layers", "gap? MHA ModeSelf FormatNone×34 forward", false},
      {"mha.normalbp.vulkan", "layers", "gap? MHA NormalBP ModeSelf + RoPEᵀ", false},
      {"swiglu.forward.vulkan", "layers", "gap? SwiGLU FormatNone×34 forward", false},
      {"swiglu.normalbp.vulkan", "layers", "gap? SwiGLU NormalBP", false},
      {"rmsnorm.vulkan", "layers", "gap? RMSNorm FormatNone×34 forward", false},
      {"rmsnorm.normalbp.vulkan", "layers", "gap? RMSNorm NormalBP", false},
      {"layernorm.vulkan", "layers", "gap? LayerNorm FormatNone×34 forward", false},
      {"layernorm.normalbp.vulkan", "layers", "gap? LayerNorm NormalBP", false},
      {"softmax.vulkan", "layers", "gap? Softmax FormatNone×34 forward", false},
      {"softmax.normalbp.vulkan", "layers", "gap? Softmax NormalBP dX-only", false},
      {"embedding.forward.vulkan", "layers", "gap? Embedding FormatNone×34 gather", false},
      {"embedding.normalbp.vulkan", "layers", "gap? Embedding scatter NormalBP", false},
      {"cnn.vulkan", "layers", "gap? CNN1/2/3 FormatNone×34 forward", false},
      {"cnn.normalbp.vulkan", "layers", "gap? CNN1/2/3 NormalBP", false},
      {"convt.vulkan", "layers", "gap? ConvT1/2/3 FormatNone×34 forward", false},
      {"convt.normalbp.vulkan", "layers", "gap? ConvT1/2/3 NormalBP", false},
      {"rnn.vulkan", "layers", "gap? RNN FormatNone×34 forward", false},
      {"rnn.normalbp.vulkan", "layers", "gap? RNN BPTT NormalBP", false},
      {"lstm.vulkan", "layers", "gap? LSTM FormatNone×34 forward", false},
      {"lstm.normalbp.vulkan", "layers", "gap? LSTM BPTT NormalBP", false},
      {"sequential.vulkan", "layers", "gap? Sequential Dense chain forward", false},
      {"sequential.normalbp.vulkan", "layers", "gap? Sequential NormalBP", false},
      {"residual.vulkan", "layers", "gap? Residual F(x)+x forward", false},
      {"residual.normalbp.vulkan", "layers", "gap? Residual NormalBP", false},
      {"kmeans.vulkan", "layers", "gap? KMeans soft assignment forward", false},
      {"kmeans.normalbp.vulkan", "layers", "gap? KMeans NormalBP", false},
      {"mamba.vulkan", "layers", "gap? Mamba Dense+scan forward", false},
      {"mamba.normalbp.vulkan", "layers", "gap? Mamba NormalBP", false},
      {"metacognition.vulkan", "layers", "gap? Metacognition forward", false},
      {"metacognition.normalbp.vulkan", "layers", "gap? Metacognition NormalBP", false},
      {"gdn.vulkan", "layers", "gap? GDN Float32-only forward", false},
      {"gdn.normalbp.vulkan", "layers", "gap? GDN truncated-BPTT NormalBP", false},
      {"stack.vulkan", "layers", "gap? Stack NestedStack-of-Dense", false},
      {"stack.normalbp.vulkan", "layers", "gap? Stack NormalBP", false},
      {"parallel.vulkan", "layers", "gap? Parallel Dense branches + combine", false},
      {"parallel.normalbp.vulkan", "layers", "gap? Parallel NormalBP", false},
      {"seqmix.vulkan", "layers", "gap? seqmix Kind name smoke", false},

      // --- Runtime / architecture / training (climb toward cuda 0.79) ---
      {"arch.grid", "architecture", "gap? Grid topology + binds", false},
      {"runtime.forward", "runtime", "gap? volume forward", false},
      {"runtime.backward", "runtime", "gap? volume backward", false},
      {"runtime.step", "runtime", "gap? Line StepBP FamilyBP", false},
      {"runtime.mesh", "runtime", "gap? Heterogeneous MeshBP", false},
      {"train.modes.x29", "training", "gap? all 29 TrainModes + Freeze/Shadow/Adv/Memory", false},
      {"train.advanced", "training", "gap? Freeze/Shadow/Adversarial/Memory + mixed branches",
       false},
      {"train.camsync", "training", "gap? CamSync/BlendStores", false},
      {"abi.model_session", "abi", "gap? welvet_model_* Mount/Forward/TrainMSE/Export", false},

      // --- Systems / model ---
      {"systems.tween", "systems", "gap? ModeTween projected credit", false},
      {"systems.tween_chain", "systems", "gap? ModeTweenChain FamilyBP", false},
      {"systems.tween_split", "systems", "gap? ModeTweenSplit", false},
      {"systems.tween_alt", "systems", "gap? ModeTweenAlt", false},
      {"systems.dna", "systems", "gap? DNA", false},
      {"systems.evolution", "systems", "gap? evolution", false},
      {"systems.tanhi", "systems", "gap? TANHI", false},
      {"systems.telemetry", "systems", "gap? telemetry", false},
      {"model.tokenizer", "model", "gap? tokenizer", false},
      {"model.entity", "model", "gap? ENTITY", false},
      {"model.transformer", "model", "gap? transformer", false},
      {"model.sampling", "model", "gap? sampling", false},
      {"model.hf", "model", "gap? HF load", false},
      {"lucy.lpd", "lucy", "gap? Lucy/LPD", false},

      // --- Delivery / validation ---
      {"w2a.go_v_vulkan", "interop", "gap? w2a Go_v_Vulkan Dense forward parity + speed", false},
      {"nac.go_delegate", "interop", "gap? Go Welvet NAC → Vulkan dispatch", false},
      {"determinism.policy", "vulkan", "gap? documented reduction order + exact_bit bands", false},
  };
  return k;
}

} // namespace

const std::vector<Entry>& all() { return build(); }

std::size_t count_gaps() {
  std::size_t n = 0;
  for (const auto& e : all()) {
    if (!e.implemented) ++n;
  }
  return n;
}

std::size_t count_done() {
  std::size_t n = 0;
  for (const auto& e : all()) {
    if (e.implemented) ++n;
  }
  return n;
}

void print_census(std::ostream& os) {
  os << "welvet.vulkan gaps: done=" << count_done() << " open=" << count_gaps() << "\n";
  for (const auto& e : all()) {
    os << (e.implemented ? "  [done] " : "  [gap?] ") << e.id << " — " << e.detail << "\n";
  }
}

std::string mark(std::string_view what) {
  return std::string("gap? ") + std::string(what);
}

} // namespace welvet::gaps
