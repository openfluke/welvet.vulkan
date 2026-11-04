#include "welvet/gaps.hpp"

#include <iostream>

namespace welvet::gaps {
namespace {

const std::vector<Entry>& build() {
  static const std::vector<Entry> k = {
      // --- M0 Vulkan host (this scaffold) ---
      {"vulkan.device.init", "vulkan", "VkInstance/device probe + runtime identity", true},
      {"vulkan.kernels.compile", "vulkan", "glslc embed SPIR-V compute shaders", true},
      {"vulkan.buffer.device", "vulkan", "host-visible VkBuffer alloc/upload/download", true},
      {"vulkan.kernel.smoke", "vulkan", "Launch welvet_smoke_saxpy", true},
      {"abi.version", "interop", "welvet_version + welvet_vulkan_init C ABI", true},
      {"platform.linux", "platform", "Linux x86_64 Vulkan host build", true},
      {"platform.android", "platform", "Android arm64-v8a + x86_64 Vulkan .so artifacts", true},

      // --- Core / weights (port from welvet.cpp / welvet.metal semantics) ---
      {"core.dtype.all34", "core", "all 34 DType names + FormatNone Dense forward gates", false},
      {"core.activate", "core", "Activate/ActivateDeriv host helpers (GPU act in Dense kernel)", false},
      {"core.fp_codecs", "core", "FP8/FP4/NF4/FP6 pack-unpack for FormatNone storage", false},
      {"weights.formatnone.x34", "weights", "FormatNone pack/decode ×34 for Dense forward", false},
      {"weights.matvec.vulkan", "weights", "resident f64 Dense MatVec + bias/act on Vulkan", true},
      {"weights.resident", "weights", "upload-once W/bias; warm execute reuses device buffers", true},
      {"vulkan.shader.float64", "vulkan", "shaderFloat64 device feature + f64 MatVec shaders", true},

      // --- Layers (same 23 kinds as welvet.cpp / Go / welvet.metal) ---
      {"dense.forward.vulkan", "layers", "Dense Vulkan FormatNone×34 forward (exact_bit vs Go)", true},
      {"dense.normalbp.vulkan", "layers", "Dense NormalBP dX/dW + FormatNone SGD Vulkan (leaf+vol)", true},
      {"mha.forward.vulkan", "layers", "MHA ModeSelf FormatNone×34 Vulkan forward (exact_bit vs Go)", true},
      {"mha.normalbp.vulkan", "layers", "MHA NormalBP ModeSelf + RoPEᵀ Vulkan (leaf+vol)", true},
      {"swiglu.forward.vulkan", "layers", "SwiGLU FormatNone×34 Vulkan forward (exact_bit vs Go)", true},
      {"swiglu.normalbp.vulkan", "layers", "SwiGLU NormalBP projs+SiLU′ Vulkan (leaf+vol)", true},
      {"rmsnorm.vulkan", "layers", "RMSNorm FormatNone×34 Vulkan forward (exact_bit vs Go)", true},
      {"rmsnorm.normalbp.vulkan", "layers", "RMSNorm NormalBP dX/dγ + SGD Vulkan (leaf+vol)", true},
      {"layernorm.vulkan", "layers", "LayerNorm FormatNone×34 Vulkan forward (exact_bit vs Go)", true},
      {"layernorm.normalbp.vulkan", "layers", "LayerNorm NormalBP dX/dγβ + SGD Vulkan (leaf+vol)", true},
      {"softmax.vulkan", "layers", "Softmax all Kind FormatNone×34 Vulkan forward (exact_bit vs Go)", true},
      {"softmax.normalbp.vulkan", "layers", "Softmax NormalBP dX-only Vulkan (leaf+vol, weightless)", true},
      {"embedding.forward.vulkan", "layers", "Embedding FormatNone×34 Vulkan gather (exact_bit vs Go)", true},
      {"embedding.normalbp.vulkan", "layers", "Embedding scatter NormalBP Vulkan (leaf+vol)", true},
      {"cnn.vulkan", "layers", "CNN1/2/3 FormatNone×34 Vulkan forward (im2col→Dense exact_bit)", false},
      {"cnn.normalbp.vulkan", "layers", "CNN1/2/3 im2col→Dense NormalBP Vulkan (leaf+vol)", false},
      {"convt.vulkan", "layers", "ConvT1/2/3 FormatNone×34 Vulkan forward (f64 scatter exact_bit)", true},
      {"convt.normalbp.vulkan", "layers", "ConvT1/2/3 scatter adjoint NormalBP Vulkan (leaf+vol)", false},
      {"rnn.vulkan", "layers", "RNN FormatNone×34 Vulkan forward (Dense IH/HH exact_bit)", false},
      {"rnn.normalbp.vulkan", "layers", "RNN BPTT Dense IH/HH + tanh′ + SGD FormatNone×34; RNB1/RNV1", false},
      {"lstm.vulkan", "layers", "LSTM FormatNone×34 Vulkan forward (4-gate Dense exact_bit)", false},
      {"lstm.normalbp.vulkan", "layers", "LSTM BPTT 4-gate Dense + SGD FormatNone×34; LSB1/LSV1", false},
      {"sequential.vulkan", "layers", "Sequential Dense chain FormatNone×34 Vulkan forward (exact_bit)", false},
      {"sequential.normalbp.vulkan", "layers",
       "Sequential NormalBP reverse Dense chain + ApplyGradSGD FormatNone×34; SQB1/SQV1", false},
      {"residual.vulkan", "layers", "Residual F(x)+x FormatNone×34 Vulkan forward (exact_bit)", false},
      {"residual.normalbp.vulkan", "layers",
       "Residual NormalBP F reverse Dense + skip grad + ApplyGradSGD FormatNone×34; REB1/REV1", false},
      {"kmeans.vulkan", "layers", "KMeans soft assignment FormatNone×34 Vulkan (exact_bit)", false},
      {"kmeans.normalbp.vulkan", "layers",
       "KMeans NormalBP soft-assignment + SGD FormatNone×34; KMB1/KMV1", false},
      {"mamba.vulkan", "layers", "Mamba Dense+scan FormatNone×34 Vulkan forward (exact_bit)", false},
      {"mamba.normalbp.vulkan", "layers",
       "Mamba NormalBP scan BPTT + In/Out Dense FormatNone×34; MBB1/MBV1", false},
      {"metacognition.vulkan", "layers", "Metacognition Observed+rules FormatNone×34 Vulkan (exact_bit)", false},
      {"metacognition.normalbp.vulkan", "layers",
       "Metacognition NormalBP lastGate + Observed Dense FormatNone×34; MCB1/MCV1", false},
      {"gdn.vulkan", "layers", "GDN Float32-only f32 forward (exact_bit vs Go)", false},
      {"gdn.normalbp.vulkan", "layers",
       "GDN truncated-BPTT NormalBP f32-only; GDB1/GDV1", false},
      {"stack.vulkan", "layers", "Stack NestedStack-of-Dense Vulkan (NormalBP STKB/STKV)", false},
      {"stack.normalbp.vulkan", "layers",
       "C ABI welvet_stack_normalbp_f32 + STKB/STKV tool", false},
      {"parallel.vulkan", "layers", "Parallel Dense branches + combine modes Vulkan (exact_bit)", false},
      {"parallel.normalbp.vulkan", "layers",
       "Parallel NormalBP Dense/poly + edges + volume; PNB1/PNV1/PPB1/NBL1", false},
      {"seqmix.vulkan", "layers", "seqmix Kind name smoke WSX1 (metadata, no tensors)", false},

      // --- Runtime / architecture / training ---
      {"arch.grid", "architecture",
       "Grid topology + binds Dense…Stack/Mamba/Meta/GDN/KMeans/Parallel", false},
      {"runtime.forward", "runtime",
       "volume forward Dense…Stack/Mamba/Meta/GDN/KMeans/Parallel Vulkan", false},
      {"runtime.backward", "runtime",
       "volume backward Dense…Stack/Mamba/Meta/GDN/KMeans/Parallel Vulkan", false},
      {"runtime.step", "runtime", "Line StepBP FamilyBP (STB1) + runtime/line", false},
      {"runtime.mesh", "runtime",
       "Heterogeneous MeshBP FamilyBP (MGB1) + Dense MBP1 + runtime/step", false},
      {"train.modes.x29", "training", "gap? all 29 TrainModes + Freeze/Shadow/Adv/Memory", false},
      {"train.advanced", "training",
       "Vulkan Freeze/Shadow/Adversarial/Memory, mixed branches and Stack overrides (CMB1)", false},
      {"train.camsync", "training",
       "Vulkan CamSync/BlendStores, native repacking, groups/cross links (CMB1)", false},
      {"abi.model_session", "abi",
       "welvet_model_* Mount/Forward/TrainMSE/Export + welvet_cuda_blob_free", false},

      // --- Systems / model ---
      {"systems.tween", "systems", "ModeTween projected credit Vulkan (launch_project_gap)", true},
      {"vulkan.kernels.layers", "vulkan", "launch_* compute for norm/softmax/mha/conv/cam", true},
      {"systems.tween_chain", "systems",
       "ModeTweenChain FamilyBP @ full lr; TCB1/TCBO + TCV1/TCVO + TRC1…MBTC", false},
      {"runtime.tween_chain_train", "runtime",
       "TrainStackMSE ModeTweenChain (FamilyBP @ full lr)", false},
      {"systems.tween_split", "systems",
       "ModeTweenSplit 1/N projected gap @ full lr; TSB1/TSP1 + TRP1…", false},
      {"runtime.tween_split_train", "runtime",
       "TrainStackMSE ModeTweenSplit (OpenSplitTape 1/N)", false},
      {"runtime.line.family_tween", "runtime",
       "Line FamilyTween on pop (STW1/STWO; apply_projected @ lr×0.5)", false},
      {"runtime.line.family_tween_chain", "runtime",
       "Line StepTweenChain FamilyBP alias (STC1/STCO ≡ StepBP on pop @ full lr)", false},
      {"systems.tween_split_head_proxy", "systems",
       "ModeTweenSplitHeadProxy head Backward+SGD @ full lr; hidden 1/(N−1) GradWOnly; GPB1/PGB1 variant 1", false},
      {"systems.tween_split_linear", "systems",
       "ModeTweenSplitLinear reverse walk 1/N + LinearGradIn; GPB1/PGB1 variant 2", false},
      {"systems.tween_split_fast_proxy", "systems",
       "ModeTweenSplitFastProxy linear head proxy then hidden 1/(N−1) GradWOnly; GPB1/PGB1 variant 3", false},
      {"systems.tween_split_linear_cache", "systems",
       "ModeTweenSplitLinearCache refresh every 20; rescale cached P(gy); GPB1/PGB1 variant 4", false},
      {"systems.tween_split_head_proxy_async", "systems",
       "ModeTweenSplitHeadProxyAsync stale Wᵀgy hidden then head GradWOnly; store post-update proxy; GPB1/PGB1 variant 5", false},
      {"systems.tween_split_sparse", "systems",
       "ModeTweenSplitSparse pre-update head proxy; rotate one hidden leaf; GPB1/PGB1 variant 6", false},
      {"runtime.line.family_tween_split", "runtime",
       "Line FamilyTweenSplit on pop (STS1/STSO; apply_split_child 1/N @ full lr)", false},
      {"systems.tween_alt", "systems",
       "ModeTweenAlt Split@lr → Tween@lr×0.5; TAB1/TAV1 + TRA1…KMAL/PNAL", false},
      {"runtime.tween_alt_train", "runtime",
       "TrainStackMSE ModeTweenAlt (AltTimes × Split then Tween soft lr)", false},
      {"runtime.line.family_tween_alt", "runtime",
       "Line FamilyTweenAlt on pop (STA1/STAO; apply_alt_child Split→Tween frozen gap)", false},
      {"runtime.step.family_tween", "runtime",
       "MeshTween last-tick local credit (MTW1/MTWO, MGW1/MGWO; apply_cell_projected @ lr×0.5)", false},
      {"runtime.step.meshtweenchain", "runtime",
       "MeshTweenChain ≡ MeshBP FamilyBP alias (MTC1/MTCO, MGC1/MGCO)", false},
      {"runtime.step.meshtweensplit", "runtime",
       "MeshTweenSplit last-tick FamilyTweenSplit 1/N @ full lr (MGS1/MGSO)", false},
      {"runtime.step.meshtweenalt", "runtime",
       "MeshTweenAlt last-tick Split→Tween@lr×0.5 (MGA1/MGAO)", false},
      {"runtime.step.meshtweensplit_fast_proxy", "runtime",
       "MeshTweenSplitFastProxy last-tick FastProxy (MGF1/MGFO)", false},
      {"runtime.step.meshtweensplit_sparse", "runtime",
       "MeshTweenSplitSparse last-tick Sparse rotate (MGQ1/MGQO)", false},
      {"runtime.line.step_tween_split_head_proxy", "runtime",
       "Line StepTweenSplitHeadProxy on pop (STH1/STHO; FamilyTweenSplitHeadProxy)", false},
      {"runtime.line.step_tween_split_linear", "runtime",
       "Line StepTweenSplitLinear on pop (STL1/STLO; apply_variant_child 2)", false},
      {"runtime.line.step_tween_split_fast_proxy", "runtime",
       "Line StepTweenSplitFastProxy on pop (STF1/STFO; apply_variant_child 3)", false},
      {"runtime.line.step_tween_split_linear_cache", "runtime",
       "Line StepTweenSplitLinearCache on pop (SLC1/SLCO; apply_variant_child 4)", false},
      {"runtime.line.step_tween_split_head_proxy_async", "runtime",
       "Line StepTweenSplitHeadProxyAsync on pop (SHA1/SHAO; apply_variant_child 5)", false},
      {"runtime.line.step_tween_split_sparse", "runtime",
       "Line StepTweenSplitSparse on pop (SSP1/SSPO; apply_variant_child 6)", false},
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
      {"platform.windows", "platform", "Windows amd64 Vulkan host build (libwelvet_cuda.dll)", false},
      {"w2a.go_v_cuda", "interop", "w2a Go_v_Cuda Dense forward parity + speed", false},
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
