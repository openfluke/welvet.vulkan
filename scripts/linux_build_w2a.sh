#!/usr/bin/env bash
# Build welvet.vulkan then run sibling w2a Go↔Vulkan suites.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W2A="${W2A_DIR:-$ROOT/../w2a}"
BUILD="${BUILD_DIR:-$ROOT/build}"

usage() {
  cat <<EOF
usage: $0 [flags]

  (default)           build + TestGoVVulkanDenseForward
  --smoke             build + example_vulkan_smoke
  --gaps              build + welvet_dense_fwd --gaps
  --dense-fwd         Dense FormatNone×34 forward
  --rmsnorm-fwd       RMSNorm FormatNone×34 forward
  --layernorm-fwd     LayerNorm FormatNone×34 forward
  --softmax-fwd       Softmax forward
  --swiglu-fwd        SwiGLU forward
  --embedding-fwd     Embedding forward
  --mha-fwd           MHA forward
  --cnn-fwd           CNN1/2/3 forward
  --convt-fwd         ConvT1/2/3 forward
  --rnn-fwd           RNN forward
  --lstm-fwd          LSTM forward
  --sequential-fwd    Sequential forward
  --residual-fwd      Residual forward
  --mamba-fwd         Mamba forward
  --metacognition-fwd Metacognition forward
  --gdn-fwd           GDN forward
  --kmeans-fwd        KMeans forward
  --parallel-fwd      Parallel forward
  --seqmix-fwd        seqmix smoke
  --leaf-fwd-all      all leaf FormatNone forward suites
  --dense-train       Dense NormalBP leaf + volume (7 cases)
  --rmsnorm-train     RMSNorm NormalBP leaf + volume (7 cases)
  --layernorm-train   LayerNorm NormalBP leaf + volume (7 cases)
  --softmax-train     Softmax NormalBP leaf + volume (9 cases)
  --swiglu-train      SwiGLU NormalBP leaf + volume (7 cases)
  --embedding-train   Embedding NormalBP leaf + volume (7 cases)
  --mha-train         MHA NormalBP leaf + volume (8 cases)
  --cnn-train         CNN1/2/3 NormalBP leaf + volume
  --convt-train       ConvT1/2/3 NormalBP leaf + volume
  --rnn-train         RNN NormalBP leaf + volume
  --lstm-train        LSTM NormalBP leaf + volume
  --sequential-train  Sequential NormalBP leaf + volume
  --residual-train    Residual NormalBP leaf + volume
  --stack-train       Stack NestedStack-of-Dense NormalBP leaf + volume
  --mamba-train       Mamba NormalBP leaf + volume
  --metacognition-train Metacognition NormalBP leaf + volume
  --gdn-train         GDN NormalBP leaf + volume (f32)
  --kmeans-train      KMeans NormalBP leaf + volume
  --parallel-train    Parallel + ParallelPoly + NormalBPEdges
  --normalbp-all      Full TestGoVVulkanNormalBP aggregator
  --stepbp-train      StepBP Line STB1 + Dense MeshBP MBP1 (+ ModeMeshBP vols)
  --meshbp-train      Heterogeneous MeshBP MGB1 (+ StepBP/volume + NormalBP + Vulkan-only)
  --tween-train       ModeTween TWB1/TWV1 + per-kind volumes (TestGoVVulkanTweenTrain)
  --tweenchain-train  ModeTweenChain TCB1/TCV1 + per-kind volumes (TestGoVVulkanTweenChainTrain)
  --tweensplit-train  ModeTweenSplit TSB1/TSP1 + per-kind volumes (TestGoVVulkanTweenSplitTrain)
  --steptweensplit-train  ModeStepTweenSplit STS1/STSO + TweenSplit volumes (TestGoVVulkanStepTweenSplitTrain)
  --tweenalt-train    ModeTweenAlt TAB1/TAV1 + per-kind volumes (TestGoVVulkanTweenAltTrain)
  --steptweenalt-train  ModeStepTweenAlt STA1/STAO + TweenAlt volumes (TestGoVVulkanStepTweenAltTrain)
  --meshtween-train   MeshTween MTW1/MGW1 + ModeTween volumes (TestGoVVulkanMeshTweenTrain)
  --meshtweenchain-train  MeshTweenChain MTC1/MGC1 + TweenChain volumes (TestGoVVulkanMeshTweenChainTrain)
  --steptween-train  ModeStepTween STW1/STWO + Tween volumes (TestGoVVulkanStepTweenTrain)
  --steptweenchain-train  ModeStepTweenChain STC1/STCO + TweenChain volumes (TestGoVVulkanStepTweenChainTrain)
  --proxy-train       TweenSplit proxy modes GPB1/PGB1 (TestGoVVulkanProxyTrain + ProxyVolumeTrain)
  --mesh-split-modes  MeshTweenSplit/Alt/FastProxy/Sparse clock + shared volumes (TestGoVVulkanMeshSplitModes)
  --training-25-29-all  StepTweenSplit Linear…Sparse Line proxies (TestGoVVulkanStepProxyTrain)
  --cameral-all         Freeze/Shadow/Adversarial/Memory + CamSync (TestGoVVulkanCameralTrain|CamSync|…)
  --training-all        Aggregated NormalBP + Step/Mesh/Tween/proxy training matrix
  --go-routing          Mixed Go/C++/CUDA JSON routing (requires install + C++ artifact)
  --help
EOF
}

MODE=dense-fwd
if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --smoke) MODE=smoke; shift ;;
    --gaps) MODE=gaps; shift ;;
    --cameral-all) MODE=cameral-all; shift ;;
    --training-all) MODE=training-all; shift ;;
    --go-routing) MODE=go-routing; shift ;;
    --dense-fwd|--rmsnorm-fwd|--layernorm-fwd|--softmax-fwd|--swiglu-fwd|--embedding-fwd|--mha-fwd|--cnn-fwd|--convt-fwd|--rnn-fwd|--lstm-fwd|--sequential-fwd|--residual-fwd|--mamba-fwd|--metacognition-fwd|--gdn-fwd|--kmeans-fwd|--parallel-fwd|--seqmix-fwd|--leaf-fwd-all|--dense-train|--rmsnorm-train|--layernorm-train|--softmax-train|--swiglu-train|--embedding-train|--mha-train|--cnn-train|--convt-train|--rnn-train|--lstm-train|--sequential-train|--residual-train|--stack-train|--mamba-train|--metacognition-train|--gdn-train|--kmeans-train|--parallel-train|--normalbp-all|--stepbp-train|--meshbp-train|--tween-train|--tweenchain-train|--tweensplit-train|--steptweensplit-train|--tweenalt-train|--steptweenalt-train|--meshtween-train|--meshtweenchain-train|--steptween-train|--steptweenchain-train|--proxy-train|--mesh-split-modes|--training-25-29-all)
      MODE="${1#--}"; shift ;;
    *)
      echo "unknown flag '$1'" >&2
      usage
      exit 2
      ;;
  esac
fi

"$ROOT/scripts/build_linux.sh"

export WELVET_VULKAN_DENSE_FWD="$BUILD/welvet_dense_fwd"
if [[ -z "${WELVET_VULKAN_LIBRARY:-}" ]]; then
  if [[ -f "$ROOT/../welvet/native/vulkan/linux_x86_64/libwelvet_vulkan.so" ]]; then
    export WELVET_VULKAN_LIBRARY="$ROOT/../welvet/native/vulkan/linux_x86_64/libwelvet_vulkan.so"
  elif [[ -f "$BUILD/libwelvet_vulkan.so" ]]; then
    export WELVET_VULKAN_LIBRARY="$BUILD/libwelvet_vulkan.so"
  fi
fi
if [[ -z "${WELVET_DENSE_FWD:-}" ]]; then
  for cand in "$ROOT/../welvet.cpp/build/welvet_dense_fwd" "$ROOT/../welvet.cpp/build/Release/welvet_dense_fwd"; do
    if [[ -x "$cand" ]]; then
      export WELVET_DENSE_FWD="$cand"
      break
    fi
  done
fi

run_go_test() {
  local pattern="$1"
  if [[ ! -d "$W2A" ]]; then
    echo "w2a sibling not found at $W2A" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="${ROOT}/.deps/cuda/cuda/lib64:${BUILD}:${LD_LIBRARY_PATH:-}"
  # Default 1 measured bench iter (match metal); set WELVET_VULKAN_BENCH_ITERS=2000 for long perf.
  export WELVET_VULKAN_BENCH_ITERS="${WELVET_VULKAN_BENCH_ITERS:-1}"
  (
    cd "$W2A"
    env -u CC -u CXX -u CMAKE_CXX_COMPILER -u CMAKE_C_COMPILER \
      PATH="/usr/bin:/usr/local/go/bin:${PATH}" \
      WELVET_VULKAN_DENSE_FWD="$WELVET_VULKAN_DENSE_FWD" \
      WELVET_VULKAN_LIBRARY="${WELVET_VULKAN_LIBRARY:-}" \
      WELVET_DENSE_FWD="${WELVET_DENSE_FWD:-}" \
      WELVET_VULKAN_BENCH_ITERS="$WELVET_VULKAN_BENCH_ITERS" \
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
      go test ./tests/go_v_vulkan -v -count=1 -timeout "${WELVET_VULKAN_TEST_TIMEOUT:-90m}" -run "$pattern"
  )
}

case "$MODE" in
  smoke) "$BUILD/example_vulkan_smoke" ;;
  gaps) "$BUILD/welvet_dense_fwd" --gaps ;;
  dense-fwd) run_go_test '^TestGoVVulkanDenseForward$' ;;
  rmsnorm-fwd) run_go_test '^TestGoVVulkanRMSNorm$' ;;
  layernorm-fwd) run_go_test '^TestGoVVulkanLayerNorm$' ;;
  softmax-fwd) run_go_test '^TestGoVVulkanSoftmax$' ;;
  swiglu-fwd) run_go_test '^TestGoVVulkanSwiGLU$' ;;
  embedding-fwd) run_go_test '^TestGoVVulkanEmbedding$' ;;
  mha-fwd) run_go_test '^TestGoVVulkanMHA$' ;;
  cnn-fwd) run_go_test '^TestGoVVulkanCNN$' ;;
  convt-fwd) run_go_test '^TestGoVVulkanConvT$' ;;
  rnn-fwd) run_go_test '^TestGoVVulkanRNN$' ;;
  lstm-fwd) run_go_test '^TestGoVVulkanLSTM$' ;;
  sequential-fwd) run_go_test '^TestGoVVulkanSequential$' ;;
  residual-fwd) run_go_test '^TestGoVVulkanResidual$' ;;
  mamba-fwd) run_go_test '^TestGoVVulkanMamba$' ;;
  metacognition-fwd) run_go_test '^TestGoVVulkanMetacognition$' ;;
  gdn-fwd) run_go_test '^TestGoVVulkanGDN$' ;;
  kmeans-fwd) run_go_test '^TestGoVVulkanKMeans$' ;;
  parallel-fwd) run_go_test '^TestGoVVulkanParallel$' ;;
  seqmix-fwd) run_go_test '^TestGoVVulkanSeqmix$' ;;
  leaf-fwd-all) run_go_test '^TestGoVVulkan(DenseForward|RMSNorm|LayerNorm|Softmax|SwiGLU|Embedding|MHA|CNN|ConvT|RNN|LSTM|Sequential|Residual|Mamba|Metacognition|GDN|KMeans|Parallel|Seqmix)$' ;;
  dense-train) run_go_test '^TestGoVVulkanDenseTrain$' ;;
  rmsnorm-train) run_go_test '^TestGoVVulkanRMSNormTrain$' ;;
  layernorm-train) run_go_test '^TestGoVVulkanLayerNormTrain$' ;;
  softmax-train) run_go_test '^TestGoVVulkanSoftmaxTrain$' ;;
  swiglu-train) run_go_test '^TestGoVVulkanSwiGLUTrain$' ;;
  embedding-train) run_go_test '^TestGoVVulkanEmbeddingTrain$' ;;
  mha-train) run_go_test '^TestGoVVulkanMHATrain$' ;;
  cnn-train) run_go_test '^TestGoVVulkanCNNTrain$' ;;
  convt-train) run_go_test '^TestGoVVulkanConvTTrain$' ;;
  rnn-train) run_go_test '^TestGoVVulkanRNNTrain$' ;;
  lstm-train) run_go_test '^TestGoVVulkanLSTMTrain$' ;;
  sequential-train) run_go_test '^TestGoVVulkanSequentialTrain$' ;;
  residual-train) run_go_test '^TestGoVVulkanResidualTrain$' ;;
  stack-train) run_go_test '^TestGoVVulkanStackTrain$' ;;
  mamba-train) run_go_test '^TestGoVVulkanMambaTrain$' ;;
  metacognition-train) run_go_test '^TestGoVVulkanMetacognitionTrain$' ;;
  gdn-train) run_go_test '^TestGoVVulkanGDNTrain$' ;;
  kmeans-train) run_go_test '^TestGoVVulkanKMeansTrain$' ;;
  parallel-train) run_go_test '^TestGoVVulkan(ParallelTrain|ParallelPolyTrain|NormalBPEdges)$' ;;
  normalbp-all) run_go_test '^TestGoVVulkanNormalBP$' ;;
  stepbp-train) run_go_test '^TestGoVVulkanStepBPTrain$' ;;
  meshbp-train) run_go_test '^(TestGoVVulkanMeshBPTrain|TestGoVVulkanStepBPTrain|TestGoVVulkanNormalBP)$' ;;
  tween-train) run_go_test '^TestGoVVulkanTweenTrain$' ;;
  tweenchain-train) run_go_test '^TestGoVVulkanTweenChainTrain$' ;;
  tweensplit-train) run_go_test '^TestGoVVulkanTweenSplitTrain$' ;;
  steptweensplit-train) run_go_test '^TestGoVVulkanStepTweenSplitTrain$' ;;
  tweenalt-train) run_go_test '^TestGoVVulkanTweenAltTrain$' ;;
  steptweenalt-train) run_go_test '^TestGoVVulkanStepTweenAltTrain$' ;;
  meshtween-train) run_go_test '^(TestGoVVulkanMeshTweenTrain|TestGoVVulkanMeshBPTrain|TestGoVVulkanTweenTrain)$' ;;
  meshtweenchain-train) run_go_test '^(TestGoVVulkanMeshTweenChainTrain|TestGoVVulkanMeshTweenTrain|TestGoVVulkanTweenChainTrain)$' ;;
  steptween-train) run_go_test '^(TestGoVVulkanStepTweenTrain|TestGoVVulkanTweenTrain)$' ;;
  steptweenchain-train) run_go_test '^(TestGoVVulkanStepTweenChainTrain|TestGoVVulkanTweenChainTrain)$' ;;
  proxy-train) run_go_test '^(TestGoVVulkanProxyTrain|TestGoVVulkanProxyVolumeTrain)$' ;;
  mesh-split-modes) run_go_test '^TestGoVVulkanMeshSplitModes$' ;;
  training-25-29-all) run_go_test '^TestGoVVulkanStepProxyTrain$' ;;
  training-all) run_go_test '^TestGoVVulkan(NormalBP|StepBPTrain|MeshBPTrain|TweenTrain|TweenChainTrain|TweenSplitTrain|StepTweenSplitTrain|TweenAltTrain|StepTweenAltTrain|MeshTweenTrain|MeshTweenChainTrain|StepTweenTrain|StepTweenChainTrain|ProxyTrain|ProxyVolumeTrain|MeshSplitModes|StepProxyTrain)$' ;;
  cameral-all) WELVET_VULKAN_TEST_TIMEOUT="${WELVET_VULKAN_TEST_TIMEOUT:-180m}" run_go_test '^(TestGoVVulkanCameralTrain|TestGoVVulkanCamSync|TestGoStoreReplaceInvalidatesWires|TestGoVVulkanOnly)$' ;;
  go-routing)
    "$ROOT/scripts/linux_install_go.sh"
    export WELVET_REQUIRE_CPP_ROUTING=1
    export WELVET_VULKAN_BENCH_ITERS="${WELVET_VULKAN_BENCH_ITERS:-1}"
    export WELVET_VULKAN_LIBRARY="${WELVET_VULKAN_LIBRARY:-$ROOT/../welvet/native/vulkan/linux_x86_64/libwelvet_vulkan.so}"
    run_go_test '^(TestGoNativeVulkan.*|TestGoVVulkanOnly)$'
    ;;
esac
