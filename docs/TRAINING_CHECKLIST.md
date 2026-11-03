# Training checklist — welvet.vulkan

One **TrainMode** at a time. Finish that mode’s table (or the rows you care about), bump version, move on.

Canonical enum: Go `layers/parallel/train_mode.go` (`TrainMode`).  
`inherit` (0) = resolve-only — not a ship target.

**Port mirror of** [`welvet.cuda/docs/TRAINING_CHECKLIST.md`](../../welvet.cuda/docs/TRAINING_CHECKLIST.md)
(and metal/cpp). Same cases / wires / column meanings; status here is **Vulkan** only.
cuda/metal/cpp checkmarks do **not** transfer credit.

**Current engine:** `0.79` string (ported); leaf FormatNone forwards + NormalBP signed via w2a.
Climb target: welvet.cuda **0.79** training modes — Tween* / Mesh* / Step* still open after NormalBP.

---

## Vulkan-only mandate (non-negotiable)

**welvet.vulkan = GPU Vulkan compute only.** Host/CPU math belongs in **welvet.cpp** (and Go).  
Host numerical training loops are forbidden. Host scheduling, validation, allocation and storage staging are permitted; residency and performance work remain separate gates.

There is **no** silent host/CPU numerical fallback labeled as Vulkan. Unsupported dtype/layer/mode → hard error.

Current executable contract: [VULKAN_ONLY.md](VULKAN_ONLY.md).

Validation will live in sibling **w2a** (`go_v_vulkan`), not in this tree — same pattern as `go_v_cuda` / `go_v_metal` / `go_v_cpp`.

```sh
./scripts/build_linux.sh
./build/example_vulkan_smoke
./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
# as suites land:
# ./scripts/linux_build_w2a.sh --smoke
# ./scripts/linux_build_w2a.sh --leaf-fwd-all
# ./scripts/linux_build_w2a.sh --normalbp-all
# ./scripts/linux_build_w2a.sh --training-all
```

## w2a Go↔Vulkan (leaf FormatNone forward) — signed

Run: `./scripts/linux_build_w2a.sh --leaf-fwd-all` (or per-layer `--*-fwd`).  
Exact tensor bits vs Go (+ cpp speed column when present). Stack has no leaf forward magic.

| Layer | Test | Wire | Status |
|---|---|---|:---:|
| Dense | `TestGoVVulkanDenseForward` | `WVD3`→`WVO1` | [x] |
| RMSNorm | `TestGoVVulkanRMSNorm` | `WVR1`→`WVRO` | [x] |
| LayerNorm | `TestGoVVulkanLayerNorm` | `WVL1`→`WVLO` | [x] |
| Softmax | `TestGoVVulkanSoftmax` | `WSM1`→`WSMO` | [x] |
| SwiGLU | `TestGoVVulkanSwiGLU` | `WVS1`→`WVSO` | [x] |
| Embedding | `TestGoVVulkanEmbedding` | `WEM1`→`WEMO` | [x] |
| MHA | `TestGoVVulkanMHA` | `WVM1`→`WVMO` | [x] |
| CNN1/2/3 | `TestGoVVulkanCNN` | `WVC*`→`WVCO` | [x] |
| ConvT1/2/3 | `TestGoVVulkanConvT` | `WVT*`→`WVTO` | [x] |
| RNN | `TestGoVVulkanRNN` | `WRN1`→`WRNO` | [x] |
| LSTM | `TestGoVVulkanLSTM` | `WLS1`→`WLSO` | [x] |
| Sequential | `TestGoVVulkanSequential` | `WSQ1`→`WSQO` | [x] |
| Residual | `TestGoVVulkanResidual` | `WRE1`→`WREO` | [x] |
| Mamba | `TestGoVVulkanMamba` | `WMB1`→`WMBO` | [x] |
| Metacognition | `TestGoVVulkanMetacognition` | `WMC1`→`WMCO` | [x] |
| GDN | `TestGoVVulkanGDN` | `WGD1`→`WGDO` | [x] f32 |
| KMeans | `TestGoVVulkanKMeans` | `WKM1`→`WKMO` | [x] |
| Parallel | `TestGoVVulkanParallel` | `WPL1`→`WPLO` | [x] |
| seqmix | `TestGoVVulkanSeqmix` | `WSX1`→`WSXO` | [x] |
| Stack | `TestGoVVulkanStackTrain` | `STKB`→`STKO` / `STKV`→`STVO` | [x] NormalBP |

## w2a Go↔Vulkan (Freeze / Shadow / Adversarial / Memory / Cameral / CamSync) — sign-off evidence

Not signed yet. Same scope as welvet.metal when the Vulkan path lands: Freeze/Shadow, Adversarial/Memory, Cameral BranchModes, CamSync/BlendStores — leaf + placed volume + native dtype sweep — via w2a. No welvet.cpp or welvet.metal credit transfers.

| Scope | Inside node | Placed volume | Native dtype sweep | Evidence |
|---|:---:|:---:|:---:|---|
| Freeze / Shadow | ⬜ | ⬜ | ⬜ | |
| Adversarial / Memory | ⬜ | ⬜ | ⬜ | |
| Shadow teacher credit | ⬜ | ⬜ | ⬜ | |
| Cameral BranchModes | ⬜ | ⬜ | ⬜ | |
| CamSync / BlendStores | ⬜ | ⬜ | ⬜ | |
| Native replacement / coherence | ⬜ | ⬜ | ⬜ | |

### Numerical execution inventory

| Path | Execution today |
|---|---|
| NormalBP layers | ✅ Dense…Stack + Mamba + Metacognition + GDN(f32) + KMeans + Parallel leaf+vol signed |
| MSE / MSEGrad | ✅ host f64 match Go (Dense NormalBP) |
| SGD and native storage updates | ✅ FormatNone Dense SGD |
| GDN | ✅ f32 truncated-BPTT NormalBP leaf+vol |
| Tween / TweenChain / Split / Alt / proxy / mesh variants | ⬜ ModeTween…MeshTweenSplitSparse + StepTweenSplit HeadProxy/Linear/FastProxy/LinearCache/Async/Sparse signed; Freeze/Shadow / CamSync open |
| Scheduling and data movement | Host validation, scheduling, allocation, copies and storage staging (permitted) |
| GPU-only enforcement | M0 host scaffold only; numerical Vulkan-only gate lands with first training suites |

**Project parity target:** best-effort deterministic agreement with Go, aiming
for 100% and accepting 60–100% on w2a's spectrum (`industry` through `exact_bit`).
Long-horizon training need not reproduce an identical CPU trajectory. See
[the README policy](../README.md#numerical-parity-target).

**Fixture precision policy (when suites land):** exact tensor bits where required by each case; finite loss
uses `abs(go-vulkan) <= 1e-12 * max(1, abs(go))` unless a row declares another band. Document
measured bands before signing ⬜. GDN / Softmax / MHA may use the same spectrum floors as welvet.metal once evidenced on Vulkan.

Checkmarks below indicate validated Vulkan correctness under these bands, not a speed
win or completion of GPU residency/performance work. Unsupported and unverified
modes remain unchecked. **Nothing below is signed at M0.**

---

### Columns (every table)

| Column | Means |
|---|---|
| **Inside node** | Layer trains correctly in a `1×1×1` cell (single-node / ordinary NN case). go_v_vulkan / exact_bit (or declared band). |
| **3D volumetric** | Same mode works when that layer sits in a multi-cell Grid (volume Forward/Backward + credit across cells). |
| **Fully signed off** | **All** supported FormatNone dtypes for that layer under this mode (not just f32). Softmax = dX only. GDN = f32-only in Go → sign off when f32 path is done. |

Tick `[ ]` when validated. **Ver** records a release tag when one exists, otherwise a dated unreleased validation. **Fully signed off** requires every supported dtype, not only Float32 smoke coverage.

**All 29 training modes:** run `./scripts/linux_build_w2a.sh --training-all`
to build and run the full training matrix plus Vulkan-only regressions.
Run `./scripts/linux_build_w2a.sh --training-25-29-all` for the final five Line
modes, shared proxy volumes and Vulkan-only guard.

**NormalBP subset:** run `./scripts/linux_build_w2a.sh --normalbp-all` to build
Vulkan and run both `TestGoVVulkanNormalBP` and `TestGoVVulkanOnly` in sibling `w2a`.
Run `./scripts/linux_build_w2a.sh --step-tween-all` for NormalBP plus all five
sections 2–6 and the Vulkan-only guard.
The combined runners default to one measured benchmark iteration per fixture;
set `WELVET_VULKAN_BENCH_ITERS=2000` for longer performance measurements.

---

## w2a Go↔Vulkan (Dense NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --dense-train`  

Wire: `WDB1`/`WDBO` (leaf), `WVN1`/`WVNO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: warm-up + `--bench=N` (in-process; no spawn) → **goµs / vkµs / spd%Δ** (`(vulkan−go)/go×100`, neg ⇒ Vulkan faster). Resident soft-double W (no remount each op); microbench spd%Δ is GPU-launch-bound vs Go host.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 ReLU (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔Vulkan (RMSNorm NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --rmsnorm-train`  
Wire: `WRB1`/`WRBO` (leaf), `WRV1`/`WRVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention as Dense (leaf mount-once + γ reset; no remount each iter).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dγ/γ/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔Vulkan (LayerNorm NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --layernorm-train`  
Wire: `WLB1`/`WLBO` (leaf), `WLV1`/`WLVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + γ/β reset).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dγβ/γβ/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔Vulkan (SwiGLU NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --swiglu-train`  
Wire: `WSB1`/`WSBO` (leaf), `WSV1`/`WSVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔Vulkan (Softmax NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --softmax-train`  
Wire: `WXB1`/`WXBO` (leaf), `WXV1`/`WXVO` (volume). Softmax = dX only (weightless).  
Timing: leaf mount-once (reuse layer; no W). Band: **≥loose** floor (Softmax hard to bit-match); leaf×34 / volume hit **strict (~80%)**; Kind×34 worst **loose**; Gumbel = smoke.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Standard Float32 (dX/loss) | [x] | strict + speed |
| 2 | Leaf all Kind Float32 (+ gumbel smoke) | [x] | ≥loose; gumbel non-det |
| 3 | Leaf Standard × all 34 dtypes | [x] | strict ×34 + speed |
| 4 | Leaf all Kind × all 34 dtypes | [x] | ≥loose (worst=loose) |
| 5 | Volume 1×1×1×1 Float32 | [x] | inside-node strict |
| 6 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer strict |
| 7 | Volume 1×1×1×1 × all 34 dtypes | [x] | strict ×34 |
| 8 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric strict |
| 9 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell strict ×34 |

## w2a Go↔Vulkan (Embedding NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --embedding-train`  
Wire: `WEB1`/`WEBO` (leaf), `WEV1`/`WEVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + W reset). dX zeros (discrete IDs); dW scatter.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + spectrum |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 |

## w2a Go↔Vulkan (MHA NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --mha-train`  
Wire: `WHB1`/`WHBO` (leaf), `WHV1`/`WHVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (resident Dense Q/K/V/O; microbench spd%Δ is launch-bound). ModeSelf baseline (no dropout/QKNorm/cross).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf DecoderCausal Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf GQA + Encoder Float32 | [x] | exact_bit |
| 3 | Leaf DecoderCausal × all 34 dtypes | [x] | W exact_bit; dX/dW ≥strict (uint2 dW ~1 ULP) |
| 4 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 5 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 6 | Volume 1×1×1×1 × all 34 dtypes | [x] | + spectrum |
| 7 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed; subnormal weight differences allowed |
| 8 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 |

## w2a Go↔Vulkan (CNN1/2/3 NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --cnn-train`  
Wire: `C1B1`/`C1BO`, `C2B1`/`C2BO`, `C3B1`/`C3BO` (leaf); `C*V1`/`C*VO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + proj W reset). Vulkan convolution gather forward + input/filter gradient kernels.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Leaf Float32 CNN1/2/3 | [x] | exact_bit + speed |
| 4–6 | Leaf ×34 CNN1/2/3 | [x] | exact_bit spectrum |
| 7–9 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving pad |
| 10–12 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Vulkan (ConvT1/2/3 NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --convt-train`  
Wire: `T1B1`/`T1BO`, `T2B1`/`T2BO`, `T3B1`/`T3BO` (leaf); `T*V1`/`T*VO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + proj W reset). Scatter forward + act Jacobian + scatter adjoint (no bias).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Leaf Float32 ConvT1/2/3 | [x] | exact_bit + speed |
| 4–6 | Leaf ×34 ConvT1/2/3 | [x] | exact_bit spectrum |
| 7–9 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving pad |
| 10–12 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Vulkan (RNN NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --rnn-train`  
Wire: `RNB1`/`RNBO` (leaf); `RNV1`/`RNVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. BPTT via Dense IH/HH + tanh′; loom `[dIH|dHH|dBias]`.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | in=hid chain |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Vulkan (LSTM NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --lstm-train`  
Wire: `LSB1`/`LSBO` (leaf); `LSV1`/`LSVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. BPTT 4-gate Dense; loom `[i|f|g|o]` packs.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | in=hid chain |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Vulkan (Sequential NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --sequential-train`  
Wire: `SQB1`/`SQBO` (leaf); `SQV1`/`SQVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. Reverse Dense Dim→Dim chain; loom Depth×[Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving Dim |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Vulkan (Residual NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --residual-train`  
Wire: `REB1`/`REBO` (leaf); `REV1`/`REVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. F reverse Dense + skip grad; loom Depth×[Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | y=F(x)+x |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Vulkan (Mamba NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --mamba-train`  
Wire: `MBB1`/`MBBO` (leaf); `MBV1`/`MBVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. Scan BPTT + In/Out Dense; loom In|Out|ALog|DSkip.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving D |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Vulkan (Metacognition NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --metacognition-train`  
Wire: `MCB1`/`MCBO` (leaf); `MCV1`/`MCVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. lastGate-scale + Observed Dense; loom [Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | default cool rules |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Vulkan (GDN NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --gdn-train`  
Wire: `GDB1`/`GDBO` (leaf); `GDV1`/`GDVO` (volume). ≥loose leaf / ≥industry volume vs Go (Vulkan f32 scan; declared GPU precision floors).
Timing: same `--bench=N` / spd%Δ convention. Truncated-BPTT (state/conv stop-grad); f32-only (no ×34).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | ≥loose (dW floor) + speed |
| 2 | Volume 1×1×1×1 | [x] | ≥industry |
| 3 | Volume 1×1×1×2 chain | [x] | ≥industry |
| 4 | Volume 2×2×2×1 | [x] | ≥industry; scaled init |

## w2a Go↔Vulkan (Stack NestedStack-of-Dense NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --stack-train`  
Wire: `STKB`/`STKO` (leaf); `STKV`/`STVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. NestedStack of Dense Dim→Dim Linear; loom n_children×[Dim×Dim]. Avoids `STB1` (StepBP).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | NestedStack Dense |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | binary 211 post denorm ≤1e-18 OK |

---

## w2a Go↔Vulkan (KMeans NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --normalbp-all` (`KMeans` and KMeans
`NormalBPEdges` groups in `TestGoVVulkanNormalBP`).
Wires: `KMB1`/`KMBO`, `KMV1`/`KMVO`; exported `welvet_kmeans_normalbp_f32` ABI.

| Evidence | Status | Scope |
|---|:---:|---|
| Probabilities and Features Float32 | [x] | Forward, stored pre, dX, dW, native SGD, finite MSE loss |
| Probabilities and Features ×34 | [x] | FormatNone native centers; both output modes |
| All eight activations / temperatures | [x] | Probabilities and Features; temperature 0→1, 0.25, 2.5; rank-3 inputs |
| Volume 111 and two-layer chain Float32 | [x] | Features, complete reverse credit and updated centers |
| Volume 111 / 211 ×34 | [x] | Output, input gradient, loss and updated native weights |
| Volume 222 Float32 | [x] | Multi-cell joins and reverse-hop credit |

**Status:** ✅ signed (Vulkan).
reconstruction, activation, assignment/center/input gradients and SGD. Go's
Float32 activation/derivative materialization points are preserved. Dynamic GPU
scratch has no fixed cluster-count cap. The narrow input-gradient cancellation
band above is explicit; larger errors and non-finite values fail. Performance
and complete graph residency remain separate work.

---

## w2a Go↔Vulkan (Parallel NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --normalbp-all` (`Parallel`, `ParallelPoly`,
Parallel `NormalBPEdges`, and `ParallelVolumes` groups).
Wires: `PNB1`/`PNBO`, `PPB1`/`PPBO`, `PNV1`/`PNVO`, `NBL1`/`NBLO`.
Exported Dense-branch ABI: `welvet_parallel_normalbp_f32`.

| Evidence | Status | Scope |
|---|:---:|---|
| All seven combines Float32 | [x] | Concat, Add, Avg, Filter, Max, SparseK, Disagree |
| Direct gradients for all seven combines ×34 | [x] | Rank-3 pre/post, dX, dW, native SGD; Filter gate weights included |
| Polymorphic branches | [x] | All 23 branch kinds Float32; 15 dtype-generic branch fixtures ×34 |
| Nested Parallel / Stack | [x] | Recursive GPU forward/backward and packed updates |
| Routing edges | [x] | First-branch Max ties, index-ordered SparseK ties, SparseK=1, zero lr, unequal-width Concat, nested Filter gate |
| Volume 111 / chain / 211 ×34 | [x] | Add and Avg; loss, output, input gradients, native weights |
| Volume 222 Float32 | [x] | Add branches through multi-cell graph joins |
| ABI / host-API guard | [x] | Exported ABI gradients and SGD match C++ Vulkan API; legacy CPU matvec still rejects |

**Status:** ✅ signed (Vulkan).
kernels and stage buffers. Vulkan performs every combine, routing operation,
softmax gate derivative and input-gradient reduction. Input gradients cast only
after the promoted branch sum; Avg backward uses the promoted reciprocal.
Integer-backed 53-bit arithmetic preserves cancellation and Go arm64 fused
multiply/add rounding in Disagree. Native FormatNone weights remain native.
Other TrainModes, full residency and throughput tuning retain separate gates.

---

## w2a Go↔Vulkan (KMeans + Parallel, sections 2–6) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --step-tween-all`.
Cases: [`kmeans_parallel_modes.go`](../../w2a/suites/go_v_vulkan/kmeans_parallel_modes.go).
Each mode adds the same 50 groups below, with fresh Go fixtures and independent
benchmark replay. The combined suite also reruns the 219 NormalBP groups and
`TestGoVVulkanOnly`.

| New evidence per mode | Groups | Status | Scope |
|---|---:|:---:|---|
| KMeans Probabilities / Features | 4 | [ ] | Float32 and ×34; SiLU, temperature 0.7, shape-changing KMeans→Dense chain |
| All seven Parallel combines | 14 | [ ] | Float32 and ×34; three branches, Filter gate export, Concat→Dense width change |
| Polymorphic Parallel branches | 23 | [ ] | Every branch kind Float32; nested Parallel/Stack, recurrent and convolutional branches |
| Stateful Filter collection | 1 | [ ] | Metacognition rules and gate, capture-before-update semantics |
| KMeans / Parallel volume credit | 8 | [ ] | 111 / two-layer chain / 211 ×34, plus 222 Float32 for each kind |

**Status:** ⬜ not signed yet (Vulkan).
Line ticks in the multi-hop fixtures. Tween and StepTween project the global
gap onto each trainable Parallel leaf and its gate at half lr; the collect tape
is complete before SGD starts. Filter performs its Go-compatible extra forward,
including stateful rule counters. TweenChain uses full-rate BP. Volume credit
uses the same shared Forward+Step paths as the existing mode fixtures; Parallel
volume Tween treats the cell as a local backward/update unit.

Leaf wires: `STB1`/`STBO`, `TWB1`/`TWBO`, `TCB1`/`TCBO`, `STW1`/`STWO`,
`STC1`/`STCO`. Volume wires: `KMV1`/`KMVO`, `KMTV`/`KMTO`, `KMTC`/`KMCO`,
`PNV1`/`PNVO`, `PNTV`/`PNTO`, `PNTC`/`PNCO`. Step variants reuse their named
parent volume credit as specified in each mode section.

Native weight checks remain exact or within the existing FP32 subnormal policy;
GDN branches retain the declared ≥strict weight floor, exact packed lengths,
and finite loss band. Benchmark outputs must match independent fresh outputs.
KMeans volume input gradients retain the narrow cancellation band declared
above. GPU-only probes also exercise KMeans/Parallel mesh SGD. Full mesh-clock
Go parity is validated separately in section 7. Full residency and throughput
remain separate work.

---

## w2a Go↔Vulkan (StepBP Line + MeshBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --stepbp-train`  
Wires: `STB1`/`STBO` (Line FamilyBP); `MBP1`/`MBPO` (Dense mesh-clock FamilyBP).
`TestGoVVulkanStepBPTrain` contains **134 case groups**, including the dtype loops below.
The runner also executes `TestGoVVulkanOnly` and defaults to one measured iteration.
Every forward, backward, loss, gradient join and SGD calculation uses Vulkan kernels;
host code schedules ticks, stores tapes and stages buffers.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Case group | Status | Evidence |
|---|:---:|---|
| Dense multi-hop Line Float32 | [ ] | Per-tick tapes and FamilyBP on pop; exact weights, finite loss band |
| Dense Line ×34 | [ ] | All supported FormatNone dtypes |
| RMSNorm / LayerNorm Line Float32 + ×34 | [ ] | Exact weights; finite loss band |
| Softmax Line Float32 + ×34 | [ ] | Weightless loss check uses the finite loss band |
| SwiGLU / Sequential / Residual Line Float32 + ×34 | [ ] | Exact weights; finite loss band |
| Mamba / Metacognition Line Float32 + ×34 | [ ] | Exact weights; finite loss band |
| GDN Line Float32 | [ ] | ≥strict weights; finite loss band; GPU float2 reductions/transcendentals |
| NestedStack Line Float32 + ×34 | [ ] | Recursive flattening and child updates |
| Embedding / MHA Line Float32 + ×34 | [ ] | Exact weights; finite loss band |
| CNN1 / ConvT1 Line Float32 + ×34 | [ ] | Explicit Line dispatch |
| CNN2/3 / ConvT2/3 Line ×34 | [ ] | Explicit rank-4/rank-5 inputs, including Float32 |
| RNN / LSTM Line Float32 + ×34 | [ ] | Exact weights; finite loss band |
| Dense mesh-clock 111 / 211 / 222 Float32 | [ ] | Double-buffer forward ticks, reverse-time credit and SGD |
| Dense mesh-clock 111 / 222 ×34 | [ ] | Repeated ticks and native storage updates |
| Dense mesh-clock disabled cell + remote link | [ ] | Disabled passthrough and remote gradient routing |
| Volume 222 Float32: all 23 supported families | [ ] | Shared volume Forward+Step path used by Go ModeMeshBP |
| Volume 211 ×34: all 22 dtype-generic families | [ ] | GDN is Float32-only; retains its volume ≥industry floor |
| Benchmark fixture reset | [ ] | Line dtype sweeps and GDN compare benchmark results with fresh-run results; Dense mesh dtype/routing fixtures do likewise |

**Scope:** volume rows deliberately reuse the NormalBP volume fixtures because
Go's grid ModeMeshBP uses that same Forward+Step path. They do not establish
mesh-clock BPTT parity for non-Dense cells. The separate `MGB1` all-family
clock fixtures and section 7 provide that evidence.
KMeans and Parallel are signed off for Line/shared-volume credit above. No CPU fallback is supplied.

**Status:** ⬜ signed (Vulkan) — `TestGoVVulkanStepBPTrain` PASS (134 groups, ~838s @ BENCH_ITERS=1).
repetitions rebuild the initial child configuration and weights; timing includes
that reset, matching the Go fixture setup. These small correctness fixtures do
not establish a performance win.

## w2a Go↔Vulkan (Tween ModeTween) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tween-train`  
Wires: `TWB1`/`TWBO` (Stack ModeTween leaf), `TWV1`/`TWVO` (Dense volume),
and per-kind volume wires `TRV1`…`MBTV`, `MCTV`, `GDTV`, `SKTV`.
`TestGoVVulkanTweenTrain` contains **130 case groups**, including **29 leaf ×34**
and **26 volume ×34** dtype sweeps. GDN is Float32-only.

All numerical projection, forward, backward, loss and SGD work runs on Vulkan.
The global gap is block-averaged onto each local output shape and applied at
`soft_lr = lr × 0.5`; Sequential/Residual expand their Dense leaves without
chain-rule credit. Host code schedules the walk and stages buffers/tapes.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Case group | Status | Evidence |
|---|:---:|---|
| Dense multi-hop Float32 + 1-hop ×34 | [ ] | Width-changing projected credit; native weight updates |
| RMSNorm / LayerNorm Float32 + ×34 | [ ] | Exact weights; finite loss band |
| Softmax Float32 + ×34 | [ ] | Weightless loss checks; GPU exp uses promoted evaluation before Float32 rounding |
| SwiGLU Float32 + ×34 | [ ] | Exact weights; finite loss band |
| Sequential / Residual Float32 + ×34 | [ ] | Leaf-projected credit, original forward inputs retained |
| Mamba / Metacognition Float32 + ×34 | [ ] | Exact weights; finite loss band |
| GDN Float32 | [ ] | ≥strict updated weights; finite loss band (one Float32-step weight difference remains) |
| NestedStack Float32 + ×34 | [ ] | Flattened leaf walk; exact weights and finite loss band |
| Embedding / MHA Float32 + ×34 | [ ] | Exact weights; finite loss band |
| CNN1/2/3 / ConvT1/2/3 ×34 | [ ] | Explicit per-dimension leaf fixtures, including Float32 and rank-5 inputs |
| RNN / LSTM Float32 + ×34 | [ ] | Exact weights; finite loss band |
| Dense volume 111 / 211 / 222 Float32 | [ ] | Reverse-hop projected local updates |
| Volume 222 Float32: all 23 supported families | [ ] | Per-kind local update dispatch; finite loss band |
| Volume 211 ×34: all 22 dtype-generic families | [ ] | Native storage updates across all supported dtypes |
| GPU projection and nested-credit primitives | [ ] | Downsample, upsample, cancellation, invalid shapes, and original-input local updates |
| Benchmark fixture reset | [ ] | All leaf dtype sweeps and Dense volume dtype cases compare measured-run output against an independent run |

**Status:** ⬜ ModeTween signed (Vulkan).
Loss uses the finite float2 band above. GDN leaf weights require ≥strict, rather
than exact bits. Volume weight subnormal flushing follows the existing Vulkan
policy. KMeans and Parallel have independent ModeTween evidence above. No host
numerical fallback is supplied.

**Validated:** 2026-10-28. `TestGoVVulkanTweenTrain` — **130 case groups** PASS in
717.464 seconds (`WELVET_VULKAN_BENCH_ITERS=1`, `-timeout 90m`). Projection uses
Vulkan `launch_project_gap` (f64 block averages). This sign-off is ModeTween only.
TweenChain / Split / Alt and Step/Mesh variants have separate sign-offs below.
Graph residency and performance tuning remain open; benchmark timings include
resetting the fixture.


## w2a Go↔Vulkan (TweenChain ModeTweenChain) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tweenchain-train`  
Wires: `TCB1`/`TCBO` (Stack FamilyBP at full lr), `TCV1`/`TCVO` (Dense volume),
and per-kind volume wires `TRC1`…`MBTC`, `MCTC`, `GDTC`, `SKTC`.
`TestGoVVulkanTweenChainTrain` contains **130 case groups**, including **29 leaf ×34**
and **26 volume ×34** dtype sweeps. GDN is Float32-only.

Forward, backward, MSE/MSEGrad and native SGD use Vulkan kernels. TweenChain
propagates the true chain-rule input gradient at **full lr**, including through
Sequential, Residual and NestedStack. Volume credit is the same Forward+Step
operation as NormalBP; the named TweenChain entry point and every wire are
exercised here. Host code handles scheduling, allocation and tape/buffer staging.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Case group | Status | Evidence |
|---|:---:|---|
| Dense multi-hop Float32 + 1-hop ×34 | [ ] | Full-lr chain-rule credit; exact updated weights and finite loss band |
| RMSNorm / LayerNorm Float32 + ×34 | [ ] | Exact weights; finite loss band |
| Softmax Float32 + ×34 | [ ] | Weightless loss check uses the finite Vulkan loss band |
| SwiGLU Float32 + ×34 | [ ] | Exact weights; finite loss band |
| Sequential / Residual Float32 + ×34 | [ ] | Full chain rule through Dense children |
| Mamba / Metacognition Float32 + ×34 | [ ] | Exact weights; finite loss band |
| GDN Float32 | [ ] | ≥strict updated weights; finite loss band; Float32 rounding difference retained |
| NestedStack Float32 + ×34 | [ ] | Recursive flattening with reverse chain-rule credit |
| Embedding / MHA Float32 + ×34 | [ ] | Exact weights; finite loss band |
| CNN1/2/3 / ConvT1/2/3 ×34 | [ ] | Explicit per-dimension fixtures, including Float32 and rank-5 inputs |
| RNN / LSTM Float32 + ×34 | [ ] | Exact weights; finite loss band |
| Dense volume 111 / 211 / 222 Float32 | [ ] | NormalBP-equivalent volume credit at full lr |
| Volume 222 Float32: all 23 supported families | [ ] | Per-kind TweenChain protocol; Softmax volume uses NormalBP Softmax floor |
| Volume 211 ×34: all 22 dtype-generic families | [ ] | Native weight updates and chain-rule input gradients |
| GPU chain-rule/full-lr primitives | [ ] | Named TCB1/TCV1 entry points; FamilyBP @ full lr via systems/tween_chain |
| Benchmark fixture reset | [ ] | Leaf dtype sweeps and Dense volume dtype fixtures compare measured-run output with an independent run |

**Status:** ⬜ ModeTweenChain signed (Vulkan).
loss band apply where specified. Softmax volume gin retains the NormalBP Softmax
DetLoose+/~strict floor. GDN leaf weights retain ≥strict; its volume retains
≥industry. The existing subnormal and KMeans cancellation policies remain
explicit above. KMeans/Parallel have independent evidence for this mode.
Remaining variants retain their own gates. Residency and performance tuning
remain open; benchmark timings include fixture reset.

## w2a Go↔Vulkan (TweenSplit ModeTweenSplit) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tweensplit-train`.
`TestGoVVulkanTweenSplitTrain` contains **131 case groups**: the existing 80 plus
51 KMeans/Parallel groups. Wires: `TSB1`/`TSBO`; existing per-kind Split volumes
plus `KMSP`/`KMSO` and `PNSP`/`PNSO`.

Single collect tape; global 1/N share at full lr. All numerical forward, loss,
projection, 1/N scaling, local backward and native SGD work uses Vulkan kernels.
Host code collects tapes, schedules leaves and stages buffers. No chain-rule
input gradient is claimed for this mode.

| Evidence | Status | Scope |
|---|:---:|---|
| All 23 supported families | [ ] | Native dtype sweeps for the 22 generic families; GDN Float32 only |
| Both KMeans outputs | [ ] | Probabilities / Features, Float32 and ×34, including shape-changing KMeans→Dense |
| Seven Parallel combines | [ ] | Float32 and ×34; Filter gate counts as a trainable leaf; Concat→Dense width changes |
| Polymorphic Parallel | [ ] | All 23 branch kinds Float32, including nested Stack/Parallel and rank-4/rank-5 convolutional branches |
| Stateful Filter | [ ] | Metacognition capture and gate update; complete tape before SGD |
| Mixed weightless/trainable leaves | [ ] | Dense + Softmax Parallel followed by Dense ×34; Softmax excluded from N |
| Shared volume credit | [ ] | All 23 families; KMeans/Parallel 111/chain/211 ×34 and 222 Float32 |
| Benchmark reset | [ ] | Fresh loss/weights match measured runs; all new leaf and volume fixtures require exact Vulkan replay across at least two iterations |
| Collect primitives | [ ] | Existing hand-calculated 1/N sharing and tape preservation guards |

Parallel leaf-mode Split recursively collects trainable branches, nested
containers, Sequential/Residual Dense leaves and the Filter gate. It reuses
Tween's collector, including Filter's extra forward and state transitions;
updates begin only after collection. Split loss is scored from the collected
output, whose native branch axes can differ from regular Parallel Forward.
The Go fixture discovers target shapes through Go's public `OpenSplitTape`.
Volume Split retains its existing grid contract: each placed KMeans/Parallel
cell is one local backward/SGD unit; Sequential/Residual expand Dense children.

Existing precision gates apply: finite loss band, exact native weights or the
documented FP32 subnormal policy. GDN leaf/Parallel branch weights retain
≥strict and GDN shared volumes retain ≥industry. No precision gates were
weakened. Other Split/Alt modes retain their separate sign-off sections. Full
residency and throughput tuning remain separate work.

**Status:** ⬜ ModeTweenSplit signed (Vulkan). All 131 TweenSplit groups passed in
775.850 seconds under the existing precision gates.

## w2a Go↔Vulkan (StepTweenSplit ModeStepTweenSplit) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptweensplit-train`.

Wires: `STS1`/`STSO`; shared volume `TSP1` and per-kind Split variants. `TestGoVVulkanStepTweenSplitTrain`: **130 case groups**.

Line per-hop Split at full lr; N counts trainable leaves under that hop. The popped global gap stays frozen. Numerical work uses Vulkan kernels; host code handles scheduling, tapes and staging. Parallel collect matches Go `combineCollected` (Filter keeps the extra forward). KMeans/Parallel Line multi-tick fixtures allow ≥strict weights where Vulkan MatMul ulps accumulate across five pops (single-pop remains exact_bit).

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [ ] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [ ] | 23 families; 22 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [ ] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [ ] | All 23 supported families (TweenSplit volume wires) |
| Volume 211 ×34 | [ ] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [ ] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [ ] | FamilyTweenSplit on pop via `apply_split_child`; frozen gap passthrough |

**Status:** ⬜ ModeStepTweenSplit signed (Vulkan). All 130 groups passed in 760.015s.

## w2a Go↔Vulkan (TweenAlt ModeTweenAlt) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tweenalt-train`  
Batch: `./scripts/linux_build_w2a.sh --tween-next-all` (all eight requested sections + Vulkan-only checks).

Wires: `TAB1`/`TABO`; volume `TAV1` and per-kind Alt variants. `TestGoVVulkanTweenAltTrain`: **130 case groups**.

Split at full lr, refresh prediction and gap, then Tween at lr×0.5. AltTimes=1 is validated. Numerical work uses Vulkan kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [ ] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [ ] | 23 families; 22 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [ ] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [ ] | All 23 supported families |
| Volume 211 ×34 | [ ] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [ ] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Collect/phase primitives | [ ] | Hand-calculated 1/N sharing and refreshed Alt loss/weights; input tape retained |

**Status:** ⬜ ModeTweenAlt signed (Vulkan). All 130 groups passed in 736.794s.

## w2a Go↔Vulkan (StepTweenAlt ModeStepTweenAlt) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptweenalt-train`  
Batch: `./scripts/linux_build_w2a.sh --tween-next-all` (all eight requested sections + Vulkan-only checks).

Wires: `STA1`/`STAO`; shared volume `TAV1` and per-kind Alt variants. `TestGoVVulkanStepTweenAltTrain`: **130 case groups**.

Line per-hop Split→Tween at full/half lr with a frozen popped gap. Its shared volume path refreshes the gap between phases. Numerical work uses Vulkan kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [ ] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [ ] | 23 families; 22 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [ ] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [ ] | All 23 supported families |
| Volume 211 ×34 | [ ] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [ ] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [ ] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Status:** ⬜ ModeStepTweenAlt signed (Vulkan). All 130 groups passed in 738.413s.

## w2a Go↔Vulkan (MeshTween mesh-clock) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --meshtween-train`.
The runner includes MeshTween, MeshBP, ModeTween and Vulkan-only regression gates.
`TestGoVVulkanMeshTweenTrain` contains **159 case groups**: 48 existing Dense/shared
volume groups, eight KMeans/Parallel volume groups, and 103 all-family clock groups.
Wires: `MGW1`/`MGWO` for actual heterogeneous clock nodes, existing `MTW1`/`MTWO`
for Dense, and the existing per-kind ModeTween volume wires.

Last-tick captured mesh credit applies local projected updates at lr×0.5.
`step::train_mse(FamilyTween)` and the generic clock wire share
`step::apply_tween`; warmup/remote fixtures supply the captured state directly.
All numerical forward, gap projection, backward and native SGD work uses Vulkan.
Host code schedules ticks, stores tapes, rebuilds fixtures and stages buffers.

| Evidence | Status | Scope |
|---|:---:|---|
| All 23 actual node kinds | [ ] | 111 / 211 ×34, plus 222 Float32; GDN Float32 only |
| Recurrent remote clock | [ ] | 22 shape-preserving families; detached warmup then captured ticks |
| KMeans output modes | [ ] | Probabilities and Features ×34 in three-node chains |
| Parallel combines | [ ] | All seven ×34, including Filter gate updates and Concat width changes |
| Shared volume credit | [ ] | All 23 families; supported dtype loops, including KMeans/Parallel 111/chain/211 ×34 and 222 Float32 |
| Zero learning rate | [ ] | Dense/KMeans/Parallel preserve native weights ×34 |
| Captured tensors and updates | [ ] | Every tick/node input, pre and post; final loss/output and all updated weights |
| Exact replay | [ ] | Independent fresh Vulkan result matches at least two measured iterations exactly |
| Credit primitive | [ ] | Existing hand-calculated two-cell update distinguishes local Tween from BPTT/full-lr credit |

**Scope:** this is last-tick local mesh credit; no temporal backward pass or
chain-rule input/accumulated parameter gradient is exported. Sequential and
Residual project onto their Dense leaves. Stack and Parallel nodes use their
concrete backward/SGD as local units, matching the existing grid ModeTween
volume contract. Gap copying continues only when the next input length matches;
otherwise the upstream gap is zero. Go references combine Go step capture,
concrete Go backward/SGD and its public local-Tween collector in w2a.
Clock precision policy: finite loss band, exact/native-subnormal weight
checks, GDN tensors ≥loose and weights ≥strict. Shared-volume checks retain
their existing gates, including GDN ≥industry. Full graph residency and
performance tuning remain separate work. MeshTweenChain retains its own gate.

**Status:** ⬜ MeshTween signed (Vulkan). All 159 groups passed in 1551.91s.
Combined runner (MeshTween + MeshBP + ModeTween) passed in 3557.079s.

## w2a Go↔Vulkan (MeshTweenChain mesh-clock) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --meshtweenchain-train`.
The runner covers MeshTweenChain, MeshTween, ModeTweenChain and Vulkan-only guards.
`TestGoVVulkanMeshTweenChainTrain` contains **160 case groups**: 48 existing
Dense/shared-volume groups, eight KMeans/Parallel volume groups, 103 all-family
clock groups, and one direct 23-family MeshBP equivalence group.

Wires: `MGC1`/`MGCO` binds actual heterogeneous clock nodes; existing Dense
`MTC1`/`MTCO` and per-kind TweenChain volume wires retain their coverage.
MeshTweenChain uses the existing `runtime/step` FamilyBP path: captured forward
ticks → reverse-time BPTT → accumulated parameter gradients → native SGD at full
lr. Every numerical operation uses Vulkan kernels; host code schedules ticks,
stores tapes, stages buffers and rebuilds replay fixtures. The wire alias adds
no separate numerical implementation or CPU fallback.

| Evidence | Status | Scope |
|---|:---:|---|
| All 23 actual node kinds | [ ] | 111 / 211 ×34 for 22 dtype-generic families; GDN Float32 only; all 23 at 222 Float32 |
| Recurrent remote clock | [ ] | 22 shape-preserving families; detached warmup then captured ticks |
| KMeans output modes | [ ] | Probabilities / Features ×34 with real Dense adapter nodes |
| Parallel combines | [ ] | All seven ×34, including Filter gate gradients and Concat shape changes |
| Captured tensors and gradients | [ ] | Every tick/node input, pre and post; final output/loss, oldest input gradient and packed accumulated parameter gradients |
| Full-rate native SGD | [ ] | Updated weights for every enabled node; Dense/KMeans/Parallel zero-lr preservation ×34 |
| Exact replay | [ ] | Fresh Vulkan result equals at least two measured repetitions across all exported tensors, weights, tapes and loss |
| MeshBP equivalence | [ ] | Independent MeshBP and MeshTweenChain wire calls agree exactly for all 23 families at 211 Float32 |
| Shared volume credit | [ ] | All 23 families, including KMeans/Parallel 111/chain/211 ×34 and 222 Float32 |
| Credit primitive | [ ] | Existing hand-calculated test distinguishes full-rate BPTT from last-tick half-rate MeshTween |

The existing precision policy applies: finite loss band; exact/subnormal tensor
and weight gates where specified; MHA gradients ≥strict, Softmax gradients ≥loose;
GDN clock tensors ≥loose and weights ≥strict. GDN shared-volume checks retain
≥industry. Only KMeans input gradients use the narrow cancellation allowance.
Parameter gradient axes are flattened for packed comparison; activation and
input-gradient shapes remain strict. Embedding uses one 111 tick and disabled
ingress plus RMSNorm adapter nodes in grids. SwiGLU uses the bounded recurrent
trajectory from section 7. These are correctness fixtures; full residency and
throughput tuning retain separate gates.

**Status:** ⬜ MeshTweenChain signed (Vulkan). All 160 groups passed in 1555.46s.
Combined runner (MeshTweenChain + MeshTween + ModeTweenChain) passed in 3826.581s.

## w2a Go↔Vulkan (StepTween ModeStepTween) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptween-train`  
Batch: `./scripts/linux_build_w2a.sh --step-tween-all` (sections 2–6 + NormalBP + Vulkan-only checks).

Wires: `STW1`/`STWO`; shared volume `TWV1` and per-kind Tween variants. `TestGoVVulkanStepTweenTrain`: **130 case groups**.

Line per-hop projected local credit at lr×0.5, applied when a training flight pops. Numerical work uses Vulkan kernels; host code handles scheduling, tapes and staging.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [ ] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [ ] | 23 families; 22 dtype-generic families ×34, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [ ] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [ ] | All 23 supported families |
| Volume 211 ×34 | [ ] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [ ] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [ ] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Status:** ⬜ ModeStepTween signed (Vulkan). All 130 groups passed in 714.36s.
Combined runner (StepTween + ModeTween) passed in 1427.248s.
Vulkan loss band apply where specified; GDN branches retain ≥strict. The
existing subnormal policy and KMeans volume cancellation band remain explicit.
KMeans/Parallel evidence is included above. Full graph residency and
performance tuning remain open.

## w2a Go↔Vulkan (StepTweenChain ModeStepTweenChain) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptweenchain-train`  
Batch: `./scripts/linux_build_w2a.sh --step-tween-all` (sections 2–6 + NormalBP + Vulkan-only checks).

Wires: `STC1`/`STCO`; shared volume `TCV1` and per-kind TweenChain variants. `TestGoVVulkanStepTweenChainTrain`: **130 case groups**.

Line per-hop true chain-rule BP at full lr, applied when a training flight pops. Numerical work uses Vulkan kernels; host code handles scheduling, tapes and staging.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [ ] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [ ] | 23 families; 22 dtype-generic families ×34, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [ ] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [ ] | All 23 supported families |
| Volume 211 ×34 | [ ] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [ ] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [ ] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Status:** ⬜ ModeStepTweenChain signed (Vulkan). All 130 groups passed in 729.47s.
Combined runner (StepTweenChain + ModeTweenChain) passed in 1456.056s.
Vulkan loss band apply where specified; GDN branches retain ≥strict. The
existing subnormal policy and KMeans volume cancellation band remain explicit.
KMeans/Parallel evidence is included above. Full graph residency and
performance tuning remain open.

## 1 — NormalBP

Chain-rule Backward + ApplyGradSGD. Start here. Dense first.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 0.51 |
| RMSNorm | [x] | [x] | [x] | 0.51 |
| LayerNorm | [x] | [x] | [x] | 0.51 |
| SwiGLU | [x] | [x] | [x] | 0.51 |
| Softmax | [x] | [x] | [x] | 0.51 |
| Embedding | [x] | [x] | [x] | 0.51 |
| MHA | [x] | [x] | [x] | 0.51 |
| CNN1 | [x] | [x] | [x] | 0.51 |
| CNN2 | [x] | [x] | [x] | 0.51 |
| CNN3 | [x] | [x] | [x] | 0.51 |
| ConvT1 | [x] | [x] | [x] | 0.51 |
| ConvT2 | [x] | [x] | [x] | 0.51 |
| ConvT3 | [x] | [x] | [x] | 0.51 |
| RNN | [x] | [x] | [x] | 0.51 |
| LSTM | [x] | [x] | [x] | 0.51 |
| Sequential | [x] | [x] | [x] | 0.51 |
| Residual | [x] | [x] | [x] | 0.51 |
| KMeans | [x] | [x] | [x] | 0.51 |
| Mamba | [x] | [x] | [x] | 0.51 |
| Metacognition | [x] | [x] | [x] | 0.51 |
| GDN | [x] | [x] | [x] | 0.51 |
| Parallel | [x] | [x] | [x] | 0.51 |
| Stack | [x] | [x] | [x] | 0.51 |

**Checkpoint:** ✅ NormalBP signed (Vulkan) — `TestGoVVulkanNormalBP` PASS; leaf+vol ×34 (GDN f32). Next: StepBP / MeshBP.

---

## 2 — StepBP

Needs Step `Line` pipe + FamilyBP (NormalBP credit) on pop.  
Ship: thin Stack host + `runtime/line` TickLine/TrainLineMSE; wire `STB1`/`STBO`; w2a `--stepbp-train`.  
**3D volumetric** for StepBP = MeshBP (`runtime/step` mesh-clock FamilyBP + ModeMeshBP volume Step). All CNN/ConvT dimensions have explicit Line ×34 fixtures, including rank-5 inputs.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.52 |
| RMSNorm | [ ] | [ ] | [ ] | 0.52 |
| LayerNorm | [ ] | [ ] | [ ] | 0.52 |
| Softmax | [ ] | [ ] | [ ] | 0.52 |
| SwiGLU | [ ] | [ ] | [ ] | 0.52 |
| Embedding | [ ] | [ ] | [ ] | 0.52 |
| MHA | [ ] | [ ] | [ ] | 0.52 |
| CNN1 | [ ] | [ ] | [ ] | 0.52 |
| CNN2 | [ ] | [ ] | [ ] | 0.52 |
| CNN3 | [ ] | [ ] | [ ] | 0.52 |
| ConvT1 | [ ] | [ ] | [ ] | 0.52 |
| ConvT2 | [ ] | [ ] | [ ] | 0.52 |
| ConvT3 | [ ] | [ ] | [ ] | 0.52 |
| RNN | [ ] | [ ] | [ ] | 0.52 |
| LSTM | [ ] | [ ] | [ ] | 0.52 |
| Sequential | [ ] | [ ] | [ ] | 0.52 |
| Residual | [ ] | [ ] | [ ] | 0.52 |
| KMeans | [ ] | [ ] | [ ] | 0.52 |
| Mamba | [ ] | [ ] | [ ] | 0.52 |
| Metacognition | [ ] | [ ] | [ ] | 0.52 |
| GDN | [ ] | [ ] | [ ] | 0.52 |
| Parallel | [ ] | [ ] | [ ] | 0.52 |
| Stack | [ ] | [ ] | [ ] | 0.52 |

**Checkpoint:** ⬜ StepBP Line + Dense MeshBP signed (Vulkan) — STB1 FamilyBP + MBP1 mesh-clock + ModeMeshBP vols; KMeans/Parallel Line included.

---

## 3 — Tween

ModeTween: projected local credit @ `soft_lr = lr × 0.5` (Go `applyProjectedTween` — same gap on every leaf, no chain rule).  
Ship: `systems/tween` + `runtime/tween_train`; wires `TWB1`/`TWBO` + `TWV1`/`TWVO` + per-kind `TRV1`…`MBTV`; w2a `--tween-train`.  
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. 3D = reverse-hop volume 2×2×2×1 for all NormalBP-signed kinds.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.54 |
| RMSNorm | [ ] | [ ] | [ ] | 0.54 |
| LayerNorm | [ ] | [ ] | [ ] | 0.54 |
| SwiGLU | [ ] | [ ] | [ ] | 0.54 |
| Softmax | [ ] | [ ] | [ ] | 0.54 |
| Embedding | [ ] | [ ] | [ ] | 0.54 |
| MHA | [ ] | [ ] | [ ] | 0.54 |
| CNN1 | [ ] | [ ] | [ ] | 0.54 |
| CNN2 | [ ] | [ ] | [ ] | 0.54 |
| CNN3 | [ ] | [ ] | [ ] | 0.54 |
| ConvT1 | [ ] | [ ] | [ ] | 0.54 |
| ConvT2 | [ ] | [ ] | [ ] | 0.54 |
| ConvT3 | [ ] | [ ] | [ ] | 0.54 |
| RNN | [ ] | [ ] | [ ] | 0.54 |
| LSTM | [ ] | [ ] | [ ] | 0.54 |
| Sequential | [ ] | [ ] | [ ] | 0.54 |
| Residual | [ ] | [ ] | [ ] | 0.54 |
| KMeans | [ ] | [ ] | [ ] | 0.54 |
| Mamba | [ ] | [ ] | [ ] | 0.54 |
| Metacognition | [ ] | [ ] | [ ] | 0.54 |
| GDN | [ ] | [ ] | [ ] | 0.54 |
| Parallel | [ ] | [ ] | [ ] | 0.54 |
| Stack | [ ] | [ ] | [ ] | 0.54 |

**Checkpoint:** ⬜ ModeTween signed (Vulkan) — projected local @ `lr×0.5`; TWB1/TWV1 + per-kind volumes + KMeans/Parallel; 130 groups.

---

## 4 — TweenChain

ModeTweenChain: named Tween, credit = BP (Go `familyTweenChain` ≡ `familyBP` @ full lr).  
Ship: `systems/tween_chain` + `runtime/tween_chain_train`; wires `TCB1`/`TCBO` + `TCV1`/`TCVO` + per-kind `TRC1`…`MBTC`; w2a `--tweenchain-train`.  
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. 3D = volume Step 2×2×2×1 for all NormalBP-signed kinds.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.55 |
| RMSNorm | [ ] | [ ] | [ ] | 0.55 |
| LayerNorm | [ ] | [ ] | [ ] | 0.55 |
| SwiGLU | [ ] | [ ] | [ ] | 0.55 |
| Softmax | [ ] | [ ] | [ ] | 0.55 |
| Embedding | [ ] | [ ] | [ ] | 0.55 |
| MHA | [ ] | [ ] | [ ] | 0.55 |
| CNN1 | [ ] | [ ] | [ ] | 0.55 |
| CNN2 | [ ] | [ ] | [ ] | 0.55 |
| CNN3 | [ ] | [ ] | [ ] | 0.55 |
| ConvT1 | [ ] | [ ] | [ ] | 0.55 |
| ConvT2 | [ ] | [ ] | [ ] | 0.55 |
| ConvT3 | [ ] | [ ] | [ ] | 0.55 |
| RNN | [ ] | [ ] | [ ] | 0.55 |
| LSTM | [ ] | [ ] | [ ] | 0.55 |
| Sequential | [ ] | [ ] | [ ] | 0.55 |
| Residual | [ ] | [ ] | [ ] | 0.55 |
| KMeans | [ ] | [ ] | [ ] | 0.55 |
| Mamba | [ ] | [ ] | [ ] | 0.55 |
| Metacognition | [ ] | [ ] | [ ] | 0.55 |
| GDN | [ ] | [ ] | [ ] | 0.55 |
| Parallel | [ ] | [ ] | [ ] | 0.55 |
| Stack | [ ] | [ ] | [ ] | 0.55 |

**Checkpoint:** ⬜ ModeTweenChain signed (Vulkan) — FamilyBP @ full lr; TCB1/TCV1 + per-kind volumes + KMeans/Parallel; 130 groups.

---

## 5 — StepTween

Needs Line + Tween.  
Ship: `runtime/line` Credit::FamilyTween on pop (`apply_projected` @ `lr×0.5`); wire `STW1`/`STWO`; w2a `--steptween-train`.  
**3D volumetric** for StepTween = MeshTween volume (= ModeTween reverse-hop projected local; existing `TWV1`/`TRV1`… wires). CNN2/3 and ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.62 |
| RMSNorm | [ ] | [ ] | [ ] | 0.62 |
| LayerNorm | [ ] | [ ] | [ ] | 0.62 |
| SwiGLU | [ ] | [ ] | [ ] | 0.62 |
| Softmax | [ ] | [ ] | [ ] | 0.62 |
| Embedding | [ ] | [ ] | [ ] | 0.62 |
| MHA | [ ] | [ ] | [ ] | 0.62 |
| CNN1 | [ ] | [ ] | [ ] | 0.62 |
| CNN2 | [ ] | [ ] | [ ] | 0.62 |
| CNN3 | [ ] | [ ] | [ ] | 0.62 |
| ConvT1 | [ ] | [ ] | [ ] | 0.62 |
| ConvT2 | [ ] | [ ] | [ ] | 0.62 |
| ConvT3 | [ ] | [ ] | [ ] | 0.62 |
| RNN | [ ] | [ ] | [ ] | 0.62 |
| LSTM | [ ] | [ ] | [ ] | 0.62 |
| Sequential | [ ] | [ ] | [ ] | 0.62 |
| Residual | [ ] | [ ] | [ ] | 0.62 |
| KMeans | [ ] | [ ] | [ ] | 0.62 |
| Mamba | [ ] | [ ] | [ ] | 0.62 |
| Metacognition | [ ] | [ ] | [ ] | 0.62 |
| GDN | [ ] | [ ] | [ ] | 0.62 |
| Parallel | [ ] | [ ] | [ ] | 0.62 |
| Stack | [ ] | [ ] | [ ] | 0.62 |

**Checkpoint:** ⬜ ModeStepTween signed (Vulkan) — Line FamilyTween `apply_projected` @ lr×0.5; STW1/STWO + ModeTween volumes; 130 groups in 714.36s.

---

## 6 — StepTweenChain

Needs Line + TweenChain.  
Ship: `runtime/line` Credit::FamilyBP on pop (same credit as StepBP / TweenChain); wire `STC1`/`STCO`; w2a `--steptweenchain-train`.  
**3D volumetric** = MeshTweenChain volume (= ModeTweenChain volume Step wires `TCV1`/`TRC1`…).

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.63 |
| RMSNorm | [ ] | [ ] | [ ] | 0.63 |
| LayerNorm | [ ] | [ ] | [ ] | 0.63 |
| SwiGLU | [ ] | [ ] | [ ] | 0.63 |
| Softmax | [ ] | [ ] | [ ] | 0.63 |
| Embedding | [ ] | [ ] | [ ] | 0.63 |
| MHA | [ ] | [ ] | [ ] | 0.63 |
| CNN1 | [ ] | [ ] | [ ] | 0.63 |
| CNN2 | [ ] | [ ] | [ ] | 0.63 |
| CNN3 | [ ] | [ ] | [ ] | 0.63 |
| ConvT1 | [ ] | [ ] | [ ] | 0.63 |
| ConvT2 | [ ] | [ ] | [ ] | 0.63 |
| ConvT3 | [ ] | [ ] | [ ] | 0.63 |
| RNN | [ ] | [ ] | [ ] | 0.63 |
| LSTM | [ ] | [ ] | [ ] | 0.63 |
| Sequential | [ ] | [ ] | [ ] | 0.63 |
| Residual | [ ] | [ ] | [ ] | 0.63 |
| KMeans | [ ] | [ ] | [ ] | 0.63 |
| Mamba | [ ] | [ ] | [ ] | 0.63 |
| Metacognition | [ ] | [ ] | [ ] | 0.63 |
| GDN | [ ] | [ ] | [ ] | 0.63 |
| Parallel | [ ] | [ ] | [ ] | 0.63 |
| Stack | [ ] | [ ] | [ ] | 0.63 |

**Checkpoint:** ⬜ ModeStepTweenChain signed (Vulkan) — Line FamilyBP alias (`STC1`/`STCO` ≡ StepBP); ModeTweenChain volumes; 130 groups in 729.47s.

---

## 7 — MeshBP

Grid tick FamilyBP: `runtime/step` Forward×ticks → reverse-time Backward BPTT →
ApplyGradSGD. `MGB1`/`MGBO` binds actual heterogeneous node kinds; existing
`MBP1`/`MBPO` Dense fixtures remain covered. Numerical forward, loss, backward,
gradient joins and native SGD use Vulkan kernels. Host code schedules ticks,
stores tapes, stages buffers and rebuilds benchmark fixtures.

Run: `./scripts/linux_build_w2a.sh --meshbp-train`.
The runner includes `TestGoVVulkanMeshBPTrain` (**103 case groups**), the existing StepBP/volume suite,
and NormalBP. Go ModeMeshBP on grid also uses the shared
volume Forward+Step path validated at 222 Float32 and 211 ×34.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.53 |
| RMSNorm | [ ] | [ ] | [ ] | 0.53 |
| LayerNorm | [ ] | [ ] | [ ] | 0.53 |
| SwiGLU | [ ] | [ ] | [ ] | 0.53 |
| Softmax | [ ] | [ ] | [ ] | 0.53 |
| Embedding | [ ] | [ ] | [ ] | 0.53 |
| MHA | [ ] | [ ] | [ ] | 0.53 |
| CNN1 | [ ] | [ ] | [ ] | 0.53 |
| CNN2 | [ ] | [ ] | [ ] | 0.53 |
| CNN3 | [ ] | [ ] | [ ] | 0.53 |
| ConvT1 | [ ] | [ ] | [ ] | 0.53 |
| ConvT2 | [ ] | [ ] | [ ] | 0.53 |
| ConvT3 | [ ] | [ ] | [ ] | 0.53 |
| RNN | [ ] | [ ] | [ ] | 0.53 |
| LSTM | [ ] | [ ] | [ ] | 0.53 |
| Sequential | [ ] | [ ] | [ ] | 0.53 |
| Residual | [ ] | [ ] | [ ] | 0.53 |
| KMeans | [ ] | [ ] | [ ] | 0.53 |
| Mamba | [ ] | [ ] | [ ] | 0.53 |
| Metacognition | [ ] | [ ] | [ ] | 0.53 |
| GDN | [ ] | [ ] | [ ] | 0.53 |
| Parallel | [ ] | [ ] | [ ] | 0.53 |
| Stack | [ ] | [ ] | [ ] | 0.53 |

**Validated:** 2026-10-28. The 103 MeshBP case groups passed in
1356.030 seconds (`WELVET_VULKAN_BENCH_ITERS=1`) with at least two measured
replay iterations per fixture. Shared build passed.

**Checkpoint:** ⬜ MeshBP signed (Vulkan) — all 23 supported families have actual
mesh-clock node and grid BPTT evidence. All 22 dtype-generic families cover
111 and 211 ×34; GDN is Float32-only. Every family also covers 222 Float32.
Shape-changing Embedding uses a single-tick 111 node and disabled ingress plus
RMSNorm adapter nodes in grids; its integer indices are not fed back as
embedding outputs. Remote self-links cover the 22 shape-preserving families
with detached warmup followed by captured ticks. SwiGLU uses a shorter bounded
recurrent trajectory to avoid vanishing intermediate subnormals outside the
declared GPU precision contract.

Both KMeans outputs and all seven Parallel combines additionally cover ×34
three-node clock chains, including shape-changing Concat/probabilities→Dense
and Filter gate gradients/updates. Dense/KMeans/Parallel zero-lr cases preserve
native weights across ×34; `step::apply_sgd(0)` is a no-op.

Checks include every captured node input/pre/post, final output/loss, oldest
input gradient, packed accumulated parameter gradients, and post-SGD native
weights. Parameter gradients use a flat packed axis for comparison; activation
and input-gradient shapes remain strict. Existing precision policy applies:
GDN tensors ≥loose and updated weights ≥strict, MHA gradients ≥strict, Softmax
gradients ≥loose, other gates exact or the documented FP32 subnormal policy.
Only KMeans input gradients use its narrow near-zero cancellation allowance.
Fresh and benchmark Vulkan runs must match exactly across all exported tensors,
loss, weights and tapes. MeshBP measures at least two iterations to exercise
reset between repetitions. Replay timing includes rebuilding/staging; residency
and performance tuning retain separate gates. MeshTween/other modes retain
their own clock sign-off below.

## 8 — MeshTween

Needs Mesh + Tween.  
Ship: `runtime/step` `Credit::FamilyTween` — Forward×ticks → last-tick `apply_tween` / `apply_cell_projected` @ `lr×0.5` (`MGW1`/`MGWO`, existing Dense `MTW1`/`MTWO`); w2a `--meshtween-train`.

**3D volumetric** = ModeTween volume credit (`TWV1`/`TRV1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.60 |
| RMSNorm | [ ] | [ ] | [ ] | 0.60 |
| LayerNorm | [ ] | [ ] | [ ] | 0.60 |
| SwiGLU | [ ] | [ ] | [ ] | 0.60 |
| Softmax | [ ] | [ ] | [ ] | 0.60 |
| Embedding | [ ] | [ ] | [ ] | 0.60 |
| MHA | [ ] | [ ] | [ ] | 0.60 |
| CNN1 | [ ] | [ ] | [ ] | 0.60 |
| CNN2 | [ ] | [ ] | [ ] | 0.60 |
| CNN3 | [ ] | [ ] | [ ] | 0.60 |
| ConvT1 | [ ] | [ ] | [ ] | 0.60 |
| ConvT2 | [ ] | [ ] | [ ] | 0.60 |
| ConvT3 | [ ] | [ ] | [ ] | 0.60 |
| RNN | [ ] | [ ] | [ ] | 0.60 |
| LSTM | [ ] | [ ] | [ ] | 0.60 |
| Sequential | [ ] | [ ] | [ ] | 0.60 |
| Residual | [ ] | [ ] | [ ] | 0.60 |
| KMeans | [ ] | [ ] | [ ] | 0.60 |
| Mamba | [ ] | [ ] | [ ] | 0.60 |
| Metacognition | [ ] | [ ] | [ ] | 0.60 |
| GDN | [ ] | [ ] | [ ] | 0.60 |
| Parallel | [ ] | [ ] | [ ] | 0.60 |
| Stack | [ ] | [ ] | [ ] | 0.60 |

**Checkpoint:** ⬜ MeshTween signed (Vulkan) — last-tick FamilyTween @ lr×0.5; MTW1/MTWO + MGW1/MGWO + ModeTween volumes; 159 groups in 1551.91s.

---

## 9 — MeshTweenChain

Needs Mesh + TweenChain.  
Ship: `runtime/step` `Credit::FamilyBP` under mesh clock (`MGC1`/`MGCO`, existing Dense `MTC1`/`MTCO` ≡ MeshBP); w2a `--meshtweenchain-train`.

**3D volumetric** = ModeTweenChain volume credit (`TCV1`/`TRC1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.61 |
| RMSNorm | [ ] | [ ] | [ ] | 0.61 |
| LayerNorm | [ ] | [ ] | [ ] | 0.61 |
| SwiGLU | [ ] | [ ] | [ ] | 0.61 |
| Softmax | [ ] | [ ] | [ ] | 0.61 |
| Embedding | [ ] | [ ] | [ ] | 0.61 |
| MHA | [ ] | [ ] | [ ] | 0.61 |
| CNN1 | [ ] | [ ] | [ ] | 0.61 |
| CNN2 | [ ] | [ ] | [ ] | 0.61 |
| CNN3 | [ ] | [ ] | [ ] | 0.61 |
| ConvT1 | [ ] | [ ] | [ ] | 0.61 |
| ConvT2 | [ ] | [ ] | [ ] | 0.61 |
| ConvT3 | [ ] | [ ] | [ ] | 0.61 |
| RNN | [ ] | [ ] | [ ] | 0.61 |
| LSTM | [ ] | [ ] | [ ] | 0.61 |
| Sequential | [ ] | [ ] | [ ] | 0.61 |
| Residual | [ ] | [ ] | [ ] | 0.61 |
| KMeans | [ ] | [ ] | [ ] | 0.61 |
| Mamba | [ ] | [ ] | [ ] | 0.61 |
| Metacognition | [ ] | [ ] | [ ] | 0.61 |
| GDN | [ ] | [ ] | [ ] | 0.61 |
| Parallel | [ ] | [ ] | [ ] | 0.61 |
| Stack | [ ] | [ ] | [ ] | 0.61 |

**Status:** ⬜ MeshTweenChain signed (Vulkan). All 160 groups passed in 1555.46s.
Combined runner (MeshTweenChain + MeshTween + ModeTweenChain) passed in 3826.581s.

**Checkpoint:** ⬜ MeshTweenChain signed (Vulkan) — FamilyBP BPTT+SGD under mesh clock;
MTC1/MTCO + MGC1/MGCO ≡ MeshBP; ModeTweenChain volumes; 160 groups in 1555.46s.
The 22 dtype-generic families cover 111/211 ×34; GDN is Float32-only.
All 23 cover 222 Float32. Existing shared-volume gates retain their floors.

Clock evidence includes every tick/node input, pre and post; final loss/output;
oldest input gradients; packed parameter gradients accumulated across ticks;
and full-lr native SGD weights. Recurrent remote links cover 22 shape-preserving
families after detached warmup. Both KMeans output modes and all seven Parallel
combines additionally cover ×34 three-node chains. Dense/KMeans/Parallel zero-lr
cases preserve native weights ×34. Fresh and at least two measured Vulkan replays
match exactly. A direct 211 Float32 comparison verifies exact MeshBP/MeshTweenChain
wire equivalence for every supported family.

Embedding uses one 111 tick and disabled ingress plus RMSNorm adapters in grids;
its integer indices are not fed back from embedding outputs. SwiGLU uses the
bounded recurrent trajectory from section 7. Existing clock precision gates
apply: finite loss band, exact/subnormal gates, MHA gradients ≥strict, Softmax
gradients ≥loose, GDN tensors ≥loose and weights ≥strict; only KMeans input
gradients use its narrow cancellation allowance. Gradient packing normalizes
only parameter axes; activation/input-gradient shapes remain strict. The named
wire reuses Vulkan FamilyBP BPTT/SGD; it adds no numerical fallback. Residency
and performance tuning remain separate work.

---

## 10 — TweenSplit

Ship: `systems::tween_split` + `runtime/tween_split_train` — collect trainable leaves, apply \(g_i=\frac1N P(g_y)\) @ full lr (`TSB1`/`TSP1`/`TRP1`…); w2a `--tweensplit-train`.  
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. Softmax weightless (N may be 0).

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.56 |
| RMSNorm | [ ] | [ ] | [ ] | 0.56 |
| LayerNorm | [ ] | [ ] | [ ] | 0.56 |
| SwiGLU | [ ] | [ ] | [ ] | 0.56 |
| Softmax | [ ] | [ ] | [ ] | 0.56 |
| Embedding | [ ] | [ ] | [ ] | 0.56 |
| MHA | [ ] | [ ] | [ ] | 0.56 |
| CNN1 | [ ] | [ ] | [ ] | 0.56 |
| CNN2 | [ ] | [ ] | [ ] | 0.56 |
| CNN3 | [ ] | [ ] | [ ] | 0.56 |
| ConvT1 | [ ] | [ ] | [ ] | 0.56 |
| ConvT2 | [ ] | [ ] | [ ] | 0.56 |
| ConvT3 | [ ] | [ ] | [ ] | 0.56 |
| RNN | [ ] | [ ] | [ ] | 0.56 |
| LSTM | [ ] | [ ] | [ ] | 0.56 |
| Sequential | [ ] | [ ] | [ ] | 0.56 |
| Residual | [ ] | [ ] | [ ] | 0.56 |
| KMeans | [ ] | [ ] | [ ] | 0.56 |
| Mamba | [ ] | [ ] | [ ] | 0.56 |
| Metacognition | [ ] | [ ] | [ ] | 0.56 |
| GDN | [ ] | [ ] | [ ] | 0.56 |
| Parallel | [ ] | [ ] | [ ] | 0.56 |
| Stack | [ ] | [ ] | [ ] | 0.56 |

**Checkpoint:** ⬜ ModeTweenSplit signed (Vulkan) — OpenSplitTape 1/N @ full lr; TSB1/TSP1 + per-kind volumes + KMeans/Parallel; 131 groups in 775.850s.

Parallel leaf-mode Split counts all trainable branch leaves, nested Stack/Parallel
leaves and its Filter gate before applying projected 1/N gaps at full lr. Softmax
is excluded from N, including the mixed Dense/Softmax ×34 fixture. Capture and
stateful Filter's extra forward finish before any update. Split loss and targets
use the collected output's native axes, matching Go `OpenSplitTape`.
Placed KMeans/Parallel volume cells retain the existing local-unit grid contract.

All 51 KMeans/Parallel groups passed, including 111/chain/211 ×34 and 222 Float32
volumes, all 23 polymorphic Parallel branch kinds, stateful Filter, native SGD and
exact fresh-versus-measured replay with at least two iterations. Numerical work
uses Vulkan kernels; Go references stay in w2a. Other Split/Alt mode sign-offs remain
separate.

---

## 11 — StepTweenSplit

Needs Line + TweenSplit.  
Ship: `runtime/line` Credit::FamilyTweenSplit on pop (`apply_split_child` 1/N @ full lr); wire `STS1`/`STSO`; w2a `--steptweensplit-train`.  
**3D volumetric** = MeshTweenSplit volume (= ModeTweenSplit / TweenSplit volume wires `TSP1`/`TRP1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.57 |
| RMSNorm | [ ] | [ ] | [ ] | 0.57 |
| LayerNorm | [ ] | [ ] | [ ] | 0.57 |
| SwiGLU | [ ] | [ ] | [ ] | 0.57 |
| Softmax | [ ] | [ ] | [ ] | 0.57 |
| Embedding | [ ] | [ ] | [ ] | 0.57 |
| MHA | [ ] | [ ] | [ ] | 0.57 |
| CNN1 | [ ] | [ ] | [ ] | 0.57 |
| CNN2 | [ ] | [ ] | [ ] | 0.57 |
| CNN3 | [ ] | [ ] | [ ] | 0.57 |
| ConvT1 | [ ] | [ ] | [ ] | 0.57 |
| ConvT2 | [ ] | [ ] | [ ] | 0.57 |
| ConvT3 | [ ] | [ ] | [ ] | 0.57 |
| RNN | [ ] | [ ] | [ ] | 0.57 |
| LSTM | [ ] | [ ] | [ ] | 0.57 |
| Sequential | [ ] | [ ] | [ ] | 0.57 |
| Residual | [ ] | [ ] | [ ] | 0.57 |
| KMeans | [ ] | [ ] | [ ] | 0.57 |
| Mamba | [ ] | [ ] | [ ] | 0.57 |
| Metacognition | [ ] | [ ] | [ ] | 0.57 |
| GDN | [ ] | [ ] | [ ] | 0.57 |
| Parallel | [ ] | [ ] | [ ] | 0.57 |
| Stack | [ ] | [ ] | [ ] | 0.57 |

**Checkpoint:** ⬜ ModeStepTweenSplit signed (Vulkan) — Line FamilyTweenSplit 1/N @ full lr; STS1/STSO + TweenSplit volumes; 130 groups in 760.015s.

---

## 12 — TweenAlt

ModeTweenAlt: Split at full lr, refresh the prediction/gap, then Tween at lr×0.5.
Ship: `systems/tween_alt` + `runtime/tween_alt_train`; `TAB1`/`TABO` and `TAV1`/per-kind volume wires.
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. Default AltTimes=1.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.58 |
| RMSNorm | [ ] | [ ] | [ ] | 0.58 |
| LayerNorm | [ ] | [ ] | [ ] | 0.58 |
| SwiGLU | [ ] | [ ] | [ ] | 0.58 |
| Softmax | [ ] | [ ] | [ ] | 0.58 |
| Embedding | [ ] | [ ] | [ ] | 0.58 |
| MHA | [ ] | [ ] | [ ] | 0.58 |
| CNN1 | [ ] | [ ] | [ ] | 0.58 |
| CNN2 | [ ] | [ ] | [ ] | 0.58 |
| CNN3 | [ ] | [ ] | [ ] | 0.58 |
| ConvT1 | [ ] | [ ] | [ ] | 0.58 |
| ConvT2 | [ ] | [ ] | [ ] | 0.58 |
| ConvT3 | [ ] | [ ] | [ ] | 0.58 |
| RNN | [ ] | [ ] | [ ] | 0.58 |
| LSTM | [ ] | [ ] | [ ] | 0.58 |
| Sequential | [ ] | [ ] | [ ] | 0.58 |
| Residual | [ ] | [ ] | [ ] | 0.58 |
| KMeans | [ ] | [ ] | [ ] | 0.58 |
| Mamba | [ ] | [ ] | [ ] | 0.58 |
| Metacognition | [ ] | [ ] | [ ] | 0.58 |
| GDN | [ ] | [ ] | [ ] | 0.58 |
| Parallel | [ ] | [ ] | [ ] | 0.58 |
| Stack | [ ] | [ ] | [ ] | 0.58 |

**Checkpoint:** ⬜ ModeTweenAlt signed (Vulkan) — Split@lr → refresh → Tween@lr×0.5; TAB1/TABO + TAV1/per-kind; 130 groups in 736.794s.

---

## 13 — StepTweenAlt

Needs Line + TweenAlt.  
Ship: `runtime/line` Credit::FamilyTweenAlt on pop (`apply_alt_child` Split→Tween frozen gap); wire `STA1`/`STAO`; w2a `--steptweenalt-train`.  
**3D volumetric** = MeshTweenAlt volume (= ModeTweenAlt / TweenAlt volume wires `TAV1`/`TRA1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.59 |
| RMSNorm | [ ] | [ ] | [ ] | 0.59 |
| LayerNorm | [ ] | [ ] | [ ] | 0.59 |
| SwiGLU | [ ] | [ ] | [ ] | 0.59 |
| Softmax | [ ] | [ ] | [ ] | 0.59 |
| Embedding | [ ] | [ ] | [ ] | 0.59 |
| MHA | [ ] | [ ] | [ ] | 0.59 |
| CNN1 | [ ] | [ ] | [ ] | 0.59 |
| CNN2 | [ ] | [ ] | [ ] | 0.59 |
| CNN3 | [ ] | [ ] | [ ] | 0.59 |
| ConvT1 | [ ] | [ ] | [ ] | 0.59 |
| ConvT2 | [ ] | [ ] | [ ] | 0.59 |
| ConvT3 | [ ] | [ ] | [ ] | 0.59 |
| RNN | [ ] | [ ] | [ ] | 0.59 |
| LSTM | [ ] | [ ] | [ ] | 0.59 |
| Sequential | [ ] | [ ] | [ ] | 0.59 |
| Residual | [ ] | [ ] | [ ] | 0.59 |
| KMeans | [ ] | [ ] | [ ] | 0.59 |
| Mamba | [ ] | [ ] | [ ] | 0.59 |
| Metacognition | [ ] | [ ] | [ ] | 0.59 |
| GDN | [ ] | [ ] | [ ] | 0.59 |
| Parallel | [ ] | [ ] | [ ] | 0.59 |
| Stack | [ ] | [ ] | [ ] | 0.59 |

**Checkpoint:** ⬜ ModeStepTweenAlt signed (Vulkan) — Line FamilyTweenAlt Split→Tween frozen gap; STA1/STAO + TweenAlt volumes; 130 groups in 738.413s.

---

## 14 — TweenSplitHeadProxy

Head backward/SGD uses full lr; its input gradient is projected to hidden leaves with 1/(N−1) sharing. Dense hidden updates use GradWOnly.

Wires: `GPB1`/`GPBO` (variant 1); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVVulkanProxyTrain` and `TestGoVVulkanProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.64 |
| RMSNorm | [ ] | [ ] | [ ] | 0.64 |
| LayerNorm | [ ] | [ ] | [ ] | 0.64 |
| SwiGLU | [ ] | [ ] | [ ] | 0.64 |
| Softmax | [ ] | [ ] | [ ] | 0.64 |
| Embedding | [ ] | [ ] | [ ] | 0.64 |
| MHA | [ ] | [ ] | [ ] | 0.64 |
| CNN1 | [ ] | [ ] | [ ] | 0.64 |
| CNN2 | [ ] | [ ] | [ ] | 0.64 |
| CNN3 | [ ] | [ ] | [ ] | 0.64 |
| ConvT1 | [ ] | [ ] | [ ] | 0.64 |
| ConvT2 | [ ] | [ ] | [ ] | 0.64 |
| ConvT3 | [ ] | [ ] | [ ] | 0.64 |
| RNN | [ ] | [ ] | [ ] | 0.64 |
| LSTM | [ ] | [ ] | [ ] | 0.64 |
| Sequential | [ ] | [ ] | [ ] | 0.64 |
| Residual | [ ] | [ ] | [ ] | 0.64 |
| KMeans | [ ] | [ ] | [ ] | 0.64 |
| Mamba | [ ] | [ ] | [ ] | 0.64 |
| Metacognition | [ ] | [ ] | [ ] | 0.64 |
| GDN | [ ] | [ ] | [ ] | 0.64 |
| Parallel | [ ] | [ ] | [ ] | 0.64 |
| Stack | [ ] | [ ] | [ ] | 0.64 |

**Checkpoint:** ⬜ `TestGoVVulkanProxyTrain` PASS (65 groups, ~548s @ BENCH_ITERS=1); `TestGoVVulkanProxyVolumeTrain` PASS (69 groups, ~728s @ BENCH_ITERS=1).

---

## 15 — TweenSplitLinear

Reverse linear-credit traversal updates each leaf at 1/N and propagates through the updated weights. Nested Stack, Residual and all Parallel combine routes retain their Go ordering.

Wires: `GPB1`/`GPBO` (variant 2); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVVulkanProxyTrain` and `TestGoVVulkanProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.65 |
| RMSNorm | [ ] | [ ] | [ ] | 0.65 |
| LayerNorm | [ ] | [ ] | [ ] | 0.65 |
| SwiGLU | [ ] | [ ] | [ ] | 0.65 |
| Softmax | [ ] | [ ] | [ ] | 0.65 |
| Embedding | [ ] | [ ] | [ ] | 0.65 |
| MHA | [ ] | [ ] | [ ] | 0.65 |
| CNN1 | [ ] | [ ] | [ ] | 0.65 |
| CNN2 | [ ] | [ ] | [ ] | 0.65 |
| CNN3 | [ ] | [ ] | [ ] | 0.65 |
| ConvT1 | [ ] | [ ] | [ ] | 0.65 |
| ConvT2 | [ ] | [ ] | [ ] | 0.65 |
| ConvT3 | [ ] | [ ] | [ ] | 0.65 |
| RNN | [ ] | [ ] | [ ] | 0.65 |
| LSTM | [ ] | [ ] | [ ] | 0.65 |
| Sequential | [ ] | [ ] | [ ] | 0.65 |
| Residual | [ ] | [ ] | [ ] | 0.65 |
| KMeans | [ ] | [ ] | [ ] | 0.65 |
| Mamba | [ ] | [ ] | [ ] | 0.65 |
| Metacognition | [ ] | [ ] | [ ] | 0.65 |
| GDN | [ ] | [ ] | [ ] | 0.65 |
| Parallel | [ ] | [ ] | [ ] | 0.65 |
| Stack | [ ] | [ ] | [ ] | 0.65 |

**Checkpoint:** ⬜ `TestGoVVulkanProxyTrain/TweenSplitLinear` PASS (65 groups, ~548s @ BENCH_ITERS=1); `TestGoVVulkanProxyVolumeTrain/TweenSplitLinear` PASS (69 groups, ~733s @ BENCH_ITERS=1).

---

## 16 — TweenSplitFastProxy

Compute a linear head proxy before updating the head, then share it across hidden leaves at 1/(N−1). Dense local updates use GradWOnly.

Wires: `GPB1`/`GPBO` (variant 3); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVVulkanProxyTrain` and `TestGoVVulkanProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.66 |
| RMSNorm | [ ] | [ ] | [ ] | 0.66 |
| LayerNorm | [ ] | [ ] | [ ] | 0.66 |
| SwiGLU | [ ] | [ ] | [ ] | 0.66 |
| Softmax | [ ] | [ ] | [ ] | 0.66 |
| Embedding | [ ] | [ ] | [ ] | 0.66 |
| MHA | [ ] | [ ] | [ ] | 0.66 |
| CNN1 | [ ] | [ ] | [ ] | 0.66 |
| CNN2 | [ ] | [ ] | [ ] | 0.66 |
| CNN3 | [ ] | [ ] | [ ] | 0.66 |
| ConvT1 | [ ] | [ ] | [ ] | 0.66 |
| ConvT2 | [ ] | [ ] | [ ] | 0.66 |
| ConvT3 | [ ] | [ ] | [ ] | 0.66 |
| RNN | [ ] | [ ] | [ ] | 0.66 |
| LSTM | [ ] | [ ] | [ ] | 0.66 |
| Sequential | [ ] | [ ] | [ ] | 0.66 |
| Residual | [ ] | [ ] | [ ] | 0.66 |
| KMeans | [ ] | [ ] | [ ] | 0.66 |
| Mamba | [ ] | [ ] | [ ] | 0.66 |
| Metacognition | [ ] | [ ] | [ ] | 0.66 |
| GDN | [ ] | [ ] | [ ] | 0.66 |
| Parallel | [ ] | [ ] | [ ] | 0.66 |
| Stack | [ ] | [ ] | [ ] | 0.66 |

**Checkpoint:** ⬜ `TestGoVVulkanProxyTrain/TweenSplitFastProxy` PASS (65 groups, ~546s @ BENCH_ITERS=1); `TestGoVVulkanProxyVolumeTrain/TweenSplitFastProxy` PASS (69 groups, ~729s @ BENCH_ITERS=1).

---

## 17 — TweenSplitLinearCache

Linear credit refreshes every 20 epochs. Intervening updates reuse cached projections rescaled by live/cached gap norms. The tests cross the refresh boundary with 22 epochs.

Wires: `GPB1`/`GPBO` (variant 4); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVVulkanProxyTrain` and `TestGoVVulkanProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.67 |
| RMSNorm | [ ] | [ ] | [ ] | 0.67 |
| LayerNorm | [ ] | [ ] | [ ] | 0.67 |
| SwiGLU | [ ] | [ ] | [ ] | 0.67 |
| Softmax | [ ] | [ ] | [ ] | 0.67 |
| Embedding | [ ] | [ ] | [ ] | 0.67 |
| MHA | [ ] | [ ] | [ ] | 0.67 |
| CNN1 | [ ] | [ ] | [ ] | 0.67 |
| CNN2 | [ ] | [ ] | [ ] | 0.67 |
| CNN3 | [ ] | [ ] | [ ] | 0.67 |
| ConvT1 | [ ] | [ ] | [ ] | 0.67 |
| ConvT2 | [ ] | [ ] | [ ] | 0.67 |
| ConvT3 | [ ] | [ ] | [ ] | 0.67 |
| RNN | [ ] | [ ] | [ ] | 0.67 |
| LSTM | [ ] | [ ] | [ ] | 0.67 |
| Sequential | [ ] | [ ] | [ ] | 0.67 |
| Residual | [ ] | [ ] | [ ] | 0.67 |
| KMeans | [ ] | [ ] | [ ] | 0.67 |
| Mamba | [ ] | [ ] | [ ] | 0.67 |
| Metacognition | [ ] | [ ] | [ ] | 0.67 |
| GDN | [ ] | [ ] | [ ] | 0.67 |
| Parallel | [ ] | [ ] | [ ] | 0.67 |
| Stack | [ ] | [ ] | [ ] | 0.67 |

**Checkpoint:** ⬜ `TestGoVVulkanProxyTrain/TweenSplitLinearCache` PASS (88 groups, ~563s @ BENCH_ITERS=1); `TestGoVVulkanProxyVolumeTrain/TweenSplitLinearCache` PASS (92 groups, ~747s @ BENCH_ITERS=1).

---

## 18 — TweenSplitHeadProxyAsync

Apply the previous proxy to hidden leaves, update the head, then capture the next proxy through updated head weights. The first epoch uses the fresh proxy; four-epoch tests exercise stale reuse.

Wires: `GPB1`/`GPBO` (variant 5); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVVulkanProxyTrain` and `TestGoVVulkanProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.68 |
| RMSNorm | [ ] | [ ] | [ ] | 0.68 |
| LayerNorm | [ ] | [ ] | [ ] | 0.68 |
| SwiGLU | [ ] | [ ] | [ ] | 0.68 |
| Softmax | [ ] | [ ] | [ ] | 0.68 |
| Embedding | [ ] | [ ] | [ ] | 0.68 |
| MHA | [ ] | [ ] | [ ] | 0.68 |
| CNN1 | [ ] | [ ] | [ ] | 0.68 |
| CNN2 | [ ] | [ ] | [ ] | 0.68 |
| CNN3 | [ ] | [ ] | [ ] | 0.68 |
| ConvT1 | [ ] | [ ] | [ ] | 0.68 |
| ConvT2 | [ ] | [ ] | [ ] | 0.68 |
| ConvT3 | [ ] | [ ] | [ ] | 0.68 |
| RNN | [ ] | [ ] | [ ] | 0.68 |
| LSTM | [ ] | [ ] | [ ] | 0.68 |
| Sequential | [ ] | [ ] | [ ] | 0.68 |
| Residual | [ ] | [ ] | [ ] | 0.68 |
| KMeans | [ ] | [ ] | [ ] | 0.68 |
| Mamba | [ ] | [ ] | [ ] | 0.68 |
| Metacognition | [ ] | [ ] | [ ] | 0.68 |
| GDN | [ ] | [ ] | [ ] | 0.68 |
| Parallel | [ ] | [ ] | [ ] | 0.68 |
| Stack | [ ] | [ ] | [ ] | 0.68 |

**Checkpoint:** Direct Go↔Vulkan parity and repeated replay passed for all 23 families (34 native dtypes where supported; GDN Float32). Leaf fixtures retain their declared precision gates. Proxy volume loss, captured tensors and native weights require industry-or-better (60–100%) agreement, with actual deviations reported; observed async rounding differences were strict (80%). Exact same-backend replay is checked separately. Numerical projection, credit, norms and SGD run in Vulkan; host code schedules work and stages tapes. Full graph residency and performance tuning remain open.

---

## 19 — TweenSplitSparse

Update the head and one rotating hidden leaf per epoch, using the pre-update linear head proxy. Four-epoch tests exercise rotation and state reset.

Wires: `GPB1`/`GPBO` (variant 6); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVVulkanProxyTrain` and `TestGoVVulkanProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.69 |
| RMSNorm | [ ] | [ ] | [ ] | 0.69 |
| LayerNorm | [ ] | [ ] | [ ] | 0.69 |
| SwiGLU | [ ] | [ ] | [ ] | 0.69 |
| Softmax | [ ] | [ ] | [ ] | 0.69 |
| Embedding | [ ] | [ ] | [ ] | 0.69 |
| MHA | [ ] | [ ] | [ ] | 0.69 |
| CNN1 | [ ] | [ ] | [ ] | 0.69 |
| CNN2 | [ ] | [ ] | [ ] | 0.69 |
| CNN3 | [ ] | [ ] | [ ] | 0.69 |
| ConvT1 | [ ] | [ ] | [ ] | 0.69 |
| ConvT2 | [ ] | [ ] | [ ] | 0.69 |
| ConvT3 | [ ] | [ ] | [ ] | 0.69 |
| RNN | [ ] | [ ] | [ ] | 0.69 |
| LSTM | [ ] | [ ] | [ ] | 0.69 |
| Sequential | [ ] | [ ] | [ ] | 0.69 |
| Residual | [ ] | [ ] | [ ] | 0.69 |
| KMeans | [ ] | [ ] | [ ] | 0.69 |
| Mamba | [ ] | [ ] | [ ] | 0.69 |
| Metacognition | [ ] | [ ] | [ ] | 0.69 |
| GDN | [ ] | [ ] | [ ] | 0.69 |
| Parallel | [ ] | [ ] | [ ] | 0.69 |
| Stack | [ ] | [ ] | [ ] | 0.69 |

**Checkpoint:** Direct Go↔Vulkan parity and repeated replay passed for all 23 families (34 native dtypes where supported; GDN Float32). Leaf fixtures retain their declared precision gates. Proxy volume loss, captured tensors and native weights require industry-or-better (60–100%) agreement, with actual deviations reported; observed sparse rounding differences were strict (80%). Exact same-backend replay is checked separately. Numerical projection, credit, norms and SGD run in Vulkan; host code schedules work and stages tapes. Full graph residency and performance tuning remain open.

---

## 20 — MeshTweenSplit

Mesh-clock Split applies full-lr 1/N local credit to each enabled cell using the final captured tick. The gap remains frozen and passes to earlier cells only when input lengths agree.

Wires: `MGS1`/`MGSO`. `TestGoVVulkanMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.70 |
| RMSNorm | [ ] | [ ] | [ ] | 0.70 |
| LayerNorm | [ ] | [ ] | [ ] | 0.70 |
| SwiGLU | [ ] | [ ] | [ ] | 0.70 |
| Softmax | [ ] | [ ] | [ ] | 0.70 |
| Embedding | [ ] | [ ] | [ ] | 0.70 |
| MHA | [ ] | [ ] | [ ] | 0.70 |
| CNN1 | [ ] | [ ] | [ ] | 0.70 |
| CNN2 | [ ] | [ ] | [ ] | 0.70 |
| CNN3 | [ ] | [ ] | [ ] | 0.70 |
| ConvT1 | [ ] | [ ] | [ ] | 0.70 |
| ConvT2 | [ ] | [ ] | [ ] | 0.70 |
| ConvT3 | [ ] | [ ] | [ ] | 0.70 |
| RNN | [ ] | [ ] | [ ] | 0.70 |
| LSTM | [ ] | [ ] | [ ] | 0.70 |
| Sequential | [ ] | [ ] | [ ] | 0.70 |
| Residual | [ ] | [ ] | [ ] | 0.70 |
| KMeans | [ ] | [ ] | [ ] | 0.70 |
| Mamba | [ ] | [ ] | [ ] | 0.70 |
| Metacognition | [ ] | [ ] | [ ] | 0.70 |
| GDN | [ ] | [ ] | [ ] | 0.70 |
| Parallel | [ ] | [ ] | [ ] | 0.70 |
| Stack | [ ] | [ ] | [ ] | 0.70 |

**Checkpoint:** Passed all 23 actual node kinds, supported native dtypes (GDN Float32), shared TweenSplit volumes and exact repeated replay. **154 groups** in ~1541s. Numerical work runs in Vulkan; graph residency and performance tuning remain open.

---

## 21 — MeshTweenAlt

Mesh-clock Alt applies Split then half-lr Tween to the final captured local tape with the frozen popped gap. Shared volume Alt separately refreshes prediction/gap between phases.

Wires: `MGA1`/`MGAO`. `TestGoVVulkanMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.71 |
| RMSNorm | [ ] | [ ] | [ ] | 0.71 |
| LayerNorm | [ ] | [ ] | [ ] | 0.71 |
| SwiGLU | [ ] | [ ] | [ ] | 0.71 |
| Softmax | [ ] | [ ] | [ ] | 0.71 |
| Embedding | [ ] | [ ] | [ ] | 0.71 |
| MHA | [ ] | [ ] | [ ] | 0.71 |
| CNN1 | [ ] | [ ] | [ ] | 0.71 |
| CNN2 | [ ] | [ ] | [ ] | 0.71 |
| CNN3 | [ ] | [ ] | [ ] | 0.71 |
| ConvT1 | [ ] | [ ] | [ ] | 0.71 |
| ConvT2 | [ ] | [ ] | [ ] | 0.71 |
| ConvT3 | [ ] | [ ] | [ ] | 0.71 |
| RNN | [ ] | [ ] | [ ] | 0.71 |
| LSTM | [ ] | [ ] | [ ] | 0.71 |
| Sequential | [ ] | [ ] | [ ] | 0.71 |
| Residual | [ ] | [ ] | [ ] | 0.71 |
| KMeans | [ ] | [ ] | [ ] | 0.71 |
| Mamba | [ ] | [ ] | [ ] | 0.71 |
| Metacognition | [ ] | [ ] | [ ] | 0.71 |
| GDN | [ ] | [ ] | [ ] | 0.71 |
| Parallel | [ ] | [ ] | [ ] | 0.71 |
| Stack | [ ] | [ ] | [ ] | 0.71 |

**Checkpoint:** Passed all 23 actual node kinds, supported native dtypes (GDN Float32), shared TweenAlt volumes and exact repeated replay. **154 groups** in ~1761s. Numerical work runs in Vulkan; graph residency and performance tuning remain open.

---

## 22 — MeshTweenSplitFastProxy

Mesh-clock FastProxy applies per-cell pre-update head proxy credit on the final captured tick. Shared volume FastProxy uses the global collected volume tape.

Wires: `MGF1`/`MGFO`. `TestGoVVulkanMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.72 |
| RMSNorm | [ ] | [ ] | [ ] | 0.72 |
| LayerNorm | [ ] | [ ] | [ ] | 0.72 |
| SwiGLU | [ ] | [ ] | [ ] | 0.72 |
| Softmax | [ ] | [ ] | [ ] | 0.72 |
| Embedding | [ ] | [ ] | [ ] | 0.72 |
| MHA | [ ] | [ ] | [ ] | 0.72 |
| CNN1 | [ ] | [ ] | [ ] | 0.72 |
| CNN2 | [ ] | [ ] | [ ] | 0.72 |
| CNN3 | [ ] | [ ] | [ ] | 0.72 |
| ConvT1 | [ ] | [ ] | [ ] | 0.72 |
| ConvT2 | [ ] | [ ] | [ ] | 0.72 |
| ConvT3 | [ ] | [ ] | [ ] | 0.72 |
| RNN | [ ] | [ ] | [ ] | 0.72 |
| LSTM | [ ] | [ ] | [ ] | 0.72 |
| Sequential | [ ] | [ ] | [ ] | 0.72 |
| Residual | [ ] | [ ] | [ ] | 0.72 |
| KMeans | [ ] | [ ] | [ ] | 0.72 |
| Mamba | [ ] | [ ] | [ ] | 0.72 |
| Metacognition | [ ] | [ ] | [ ] | 0.72 |
| GDN | [ ] | [ ] | [ ] | 0.72 |
| Parallel | [ ] | [ ] | [ ] | 0.72 |
| Stack | [ ] | [ ] | [ ] | 0.72 |

**Checkpoint:** Passed all 23 actual node kinds, supported native dtypes (GDN Float32), shared FastProxy volumes and exact repeated replay. **172 groups** in ~2119s. Numerical work runs in Vulkan; graph residency and performance tuning remain open.

---

## 23 — MeshTweenSplitSparse

Mesh-clock Sparse updates the head and one hidden leaf per eligible local unit. Stack/Parallel nodes own rotation state; shared volume Sparse owns grid state.

Wires: `MGQ1`/`MGQO`. `TestGoVVulkanMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.73 |
| RMSNorm | [ ] | [ ] | [ ] | 0.73 |
| LayerNorm | [ ] | [ ] | [ ] | 0.73 |
| SwiGLU | [ ] | [ ] | [ ] | 0.73 |
| Softmax | [ ] | [ ] | [ ] | 0.73 |
| Embedding | [ ] | [ ] | [ ] | 0.73 |
| MHA | [ ] | [ ] | [ ] | 0.73 |
| CNN1 | [ ] | [ ] | [ ] | 0.73 |
| CNN2 | [ ] | [ ] | [ ] | 0.73 |
| CNN3 | [ ] | [ ] | [ ] | 0.73 |
| ConvT1 | [ ] | [ ] | [ ] | 0.73 |
| ConvT2 | [ ] | [ ] | [ ] | 0.73 |
| ConvT3 | [ ] | [ ] | [ ] | 0.73 |
| RNN | [ ] | [ ] | [ ] | 0.73 |
| LSTM | [ ] | [ ] | [ ] | 0.73 |
| Sequential | [ ] | [ ] | [ ] | 0.73 |
| Residual | [ ] | [ ] | [ ] | 0.73 |
| KMeans | [ ] | [ ] | [ ] | 0.73 |
| Mamba | [ ] | [ ] | [ ] | 0.73 |
| Metacognition | [ ] | [ ] | [ ] | 0.73 |
| GDN | [ ] | [ ] | [ ] | 0.73 |
| Parallel | [ ] | [ ] | [ ] | 0.73 |
| Stack | [ ] | [ ] | [ ] | 0.73 |

**Checkpoint:** Passed all 23 actual node kinds, supported native dtypes (GDN Float32), shared Sparse volumes and exact repeated replay. **172 groups** in ~2107s. Numerical work runs in Vulkan; graph residency and performance tuning remain open.

---

## 24 — StepTweenSplitHeadProxy

Line HeadProxy applies full-lr head backward/SGD on pop, then projects head input credit across hidden leaves. The popped global gap stays frozen. Shared volume coverage uses the section 14 HeadProxy path.

Wire: `STH1`/`STHO`. `TestGoVVulkanProxyTrain/StepTweenSplitHeadProxy` covers 65 leaf/Line groups, including native CNN2/3 and ConvT2/3. Section 14 supplies its shared volume evidence. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.74 |
| RMSNorm | [ ] | [ ] | [ ] | 0.74 |
| LayerNorm | [ ] | [ ] | [ ] | 0.74 |
| SwiGLU | [ ] | [ ] | [ ] | 0.74 |
| Softmax | [ ] | [ ] | [ ] | 0.74 |
| Embedding | [ ] | [ ] | [ ] | 0.74 |
| MHA | [ ] | [ ] | [ ] | 0.74 |
| CNN1 | [ ] | [ ] | [ ] | 0.74 |
| CNN2 | [ ] | [ ] | [ ] | 0.74 |
| CNN3 | [ ] | [ ] | [ ] | 0.74 |
| ConvT1 | [ ] | [ ] | [ ] | 0.74 |
| ConvT2 | [ ] | [ ] | [ ] | 0.74 |
| ConvT3 | [ ] | [ ] | [ ] | 0.74 |
| RNN | [ ] | [ ] | [ ] | 0.74 |
| LSTM | [ ] | [ ] | [ ] | 0.74 |
| Sequential | [ ] | [ ] | [ ] | 0.74 |
| Residual | [ ] | [ ] | [ ] | 0.74 |
| KMeans | [ ] | [ ] | [ ] | 0.74 |
| Mamba | [ ] | [ ] | [ ] | 0.74 |
| Metacognition | [ ] | [ ] | [ ] | 0.74 |
| GDN | [ ] | [ ] | [ ] | 0.74 |
| Parallel | [ ] | [ ] | [ ] | 0.74 |
| Stack | [ ] | [ ] | [ ] | 0.74 |


**Checkpoint:** ⬜ `TestGoVVulkanProxyTrain/StepTweenSplitHeadProxy` PASS (65 groups, ~561s @ BENCH_ITERS=1). Shared volume evidence: section 14 (`TestGoVVulkanProxyVolumeTrain` HeadProxy). Industry spectrum for Line proxy W/loss (incl. HeadProxy variant 1).

---

## 25 — StepTweenSplitLinear

Line per-hop credit on pop uses the shared Vulkan `systems::tween_split::apply_variant_child` implementation (variant 2). The popped gap stays frozen. Reverse local linear credit uses updated weights and 1/N sharing.

Wire: `STL1`/`STLO`. Tests: `TestGoVVulkanStepProxyTrain/StepTweenSplitLinear`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 15 uses `PGB1`/`PGVO` variant 2 and `TestGoVVulkanProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/linux_build_w2a.sh --training-25-29-all`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.75 |
| RMSNorm | [ ] | [ ] | [ ] | 0.75 |
| LayerNorm | [ ] | [ ] | [ ] | 0.75 |
| SwiGLU | [ ] | [ ] | [ ] | 0.75 |
| Softmax | [ ] | [ ] | [ ] | 0.75 |
| Embedding | [ ] | [ ] | [ ] | 0.75 |
| MHA | [ ] | [ ] | [ ] | 0.75 |
| CNN1 | [ ] | [ ] | [ ] | 0.75 |
| CNN2 | [ ] | [ ] | [ ] | 0.75 |
| CNN3 | [ ] | [ ] | [ ] | 0.75 |
| ConvT1 | [ ] | [ ] | [ ] | 0.75 |
| ConvT2 | [ ] | [ ] | [ ] | 0.75 |
| ConvT3 | [ ] | [ ] | [ ] | 0.75 |
| RNN | [ ] | [ ] | [ ] | 0.75 |
| LSTM | [ ] | [ ] | [ ] | 0.75 |
| Sequential | [ ] | [ ] | [ ] | 0.75 |
| Residual | [ ] | [ ] | [ ] | 0.75 |
| KMeans | [ ] | [ ] | [ ] | 0.75 |
| Mamba | [ ] | [ ] | [ ] | 0.75 |
| Metacognition | [ ] | [ ] | [ ] | 0.75 |
| GDN | [ ] | [ ] | [ ] | 0.75 |
| Parallel | [ ] | [ ] | [ ] | 0.75 |
| Stack | [ ] | [ ] | [ ] | 0.75 |


**Checkpoint:** ⬜ `TestGoVVulkanStepProxyTrain/StepTweenSplitLinear` PASS (69 groups, ~584s @ BENCH_ITERS=1). Shared volume evidence: section 15 (`PGB1`/`PGVO` variant 2).

---

## 26 — StepTweenSplitFastProxy

Line per-hop credit on pop uses the shared Vulkan `systems::tween_split::apply_variant_child` implementation (variant 3). The popped gap stays frozen. The local head proxy is captured before the head update, then shared across hidden leaves.

Wire: `STF1`/`STFO`. Tests: `TestGoVVulkanStepProxyTrain/StepTweenSplitFastProxy`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 16 uses `PGB1`/`PGVO` variant 3 and `TestGoVVulkanProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/linux_build_w2a.sh --training-25-29-all`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.76 |
| RMSNorm | [ ] | [ ] | [ ] | 0.76 |
| LayerNorm | [ ] | [ ] | [ ] | 0.76 |
| SwiGLU | [ ] | [ ] | [ ] | 0.76 |
| Softmax | [ ] | [ ] | [ ] | 0.76 |
| Embedding | [ ] | [ ] | [ ] | 0.76 |
| MHA | [ ] | [ ] | [ ] | 0.76 |
| CNN1 | [ ] | [ ] | [ ] | 0.76 |
| CNN2 | [ ] | [ ] | [ ] | 0.76 |
| CNN3 | [ ] | [ ] | [ ] | 0.76 |
| ConvT1 | [ ] | [ ] | [ ] | 0.76 |
| ConvT2 | [ ] | [ ] | [ ] | 0.76 |
| ConvT3 | [ ] | [ ] | [ ] | 0.76 |
| RNN | [ ] | [ ] | [ ] | 0.76 |
| LSTM | [ ] | [ ] | [ ] | 0.76 |
| Sequential | [ ] | [ ] | [ ] | 0.76 |
| Residual | [ ] | [ ] | [ ] | 0.76 |
| KMeans | [ ] | [ ] | [ ] | 0.76 |
| Mamba | [ ] | [ ] | [ ] | 0.76 |
| Metacognition | [ ] | [ ] | [ ] | 0.76 |
| GDN | [ ] | [ ] | [ ] | 0.76 |
| Parallel | [ ] | [ ] | [ ] | 0.76 |
| Stack | [ ] | [ ] | [ ] | 0.76 |


**Checkpoint:** ⬜ `TestGoVVulkanStepProxyTrain/StepTweenSplitFastProxy` PASS (69 groups, ~568s @ BENCH_ITERS=1). Shared volume evidence: section 16 (`PGB1`/`PGVO` variant 3).

---

## 27 — StepTweenSplitLinearCache

Line per-hop credit on pop uses the shared Vulkan `systems::tween_split::apply_variant_child` implementation (variant 4). The popped gap stays frozen. Eligible Parallel hops retain cached projections and GPU-produced norms between pops. Dense hops have no accelerator state. Seven 26-tick fixtures cross the 20-pop refresh boundary for every Parallel combine.

Wire: `SLC1`/`SLCO`. Tests: `TestGoVVulkanStepProxyTrain/StepTweenSplitLinearCache`; 76 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 17 uses `PGB1`/`PGVO` variant 4 and `TestGoVVulkanProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/linux_build_w2a.sh --training-25-29-all`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.77 |
| RMSNorm | [ ] | [ ] | [ ] | 0.77 |
| LayerNorm | [ ] | [ ] | [ ] | 0.77 |
| SwiGLU | [ ] | [ ] | [ ] | 0.77 |
| Softmax | [ ] | [ ] | [ ] | 0.77 |
| Embedding | [ ] | [ ] | [ ] | 0.77 |
| MHA | [ ] | [ ] | [ ] | 0.77 |
| CNN1 | [ ] | [ ] | [ ] | 0.77 |
| CNN2 | [ ] | [ ] | [ ] | 0.77 |
| CNN3 | [ ] | [ ] | [ ] | 0.77 |
| ConvT1 | [ ] | [ ] | [ ] | 0.77 |
| ConvT2 | [ ] | [ ] | [ ] | 0.77 |
| ConvT3 | [ ] | [ ] | [ ] | 0.77 |
| RNN | [ ] | [ ] | [ ] | 0.77 |
| LSTM | [ ] | [ ] | [ ] | 0.77 |
| Sequential | [ ] | [ ] | [ ] | 0.77 |
| Residual | [ ] | [ ] | [ ] | 0.77 |
| KMeans | [ ] | [ ] | [ ] | 0.77 |
| Mamba | [ ] | [ ] | [ ] | 0.77 |
| Metacognition | [ ] | [ ] | [ ] | 0.77 |
| GDN | [ ] | [ ] | [ ] | 0.77 |
| Parallel | [ ] | [ ] | [ ] | 0.77 |
| Stack | [ ] | [ ] | [ ] | 0.77 |


**Checkpoint:** ⬜ `TestGoVVulkanStepProxyTrain/StepTweenSplitLinearCache` PASS (76 groups, ~576s @ BENCH_ITERS=1). Shared volume evidence: section 17 (`PGB1`/`PGVO` variant 4).

---

## 28 — StepTweenSplitHeadProxyAsync

Line per-hop credit on pop uses the shared Vulkan `systems::tween_split::apply_variant_child` implementation (variant 5). The popped gap stays frozen. Eligible Parallel hops retain their previous hidden proxy between pops and capture the next proxy after updating the head. Dense hops have no accelerator state.

Wire: `SHA1`/`SHAO`. Tests: `TestGoVVulkanStepProxyTrain/StepTweenSplitHeadProxyAsync`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 18 uses `PGB1`/`PGVO` variant 5 and `TestGoVVulkanProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/linux_build_w2a.sh --training-25-29-all`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.78 |
| RMSNorm | [ ] | [ ] | [ ] | 0.78 |
| LayerNorm | [ ] | [ ] | [ ] | 0.78 |
| SwiGLU | [ ] | [ ] | [ ] | 0.78 |
| Softmax | [ ] | [ ] | [ ] | 0.78 |
| Embedding | [ ] | [ ] | [ ] | 0.78 |
| MHA | [ ] | [ ] | [ ] | 0.78 |
| CNN1 | [ ] | [ ] | [ ] | 0.78 |
| CNN2 | [ ] | [ ] | [ ] | 0.78 |
| CNN3 | [ ] | [ ] | [ ] | 0.78 |
| ConvT1 | [ ] | [ ] | [ ] | 0.78 |
| ConvT2 | [ ] | [ ] | [ ] | 0.78 |
| ConvT3 | [ ] | [ ] | [ ] | 0.78 |
| RNN | [ ] | [ ] | [ ] | 0.78 |
| LSTM | [ ] | [ ] | [ ] | 0.78 |
| Sequential | [ ] | [ ] | [ ] | 0.78 |
| Residual | [ ] | [ ] | [ ] | 0.78 |
| KMeans | [ ] | [ ] | [ ] | 0.78 |
| Mamba | [ ] | [ ] | [ ] | 0.78 |
| Metacognition | [ ] | [ ] | [ ] | 0.78 |
| GDN | [ ] | [ ] | [ ] | 0.78 |
| Parallel | [ ] | [ ] | [ ] | 0.78 |
| Stack | [ ] | [ ] | [ ] | 0.78 |


**Checkpoint:** ⬜ `TestGoVVulkanStepProxyTrain/StepTweenSplitHeadProxyAsync` PASS (69 groups, ~567s @ BENCH_ITERS=1). Shared volume evidence: section 18 (`PGB1`/`PGVO` variant 5).

---

## 29 — StepTweenSplitSparse

Line per-hop credit on pop uses the shared Vulkan `systems::tween_split::apply_variant_child` implementation (variant 6). The popped gap stays frozen. Eligible Parallel hops rotate the selected hidden leaf between pops. Dense hops have no accelerator state; the no-state hidden selection follows Go.

Wire: `SSP1`/`SSPO`. Tests: `TestGoVVulkanStepProxyTrain/StepTweenSplitSparse`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 19 uses `PGB1`/`PGVO` variant 6 and `TestGoVVulkanProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/linux_build_w2a.sh --training-25-29-all`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | 0.79 |
| RMSNorm | [ ] | [ ] | [ ] | 0.79 |
| LayerNorm | [ ] | [ ] | [ ] | 0.79 |
| SwiGLU | [ ] | [ ] | [ ] | 0.79 |
| Softmax | [ ] | [ ] | [ ] | 0.79 |
| Embedding | [ ] | [ ] | [ ] | 0.79 |
| MHA | [ ] | [ ] | [ ] | 0.79 |
| CNN1 | [ ] | [ ] | [ ] | 0.79 |
| CNN2 | [ ] | [ ] | [ ] | 0.79 |
| CNN3 | [ ] | [ ] | [ ] | 0.79 |
| ConvT1 | [ ] | [ ] | [ ] | 0.79 |
| ConvT2 | [ ] | [ ] | [ ] | 0.79 |
| ConvT3 | [ ] | [ ] | [ ] | 0.79 |
| RNN | [ ] | [ ] | [ ] | 0.79 |
| LSTM | [ ] | [ ] | [ ] | 0.79 |
| Sequential | [ ] | [ ] | [ ] | 0.79 |
| Residual | [ ] | [ ] | [ ] | 0.79 |
| KMeans | [ ] | [ ] | [ ] | 0.79 |
| Mamba | [ ] | [ ] | [ ] | 0.79 |
| Metacognition | [ ] | [ ] | [ ] | 0.79 |
| GDN | [ ] | [ ] | [ ] | 0.79 |
| Parallel | [ ] | [ ] | [ ] | 0.79 |
| Stack | [ ] | [ ] | [ ] | 0.79 |


**Checkpoint:** ⬜ `TestGoVVulkanStepProxyTrain/StepTweenSplitSparse` PASS (69 groups, ~567s @ BENCH_ITERS=1). Shared volume evidence: section 19 (`PGB1`/`PGVO` variant 6).

---


---

## Version log (Vulkan)

| Tag | Meaning |
|-----|---------|
| **0.00** (M0) | Vulkan host scaffold (device / buffers / smoke saxpy / C ABI / gaps). Engine board **0 / 100**. Training checklist mirrored from welvet.cuda — **all rows unchecked**. Climb toward cuda **0.79**. |

Prior cuda tags (0.13…0.79) are **not** inherited. Re-earn each gate on Vulkan evidence.
