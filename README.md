# welvet.vulkan — Welvet (3D volumetric AI engine) on Vulkan

**welvet.vulkan is a full rebuild of Welvet — the 3D volumetric AI engine — in C++ / Vulkan compute.** Same engine shape as Go Welvet: numerical types, layers, volumetric architecture (cells in a 3D grid), forward/backward, training modes, systems, model loading, inference, checkpoints. Execution is Vulkan compute + resident device buffers. Host orchestration is C++ (Vulkan API at the device boundary).

A `1 × 1 × 1` volume is the ordinary single-cell neural-network case (layers inside that cell). Larger volumes connect cells that each hold their own layer stacks. This Vulkan rebuild must own and execute that same topology — not “a Dense GEMV demo with a GPU sticker.”

It will run directly from C++ and through a stable C ABI. Go Welvet can later delegate via **NAC** (Native Acceleration Compute). The Vulkan engine must also build, load models, execute volumes, and train **without** a Go process.

| | |
|--|--|
| **Version** | **`0.79`** |
| **Engine board** | **79 / 100** (same 10 categories as Go Welvet; train ladder accepted) |
| **Compute** | Vulkan GPU (cross-vendor) — primary · CPU / CUDA / Metal / QNN siblings remain separate repos — never implied |
| **Parity reference** | [welvet](https://github.com/openfluke/welvet) Go **v1.6.0** (100/100) |
| **Go binding** | `nativevulkan` · `native/vulkan/linux_x86_64/libwelvet_vulkan.so` (model ABI **1**) |
| **Validation** | [w2a](https://github.com/openfluke/w2a) `go_v_vulkan` |

**This README is the runway.** Every unchecked cell is planned Vulkan work. Tick ⬜ → 🚧 → ✅ here as gates land. **Training port detail** (per-mode × layer evidence, wire names, version log) lives in [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md). [GAPS.md](GAPS.md) is the machine-oriented census mirror; [VERSIONING.md](VERSIONING.md) is the `0.x` formula. Pure-Vulkan contract: [docs/VULKAN_ONLY.md](docs/VULKAN_ONLY.md).

### What exists today (M0)

```bash
./scripts/build_linux.sh
./build/example_vulkan_smoke
./build/welvet_dense_fwd --version   # 0.79
./build/welvet_dense_fwd --identity
./build/welvet_dense_fwd --gaps
```

| Slice | Status |
|---|:---:|
| Linux x86_64 CMake → `libwelvet_vulkan.so` | ✅ |
| Vulkan device / host-visible buffers / smoke saxpy | ✅ |
| C ABI version / init / available / gaps | ✅ |
| Dense FormatNone×34 leaf forward | ⬜ |
| NormalBP + 29 train modes (cuda 0.79 runway) | ⬜ |
| CamSync / Freeze/Shadow / systems / model | ⬜ |

---

## Relationship

| Repository | Responsibility |
|---|---|
| [welvet](https://github.com/openfluke/welvet) | Go engine — volumetric architecture, layers, dtypes, quants, train modes, ENTITY |
| [welvet.cpp](https://github.com/openfluke/welvet.cpp) | Portable C++ CPU engine + C ABI (port **semantics**, not CPU kernels) |
| [welvet.metal](https://github.com/openfluke/welvet.metal) | Apple Metal sibling (separate tree; score not inherited) |
| [welvet.cuda](https://github.com/openfluke/welvet.cuda) | NVIDIA CUDA sibling — **current climb target (0.79)** |
| **welvet.vulkan** | **This repo** — full Welvet engine on Vulkan |
| [welvet.qnn](https://github.com/openfluke/welvet.qnn) | Qualcomm Hexagon HTP sibling pattern |
| [w2a](https://github.com/openfluke/w2a) | Parity / speed / determinism (`Go_v_Vulkan` planned) |

### Architecture sketch — intended flow

```text
C++ application / CLI                 Go Welvet or w2a / future NAC
          |                                       |
     C++ engine API                          stable C ABI
          |                                       |
          +------- welvet.vulkan engine ----------+
                              |
          model / ENTITY / tokenizer / transformer / sampling
                              |
             3D architecture: cells, stacks, branches, links
                              |
          forward / backward / training / Step / Mesh / systems
                              |
                VkDevice + SPIR-V pipelines + queues + capability queries
                              |
             resident device buffers (weights + tensors + training state)
                              |
                    GPU via Vulkan (cross-vendor)
                    other stacks = sibling repos (never implied)
```

## Non-negotiable rules

1. **Vulkan owns numerical work** for supported paths — no silent host fallback.
2. **Sibling scores do not transfer.** cuda/metal/cpp `[x]` is not Vulkan credit.
3. **One TrainMode at a time** — see [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).
4. **Evidence lives in w2a** once `Go_v_Vulkan` lands.

## Implementation sequence (toward cuda 0.79)

1. **M0** — device / buffers / smoke / C ABI / gaps ← **you are here (`0.00`)**
2. **M1** — Dense FormatNone×34 forward + `welvet_dense_fwd` wires
3. Leaf FormatNone forwards ×23 + seqmix
4. NormalBP Mode 1 ×23
5. StepBP → MeshBP → ModeTween* → proxy/sparse modes (cuda checklist order)
6. Freeze/Shadow / CamSync / systems / model

## Source layout

```text
welvet.vulkan/
├── CMakeLists.txt
├── shaders/welvet_smoke.comp
├── include/welvet_c.h
├── include/welvet/{status,gaps}.hpp
├── include/welvet/vulkan/{device,buffer,kernels}.hpp
├── src/vulkan/{device,buffer,kernels}.cpp
├── src/gaps/registry.cpp
├── src/interop/c_abi.cpp
├── examples/vulkan_smoke/
├── tools/dense_fwd/
├── scripts/build_linux.sh
└── docs/{VULKAN_ONLY,TRAINING_CHECKLIST}.md
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
