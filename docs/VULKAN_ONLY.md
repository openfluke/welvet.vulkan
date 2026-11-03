# Vulkan execution contract

`welvet.vulkan` is a standalone native cross-vendor GPU backend (Vulkan compute).
`welvet.cpp` is the separate CPU backend; `welvet.cuda` / `welvet.metal` are
separate vendor GPU backends. Each ships through its own native artifact and C
ABI. There is no dependency from one backend to the other, and **no host
numerical fallback** labeled as Vulkan.

## Supported computation

**M0** (host scaffolding — not engine-board credit alone):

- Vulkan instance + physical/logical device probe + runtime identity string
- Host-visible device buffer alloc / upload / download
- Compiled smoke compute shader `welvet_smoke_saxpy` (SPIR-V via `glslc`)
- C ABI: version / init / available / gaps

**M1+** (engine credit — leaf FormatNone forwards and beyond):

- FormatNone×34 decode → resident MatVec / layer kernels on Vulkan
- Dense + remaining layers + train modes — tick in
  [TRAINING_CHECKLIST.md](TRAINING_CHECKLIST.md) as suites land

Everything else (NormalBP, volume, train modes, …) is gap until ticked in
[TRAINING_CHECKLIST.md](TRAINING_CHECKLIST.md) and the [README runway](../README.md).

**Leaf dispatch note:** Softmax, SwiGLU SiLU⊙, and MHA attention forward/backward run on
Vulkan compute shaders via `src/vulkan/launches.cpp` (not host libm). Some other layer
implementations still evaluate numerics on the host today — notably **LSTM**, **RNN**,
**Mamba**, and **GDN** under `src/layers/` — until dedicated compute kernels land.

## Non-negotiable

- Supported numerical forward/backward/training paths run on **Vulkan**.
- Host code may schedule, validate, allocate and stage storage.
- Unsupported dtype/layer/mode → hard error (no silent CPU math).
- FormatNone + DType (or packed quant) is storage truth — not an undeclared f32 master.
- Parity vs Go is best-effort **60–100%** on w2a's spectrum; report the band.

## Build and delivery

```sh
./scripts/build_linux.sh
./build/example_vulkan_smoke
./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --identity
./build/welvet_dense_fwd --gaps
```

Requires CMake, a C++20 compiler, Vulkan loader/headers, and `glslc`
(LunarG Vulkan SDK recommended). Optional: `WELVET_VULKAN_VALIDATION=1` enables
the Khronos validation layer at init.
