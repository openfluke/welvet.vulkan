# welvet.vulkan versioning

**One scoreboard:** same ten categories / 100 points as Go Welvet.  
**Formula:** `version = 0.{round(earned)}` until **100 → 1.0.0**.

| Tag | Meaning |
|-----|---------|
| **0.00** | Bedrock — Vulkan host scaffold; **0 / 100** on the engine board |
| **0.79** | Train ladder accepted + Go `nativevulkan` pin (Welvet **v1.6.0**); model ABI **1**; Linux x86_64 |
| **0.x** | `round(earned)` from the engine table in [README.md](README.md) |
| **1.0.0** | Engine board **100/100** — full Go parity on Vulkan (FormatNone path + Model/IO + Systems + CamSync + peak fused, or explicit gate rewrite) |

| In (moves `0.x`) | Out |
|------------------|-----|
| Layers, FormatNone×34, quants, train modes, Grid/Line/Mesh, Tween*, systems, model/IO — **on Vulkan** | Other vendors (CUDA/Metal/QNN) — sibling repos |
| Evidence via w2a `go_v_vulkan` | welvet.cpp / welvet.cuda / welvet.metal / Go scores (not inherited) |
| Determinism + speed on named Vulkan device | Packaging polish alone |

**Sibling map:** welvet (Go ref) · welvet.cpp (CPU) · welvet.metal (Apple GPU) · welvet.cuda (NVIDIA) · **welvet.vulkan** (this) · welvet.qnn · w2a.

**Current pin:** **0.79** — install into Go via `./scripts/linux_install_go.sh` → `welvet/native/vulkan/linux_x86_64/libwelvet_vulkan.so`; Android arm64 + x86_64 via `./scripts/android_install_go.sh` → `native/vulkan/android_{arm64,x86_64}/`.
