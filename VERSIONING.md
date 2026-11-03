# welvet.vulkan versioning

**One scoreboard:** same ten categories / 100 points as Go Welvet.  
**Formula:** `version = 0.{round(earned)}` until **100 → 1.0.0**.

| Tag | Meaning |
|-----|---------|
| **0.00** | Bedrock — Vulkan host scaffold; **0 / 100** on the engine board |
| **0.x** | `round(earned)` from the engine table in [README.md](README.md) — climb toward [welvet.cuda](https://github.com/openfluke/welvet.cuda) **0.79** via [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md) |
| **1.0.0** | Engine board **100/100** — full Go parity on Vulkan (FormatNone path + Model/IO + Systems + CamSync + peak fused, or explicit gate rewrite) |

| In (moves `0.x`) | Out |
|------------------|-----|
| Layers, FormatNone×34, quants, train modes, Grid/Line/Mesh, Tween*, systems, model/IO — **on Vulkan** | Other vendors (CUDA/Metal/QNN) — sibling repos |
| Evidence via w2a `Go_v_Vulkan` (planned) | welvet.cpp / welvet.cuda / welvet.metal / Go scores (not inherited) |
| Determinism + speed on named Vulkan device | Packaging polish alone |

**Sibling map:** welvet (Go ref) · welvet.cpp (CPU) · welvet.metal (Apple GPU) · welvet.cuda (NVIDIA) · **welvet.vulkan** (this) · welvet.qnn · w2a.

Runway target for the next climb: match **welvet.cuda 0.79** mode evidence (NormalBP → Step/Mesh → Tween* → proxy/sparse), one TrainMode at a time per [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).
