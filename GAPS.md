# welvet.vulkan gap census

**Authoritative checklist:** [README.md](README.md).  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.00`** — engine board **0 / 100**. Dump: `welvet_dense_fwd --gaps`.

```bash
./scripts/build_linux.sh
./build/example_vulkan_smoke
./build/welvet_dense_fwd --gaps
```

## Done

| Slice | Detail |
|---|---|
| M0 Vulkan host | device, buffers, smoke saxpy, C ABI, Linux |

## Open

Everything else on the CUDA runway (Dense FormatNone×34 → NormalBP → Step/Mesh →
Tween* → proxy/sparse → CamSync / systems / model). See README and TRAINING_CHECKLIST.

Live counters: `./build/welvet_dense_fwd --gaps`.
