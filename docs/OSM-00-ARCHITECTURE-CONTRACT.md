# OSM-00 — ORBI StreamMoE architecture contract

Status: **BOOTSTRAP**

Reference audit: `ingeniusvictor/orbi-swiftlet-lab`  
Frozen Swiftlet baseline: `909c04213c9deb369dac0679d0872512cf3ab32e`

## Product boundary

ORBI StreamMoE is a cross-platform inference runtime. It is **not** a renamed Qwen model and does not vendor model weights.

Initial model compatibility target:

`Qwen3-Next-80B-A3B` with 4-bit routed experts and streamed expert residency.

## Non-negotiable design rules

1. Portable core is C++20.
2. Windows and Android share the same model/cache/backend contracts.
3. qpack v1 compatibility is implemented before inventing an ORBI-specific container.
4. Correctness precedes GPU optimization.
5. CPU reference execution is the semantic oracle for backend kernels.
6. Expert placement may change performance, never model semantics.
7. Storage and compute are independent abstractions.
8. Vulkan is the first shared accelerator backend.
9. Direct-I/O / no-buffering modes are benchmark options, not correctness dependencies.
10. Peak RAM claims are measured per platform; upstream Apple numbers are not copied as ORBI guarantees.

## Layering

```text
apps / integrations
  ├─ CLI
  ├─ local server
  ├─ L.U.M.I.A. provider
  └─ Android / Edge Mesh bridge
            │
            ▼
session / generation
            │
            ▼
Qwen model graph
  ├─ gated GQA
  ├─ Gated DeltaNet
  ├─ router top-k
  ├─ sparse routed MoE
  └─ shared expert
            │
     ┌──────┴───────┐
     ▼              ▼
expert cache     compute backend
     │              ├─ CPU reference
     ▼              └─ Vulkan
storage backend
  ├─ Windows
  └─ POSIX/Android
            │
            ▼
qpack-compatible container
```

## Compatibility target facts

For Qwen3-Next-80B-A3B:

- hidden size: 2048
- decoder layers: 48
- full-attention interval: 4
- routed experts/layer: 512
- selected routed experts/token/layer: 10
- MoE intermediate size: 512
- full attention layers: 12
- Gated DeltaNet layers: 36

These values are a model compatibility contract, not performance assumptions.

## First implementation sequence

1. Buildable C++20 skeleton.
2. Architecture/config structs with unit tests.
3. qpack manifest/layout parser and validation.
4. Local expert blob reader.
5. Platform-neutral bounded LFU+recency cache.
6. CPU reference primitives.
7. Windows Vulkan device/bootstrap.
8. Quantized GEMV + norm + router.
9. Sparse MoE execution.
10. GQA + Gated DeltaNet.
11. Qwen3.6-35B-A3B pilot.
12. Qwen3-Next-80B-A3B Windows pilot.
13. Android NDK/Vulkan port.
14. L.U.M.I.A. + ORBI Edge Mesh integration.

## Gate definition

OSM-00 closes when the repository builds a portable smoke-test target and the architecture constants are encoded without any platform SDK dependency.
