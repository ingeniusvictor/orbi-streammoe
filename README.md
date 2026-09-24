# ORBI StreamMoE

Cross-platform local inference runtime research project for sparse Mixture-of-Experts models, initially targeting the Qwen3-Next / Qwen3.5/3.6 MoE family on **Windows** and **Android**.

The project goal is to reproduce the key memory advantage of expert streaming: keep the dense model core resident while loading only routed experts from local storage on demand.

## Initial target

First compatibility target:

`Qwen3-Next-80B-A3B` with 4-bit expert weights, using a streamed expert working set rather than full model residency.

The model remains Qwen; ORBI StreamMoE is the runtime, container compatibility layer, cache/I/O system and platform backend.

## Planned architecture

- C++20 portable core
- qpack-compatible reader first
- bounded LFU + recency expert cache
- asynchronous expert I/O
- CPU correctness/reference backend
- Vulkan compute backend for Windows + Android
- Windows host integration for L.U.M.I.A.
- Android NDK/JNI integration for ORBI Edge Mesh
- OpenAI-compatible local service as a later integration layer

## Development gates

- **OSM-00** — repository bootstrap and architecture contract
- **OSM-01** — portable project skeleton
- **OSM-02** — qpack parser/validator
- **OSM-03** — CPU reference primitives
- **OSM-04** — expert cache + storage abstraction
- **OSM-05** — Vulkan bootstrap on Windows
- **OSM-06** — Qwen dense-core kernels
- **OSM-07** — sparse MoE routing + streamed experts
- **OSM-08** — Gated DeltaNet
- **OSM-09** — 35B Windows pilot
- **OSM-10** — 80B Windows pilot
- **OSM-11+** — Android/Edge Mesh port and pilots

## Reference implementation

The first design reference is [Swiftlet](https://github.com/leonickson1/Swiftlet), frozen for the initial audit at commit:

`909c04213c9deb369dac0679d0872512cf3ab32e`

See `ingeniusvictor/orbi-swiftlet-lab` for the upstream audit and provenance record.

No model weights are stored in this repository.
