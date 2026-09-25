# OSM-20 — Vulkan resident expert

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Aggregate the three Q4 projections of one streamed MoE expert into a reusable
Vulkan-resident object.

OSM-19 proved one Q4 projection can be uploaded once and reused across tokens.
OSM-20 lifts that property to the complete expert MLP:

```text
VulkanResidentExpert
  ├─ gate Q4 weights
  ├─ up Q4 weights
  ├─ down Q4 weights
  ├─ input activation buffer
  ├─ gate activation buffer
  ├─ up activation buffer
  ├─ hidden activation buffer
  └─ output activation buffer
```

## Creation path

```text
qpack layer file
      ↓
host ExpertCacheEntry
      ↓
bind gate/up/down sections
      ↓
decode F32/F16/BF16 affine metadata
      ↓
upload three Q4 projections once
      ↓
VulkanResidentExpert
```

The source host cache slot is not required for inference after successful
creation. OSM-20 intentionally does not yet connect GPU residency lifetime to
the host cache eviction policy; that becomes a dedicated cache-management gate.

## Token execution

For every token:

```text
host input
    │ one upload
    ▼
input buffer
   ├───────────────┐
   ▼               ▼
gate preloaded   up preloaded
Q4 GEMV          Q4 GEMV
   │               │
   ▼               ▼
gate buffer       up buffer
   └───────┬───────┘
           ▼
      Vulkan SwiGLU
           │
           ▼
      hidden buffer
           │
           ▼
     down preloaded
       Q4 GEMV
           │
           ▼
      output buffer
           │ one download
           ▼
        host output
```

No expert weight upload happens during token execution.

## Certification

The OSM-20 fixture preserves the upstream-shaped expert organization:

- one 16 KiB fixed-stride expert blob;
- gate/up/down packed affine Q4 sections;
- BF16 scale/bias metadata;
- group size 64.

The test creates one resident expert and executes at least two different token
vectors through it. Each output is compared against an independent CPU
dequantize + gate/up + SwiGLU + down oracle.

## Exit gate

OSM-20 is GREEN when:

- Windows and Linux compile/test;
- Linux executes the full resident expert path with real Vulkan;
- all three projection weights remain resident across multiple tokens;
- activation buffers are reused across calls;
- output matches the CPU oracle within the established expert tolerance;
- device and shape ownership are validated.

## Next

**OSM-21 — Vulkan resident expert cache**

Map `(layer, expert)` to `VulkanResidentExpert` under an explicit byte/slot
budget with hit/miss/eviction accounting. That creates the GPU-side cache needed
before router Top-K and weighted multi-expert execution.
