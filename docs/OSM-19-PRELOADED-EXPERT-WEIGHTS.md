# OSM-19 — preloaded Vulkan expert weights

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Remove repeated projection-weight staging from the streamed-expert hot path.

OSM-18 kept expert activations resident on Vulkan, but each gate/up/down Q4 GEMV
still recreated and uploaded packed weights plus affine metadata for every
projection dispatch.

OSM-19 introduces a persistent projection object:

```text
VulkanQ4ProjectionWeights
  ├─ packed Q4 VkBuffer
  ├─ scales VulkanFloatBuffer
  ├─ biases VulkanFloatBuffer
  ├─ out_dim / in_dim
  └─ group_size
```

A projection is uploaded once and may be reused across multiple token
executions.

## New execution path

```text
qpack/cache expert bytes
       │ one-time projection bind/decode
       ▼
VulkanQ4ProjectionWeights
       │
       ├──────── token N ────────┐
       │                         │
       ├──────── token N+1 ──────┤
       │                         │
       └──────── token ... ──────┘
                 │
                 ▼
       run_vulkan_q4_gemv_preloaded
```

This is the first step toward a Vulkan-side expert cache.

## Certification

The OSM-19 test:

- builds a deterministic affine-Q4 projection;
- uploads it once;
- verifies the persistent packed buffer and metadata buffers are valid;
- sends two different input vectors through the same uploaded projection;
- compares both GPU results against CPU dequantize + matvec;
- requires real Vulkan execution in Linux CI.

The test intentionally reuses the same input/output activation buffers as well,
so the only per-token host transfer is the activation upload/download required by
this isolated gate.

## Memory boundary

OSM-19 does not yet define eviction or a multi-expert GPU cache.

It proves the unit that such a cache will own:

**one already-uploaded gate/up/down projection.**

The next gate should aggregate three persistent projections into one
Vulkan-resident expert object and connect its lifetime to the host-side
`ExpertCache` residency policy.

## Exit gate

OSM-19 is GREEN when:

- Windows and Linux build/test;
- Linux executes the preloaded path with real Vulkan;
- one uploaded projection serves multiple token dispatches;
- results match the CPU oracle within the OSM-12 tolerance;
- device ownership and projection geometry are rejected on mismatch.

## Next

**OSM-20 — Vulkan resident expert object**

Create one object holding preloaded gate/up/down projections and execute:

```text
input
  ├─ gate GEMV
  ├─ up GEMV
  └─ SwiGLU
       ↓
     down GEMV
       ↓
     output
```

without re-uploading any expert weights between tokens.
