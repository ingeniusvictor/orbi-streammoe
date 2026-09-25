# OSM-09 — Vulkan compute round-trip

Status: **IMPLEMENTED / CI GATE**

## Goal

Execute the first real GPU compute workload in ORBI StreamMoE without yet
introducing model-specific math.

The gate proves the complete host-to-GPU-to-host path:

```text
host value
   ↓
host-visible Vulkan storage buffer
   ↓
descriptor set
   ↓
compute pipeline
   ↓
SPIR-V shader
   ↓
command buffer
   ↓
compute queue submit
   ↓
fence
   ↓
host readback
```

## Shader

The bootstrap shader writes the fixed sentinel:

`0x12345678`

into a one-word storage buffer.

The module is embedded as a minimal SPIR-V 1.0 program instead of requiring a
GLSL compiler at application runtime. It uses the original Vulkan-compatible
SPIR-V 1.0 `Uniform + BufferBlock` representation of a storage buffer so this
bootstrap remains compatible with Vulkan 1.0-class drivers.

The runtime performs a structural word-count/header validation before creating
the Vulkan shader module.

## Memory behavior

OSM-09 requests host-visible memory and prefers host-coherent memory.

If the selected memory type is host-visible but not coherent, the gate
explicitly flushes before dispatch and invalidates before readback.

This avoids accidentally depending on cache coherence behavior that differs
between desktop and mobile drivers.

## What this proves

A successful hardware execution proves that ORBI can:

- create a Vulkan storage buffer;
- allocate/bind host-visible device memory;
- create descriptor-set and pipeline layouts;
- create a shader module and compute pipeline;
- allocate/record/submit a command buffer;
- synchronize with a fence;
- read back GPU-written bytes.

That is the minimum reusable compute bridge required for later Qwen kernels.

## What this does not prove

It does not yet prove:

- Q4 dequantization correctness;
- RMSNorm;
- GEMV performance;
- routed MoE execution;
- Android sustained throughput;
- any 35B/80B model result.

## CI behavior

A CI machine with no Vulkan compute device is allowed to report a graceful
skip-path PASS because OSM-08 already validates that absence is handled
cleanly.

A machine with a compute device must execute the shader and read back the exact
sentinel; a wrong value is a hard failure.

## Exit gate

OSM-09 is green when Windows and Linux builds pass and any compute-capable test
host returns exactly `0x12345678` from GPU readback.

## Next

OSM-10 should implement the first model-relevant Vulkan primitive:
**RMSNorm**, with the existing CPU reference backend as the numerical oracle.
