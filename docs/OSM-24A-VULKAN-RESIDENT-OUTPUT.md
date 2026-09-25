# OSM-24A — resident expert Vulkan output bridge

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Create the missing bridge needed before routed-expert weighted accumulation can
stay entirely on Vulkan.

OSM-23 still used:

```text
expert A Vulkan output -> host vector
expert B Vulkan output -> host vector
                         ↓
                   host accumulation
```

OSM-24A adds an execution path where an expert consumes an already-uploaded
Vulkan activation buffer and leaves its final output in its reusable Vulkan
buffer.

## New API

`VulkanResidentExpert::run_from_buffer(...)`

takes:

- the existing Vulkan compute context;
- a `VulkanFloatBuffer` containing the hidden activation.

It executes:

```text
external Vulkan input
    ├─ gate Q4 GEMV
    ├─ up Q4 GEMV
    └─ SwiGLU
          ↓
       down Q4 GEMV
          ↓
resident expert output buffer
```

No output download occurs inside this method.

`output_buffer()` exposes the reusable output buffer for the next Vulkan
operation.

## Why this sub-gate exists

A production Top-10 routed MoE should not:

- upload the same hidden vector ten times;
- download ten expert outputs;
- accumulate them on the CPU.

Before adding a weighted-accumulate compute shader, the expert runtime first
needs a stable device-to-device contract.

OSM-24A provides exactly that boundary.

## Compatibility

The original host-facing `run(...)` API remains available. It now:

1. uploads into the expert's internal input buffer;
2. calls the same buffer execution path;
3. downloads only the final expert output.

This keeps older gates valid while removing duplicate execution logic.

## Certification

The test:

- creates one complete resident Q4 expert;
- creates one external shared Vulkan input buffer;
- uploads two different token vectors to that same buffer;
- executes the expert through `run_from_buffer`;
- verifies the output buffer handle stays unchanged;
- performs download only after execution for oracle comparison;
- compares both results against the independent CPU expert oracle.

## Exit gate

OSM-24A is GREEN when:

- Windows and Linux build/test;
- Linux runs the path with real Vulkan;
- external input buffer ownership/shape is validated;
- expert output remains resident after execution;
- the same output buffer is reused across tokens;
- numerical parity remains within the established expert tolerance.

## Next

**OSM-24B — Vulkan weighted accumulation**

Add a tiny compute primitive:

```text
accumulator[i] += routing_weight * expert_output[i]
```

Then the routed Top-K path can upload the hidden state once, execute each
resident expert from the shared input buffer, accumulate all selected outputs on
Vulkan, and perform one final readback.
