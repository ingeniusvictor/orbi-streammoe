# OSM-15 — complete streamed Q4 expert MLP

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Execute one complete Qwen-style sparse expert directly from a fixed-stride
qpack cache slot.

The expert topology is:

```text
gate = gate_proj(x)
up   = up_proj(x)
hidden = SiLU(gate) * up
out  = down_proj(hidden)
```

All three projection matrices remain packed affine Q4. The gate/up/down bytes
are read from one resident `ExpertCacheEntry`.

## Why this matters

OSM-13 connected one cached qpack projection to Vulkan.
OSM-14 added upstream-compatible F32/F16/BF16 quantization metadata.

OSM-15 composes those capabilities into the first complete sparse-expert
execution path.

The fixed-stride expert is loaded once:

```text
SSD layer file
    ↓
one expert cache slot
    ├─ gate_proj Q4
    ├─ up_proj Q4
    └─ down_proj Q4
          ↓
Vulkan Q4 GEMV x 3
          ↓
SwiGLU
          ↓
expert output
```

## Certification fixture

The fixture intentionally follows important upstream qpack properties:

- model type: qwen3_next;
- expert stride: 16 KiB;
- all three expert projections are present;
- packed weights use U32 affine Q4;
- scale/bias metadata uses BF16;
- quantization group size is 64;
- one cache miss loads the complete expert blob.

For CI-sized execution:

- hidden dimension: 128;
- expert intermediate dimension: 64;
- gate/up: 64 x 128;
- down: 128 x 64.

These are reduced dimensions with the same projection topology as a Qwen sparse
expert.

## Correctness oracle

The CPU oracle independently performs:

1. Q4 dequantize + GEMV for gate;
2. Q4 dequantize + GEMV for up;
3. SiLU(gate) * up;
4. Q4 dequantize + GEMV for down.

The streamed Vulkan path must match the CPU result.

## Current execution boundary

The three Q4 projections execute through Vulkan. SwiGLU is currently evaluated
with the portable CPU reference primitive between gate/up and down.

This is a correctness-first bridge. Because the current Vulkan GEMV bootstrap
already returns host-visible output vectors, OSM-15 does not add an additional
weight-materialization penalty.

A later gate should keep intermediate activations resident on Vulkan and run
SwiGLU there as well.

## Exit gate

OSM-15 is GREEN when:

- Windows and Linux build/test successfully;
- Linux executes the complete expert with required Vulkan compute;
- the qpack expert stride is 16 KiB;
- all gate/up/down sections bind from the same cache slot;
- only one expert storage miss is required;
- Vulkan expert output matches the independent CPU oracle.

## Next

OSM-16 should implement Vulkan SwiGLU and a persistent activation buffer so
gate/up/down can remain GPU-resident.

After that, OSM-17 can add router Top-K plus weighted expert accumulation,
creating the first complete sparse MoE layer.
