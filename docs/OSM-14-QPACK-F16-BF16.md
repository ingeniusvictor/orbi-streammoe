# OSM-14 — native qpack F16/BF16 expert metadata

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Accept the quantization metadata dtypes emitted by upstream Swiftlet-compatible
qpack containers instead of requiring float32-only synthetic metadata.

Upstream Swiftlet preserves the original safetensors dtype for each packed
expert section. Its runtime supports quantization scale/bias metadata in:

- F32
- F16
- BF16

OSM-14 mirrors those decode semantics in the portable ORBI qpack bridge.

## Decode semantics

For each scale/bias section:

- **F32**: copy IEEE-754 binary32 bits into float.
- **F16**: decode IEEE-754 binary16, including subnormals, infinities and NaNs.
- **BF16**: place the stored 16 bits in the high half of an IEEE-754 binary32.

The packed Q4 weight remains U32 and is not expanded into a full float matrix.

## Runtime path

```text
qpack expert cache slot
        │
        ├─ Q4 U32 weight  ───────────────┐
        ├─ F16/BF16/F32 scales ─decode─┐ │
        └─ F16/BF16/F32 biases ─decode┤ │
                                       ▼ ▼
                                fused Vulkan Q4 GEMV
                                       │
                                       ▼
                                  output vector
```

Metadata is decoded to a small owned float vector for correctness and portable
backend compatibility. This does not dequantize the expert weight matrix.

A later optimization may bind F16/BF16 metadata directly in Vulkan.

## Upstream compatibility basis

The frozen Swiftlet qpack repacker records each expert section's original dtype
from safetensors, and Swiftlet's checkpoint/runtime code handles F32, F16 and
BF16 scale data. ORBI therefore treats those three dtype strings as the initial
compatibility contract.

## Certification fixture

The OSM-14 test creates three equivalent qpack containers:

- F32 metadata;
- F16 metadata;
- BF16 metadata.

All three use:

- affine Q4 packed weights;
- group size 64;
- identical logical scale/bias values chosen to be exactly representable;
- the same fixed-stride expert/cache path;
- the same CPU dequantize + matvec oracle.

When Vulkan compute is available, each dtype must also execute through the
fused Q4 GEMV bridge and match the CPU oracle.

Linux CI requires Vulkan execution through Mesa llvmpipe when no hardware GPU
is exposed. Windows CI verifies portable compilation and the full test suite.

## Exit gate

OSM-14 is GREEN when:

- F32 regression remains green;
- F16 metadata decodes correctly;
- BF16 metadata decodes correctly;
- section byte sizes are validated against shape and dtype;
- Linux executes F32/F16/BF16 qpack expert projections through Vulkan;
- Windows and Linux CI both pass.

## Next

OSM-15 should build an upstream-shaped expert fixture containing all three
Qwen MoE projections:

- gate_proj
- up_proj
- down_proj

and execute a complete streamed expert MLP:

```text
gate = GEMV(gate_proj, x)
up   = GEMV(up_proj, x)
hidden = SiLU(gate) * up
out  = GEMV(down_proj, hidden)
```

That will be the first complete Qwen sparse expert execution gate.
