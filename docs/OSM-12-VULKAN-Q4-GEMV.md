# OSM-12 — Fused Vulkan affine Q4 GEMV

Status: **IMPLEMENTATION IN PROGRESS**

## Goal

Execute matrix-vector multiplication directly from MLX-style affine Q4 bytes
without first materializing a full float32 weight matrix.

For each output row and affine group:

```text
qdot = sum(q_i * x_i)
xsum = sum(x_i)
acc += scale[group] * qdot + bias[group] * xsum
```

This is algebraically equivalent to:

```text
W_float = dequantize(Q4)
y = W_float * x
```

while keeping the packed weight representation resident.

## Why this is the pivotal quantized gate

OSM-11 proved byte-level affine-Q4 dequantization. OSM-12 removes the
intermediate dequantized weight tensor and uses packed Q4 bytes directly in
compute. That is the execution pattern required by streamed MoE experts.

## Initial scope

- affine Q4 packed as eight nibbles per uint32;
- float32 scale/bias buffers;
- one logical input vector;
- one output value per matrix row;
- correctness-first row-parallel kernel;
- group-size 64 certification fixture.

Native F16/BF16 scale/bias loading and optimized subgroup reductions are
separate follow-up optimizations after semantic parity is sealed.

## Correctness oracle

The reference path is:

1. `cpu::dequantize_affine_rows`
2. `cpu::matvec_row_major`

The Vulkan fused result must match that composed CPU oracle within a small
floating-point tolerance.

## Next

After OSM-12 is GREEN, the next gate should bind this fused Q4 GEMV to a real
qpack expert section/cache slot so streamed expert bytes can execute without
copying through a dequantized model representation.
