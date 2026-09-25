# OSM-11 — Vulkan affine Q4 dequantization

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Port the first quantized Qwen primitive to Vulkan while preserving the exact
MLX affine layout already implemented by the OSM-03 CPU oracle.

For Q4, one `uint32_t` stores eight 4-bit values. For each logical value:

```text
q = (packed_word >> (lane * 4)) & 0xF
value = scale[group] * float(q) + bias[group]
```

The scale/bias group is selected from `col / group_size`.

## Initial scope

OSM-11 deliberately supports **affine Q4 only**. Q8 remains in the CPU oracle
and can be added after the Q4 path is sealed.

The shader uses:

- binding 0: packed `uint32` weights;
- binding 1: float scales;
- binding 2: float biases;
- binding 3: float output;
- push constants: total logical values, packed columns, logical columns,
  group size.

Each invocation reconstructs one logical value. Workgroup size is 64.

## Correctness target

The Vulkan result must match `cpu::dequantize_affine_rows` for:

- multiple rows;
- more than one packed word per row;
- more than one affine group per row;
- non-trivial positive and negative biases;
- edge quantized values 0 and 15.

## Why this matters

Qwen expert weights are stored quantized on disk. ORBI StreamMoE must be able
to consume those bytes without materializing a full dequantized model in RAM.
This gate proves the byte-level quantization semantics before combining
dequantization with GEMV.

## Certification

The test uses a Qwen-representative group size of 64 with 2 rows x 128 logical
columns. Packed nibbles cover the full 0..15 range through non-trivial patterns,
and scales/biases vary by row and group.

The Vulkan output is compared element-by-element to
`cpu::dequantize_affine_rows` with an absolute tolerance of `1e-6`.
Linux CI requires a real Vulkan compute context through Mesa llvmpipe when no
hardware GPU is exposed.

This correctness gate uses float32 scale/bias buffers. Native F16/BF16 qpack
scale loading belongs in the fused GEMV path, where avoiding intermediate
dequantized weights matters.

## Exit gate

OSM-11 is GREEN when:

- pinned Q4 SPIR-V exactly matches the GLSL source;
- Windows and Linux builds/tests pass;
- Linux executes the Q4 shader through Vulkan;
- GPU output matches the CPU affine-Q4 oracle;
- invalid group geometry is rejected before dispatch.

## Next

After Q4 dequant parity is GREEN, OSM-12 should fuse affine dequantization with
matrix-vector multiplication so streamed expert bytes can participate directly
in Qwen inference.
