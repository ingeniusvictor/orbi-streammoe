# OSM-13 — qpack expert cache slot to Vulkan Q4 GEMV

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Connect the storage/cache side of ORBI StreamMoE to the Vulkan Q4 execution
side without first materializing a full dequantized model.

The path certified by this gate is:

```text
qpack layer file
      ↓
ExpertStorage
      ↓
ExpertCache (fixed-stride slot)
      ↓
qpack section metadata
      ↓
packed Q4 weight + scale/bias views
      ↓
Vulkan fused Q4 GEMV
      ↓
CPU-oracle parity
```

## What this proves

OSM-12 proved that packed affine-Q4 bytes can participate directly in GPU
matrix-vector multiplication.

OSM-13 proves those packed bytes can come from the exact object that the
streaming runtime uses: a resident `ExpertCacheEntry` loaded from a qpack
expert blob.

The expert remains represented as its fixed-stride packed slot.

## Section binding

The initial binder validates:

- manifest quantization is Q4;
- projection contains `weight`, `scales`, and `biases`;
- weight dtype is `U32`;
- section offsets/sizes stay inside the cache slot;
- declared shapes agree with the packed bytes;
- scale/bias geometry agrees with group size.

OSM-13 uses F32 scale/bias sections in its synthetic qpack fixture. Real Swiftlet
qpack containers commonly carry lower-precision scale metadata; native BF16/F16
section decoding is intentionally the next compatibility step.

## Cache behavior

The certification fixture also requires:

- first expert fetch = cache miss;
- repeated fetch = cache hit;
- repeated fetch reuses the same resident slot.

This ensures the GPU bridge is attached to streamed-cache semantics, not just a
standalone byte buffer.

## Correctness oracle

The same section views are passed through:

1. `cpu::dequantize_affine_rows`
2. `cpu::matvec_row_major`

and compared with the fused Vulkan output.

## Exit gate

OSM-13 is GREEN when:

- Windows and Linux compile and test;
- Linux executes the bridge with a real Vulkan compute device (Mesa llvmpipe is
  acceptable for CI);
- cache miss/hit behavior is verified;
- qpack section binding is bounds-checked;
- GPU output matches the CPU oracle within the OSM-12 tolerance.

## Next

OSM-14 should add native qpack scale/bias dtype support (BF16/F16) and then run
the bridge against an upstream-compatible packed expert fixture rather than an
F32-only synthetic one.
