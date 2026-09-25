# OSM-10 — Vulkan RMSNorm correctness kernel

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Introduce the first model-relevant Vulkan math primitive and validate it
numerically against the OSM-03 CPU oracle.

The semantic contract is:

```text
y = x * rsqrt(mean(x^2) + eps) * weight
```

for each row independently.

## Why RMSNorm first

RMSNorm is small enough to audit precisely while exercising the same core
Vulkan resources later Qwen kernels need:

- floating-point storage buffers;
- multiple descriptor bindings;
- push constants;
- per-row compute dispatch;
- shader-generated arithmetic;
- synchronized GPU readback;
- numerical comparison against a CPU reference.

## Shader strategy

The human-readable source is:

`shaders/rmsnorm.comp`

It is compiled to SPIR-V 1.0 and pinned as:

`include/orbi/streammoe/generated/rmsnorm_spv.hpp`

Linux CI recompiles the GLSL with glslang and byte-compares the generated
header against the pinned version. This prevents source/SPIR-V drift without
requiring a shader compiler at runtime on Windows or Android.

## Correctness gate

The test uses three rows of eight values plus a non-uniform weight vector.

1. CPU output is produced by `cpu::rms_norm_inplace`.
2. The same inputs execute through Vulkan.
3. Every result element must be within `3e-5` absolute error.
4. Invalid shape input must be rejected before dispatch.

Linux CI runs with `ORBI_REQUIRE_VULKAN_COMPUTE=1`, so a missing Vulkan
context is a hard failure there. Mesa llvmpipe provides a deterministic
software Vulkan device when no hardware GPU is exposed by the hosted runner.

Windows CI remains the cross-platform compile/test lane; hardware execution
will later be certified on a real Windows Vulkan device.

## Performance boundary

This shader intentionally uses one invocation per row and loops across the
entire hidden dimension. It is a correctness kernel, not the final optimized
RMSNorm implementation.

Once numerical parity is sealed, later performance work may use subgroup
reductions and larger workgroups without changing the semantic oracle.

## Exit gate

OSM-10 is GREEN when:

- Windows and Linux builds pass;
- pinned SPIR-V exactly matches the GLSL source;
- Linux executes RMSNorm through a real Vulkan implementation;
- GPU output matches the CPU oracle within tolerance;
- no Vulkan surface/window dependency is introduced.

## Next

OSM-11 should implement the first quantized primitive needed by Qwen expert
execution: **MLX-style affine Q4 dequantization on Vulkan**, validated against
`cpu::dequantize_affine_rows`.
