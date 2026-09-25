# OSM-16 — Vulkan SwiGLU

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Move the sparse-expert activation stage from the portable CPU oracle to Vulkan.

The operation is:

```text
hidden[i] = SiLU(gate[i]) * up[i]
          = gate[i] / (1 + exp(-gate[i])) * up[i]
```

## Runtime path after this gate

```text
cached qpack expert
       │
       ├─ gate_proj Q4 ── Vulkan GEMV ──┐
       ├─ up_proj   Q4 ── Vulkan GEMV ──┤
       │                                 ▼
       │                           Vulkan SwiGLU
       │                                 │
       └─ down_proj Q4 ◄─ Vulkan GEMV ◄──┘
                         │
                         ▼
                    expert output
```

OSM-15 already executed all three Q4 projection GEMVs through Vulkan but used
the CPU reference SwiGLU between them. OSM-16 removes that CPU math dependency.

## Shader contract

- GLSL 450;
- workgroup size: 64;
- binding 0: gate float buffer;
- binding 1: up float buffer;
- binding 2: output float buffer;
- push constant: element count;
- SPIR-V is pinned in the repository and regenerated/diffed in Linux CI.

## Correctness

The standalone certification fixture uses 257 signed values so dispatch crosses
multiple workgroups and includes positive, zero-adjacent and negative inputs.

The Vulkan result is compared element-by-element with
`cpu::swiglu_inplace`.

Initial tolerance: `2e-5` absolute error, allowing small implementation
differences in the exponential function.

## Complete expert certification

The existing OSM-15 complete expert test remains active. With OSM-16 it now
executes:

1. gate Q4 GEMV on Vulkan;
2. up Q4 GEMV on Vulkan;
3. SwiGLU on Vulkan;
4. down Q4 GEMV on Vulkan.

The independent CPU oracle still performs dequantize + GEMV + SwiGLU and checks
the final output.

## Current memory boundary

This gate moves all expert arithmetic to Vulkan, but the bootstrap APIs still
return host-visible vectors between individual calls. Therefore OSM-16 proves
semantic GPU execution, not yet a zero-copy persistent GPU pipeline.

That distinction is intentional.

## Exit gate

OSM-16 is GREEN when:

- pinned SwiGLU SPIR-V exactly matches the GLSL source;
- Windows and Linux compile/test;
- Linux executes the standalone SwiGLU shader with required Vulkan compute;
- standalone GPU output matches the CPU oracle;
- the complete streamed expert MLP remains GREEN with Vulkan SwiGLU.

## Next

OSM-17 should introduce a reusable Vulkan tensor/buffer abstraction and keep
gate/up/intermediate activations resident across the expert pipeline.

After that, router Top-K plus weighted multi-expert accumulation can form the
first complete sparse MoE layer.
