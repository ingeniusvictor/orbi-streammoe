# OSM-25B — Vulkan resident shared expert

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Add the always-on Qwen3-Next shared expert to the existing routed Vulkan MoE
path without adding per-expert output readbacks.

OSM-25A froze the exact semantics:

```text
shared = down(SiLU(gate(x)) * up(x))
scale  = sigmoid(shared_expert_gate · x)
output = scale * shared
```

OSM-25B makes the shared MLP resident in Vulkan and accumulates its gated output
into the same accumulator already used by routed experts.

## Resident shared expert

`VulkanResidentSharedExpert` owns:

- resident affine-Q4 gate projection;
- resident affine-Q4 up projection;
- resident affine-Q4 down projection;
- reusable gate/up/hidden/output Vulkan activation buffers;
- a host-resident scalar gate vector.

The scalar gate remains on the host in this gate because the current Qwen router
also consumes the hidden vector on the host. Computing one 2048-element dot
product does not require another transfer. If routing later moves fully to
Vulkan, the scalar gate can move with it.

## Full sparse-MoE path

The integrated execution target is:

```text
hidden host
    │ one upload
    ▼
shared Vulkan hidden buffer
    │
    ├─ routed Top-K experts ── weighted Vulkan accumulate ─┐
    │                                                     │
    └─ resident shared expert ─ sigmoid scalar weight ─────┤
                                                          ▼
                                              Vulkan accumulator
                                                          │
                                                  one final download
```

The shared expert is always executed and is never part of Top-K selection or
the streamed routed-expert cache.

## Architectural placement

The shared expert belongs to the resident dense core of Qwen3-Next, not the
streamed routed-expert pool. OSM-25B accepts already-decoded host Q4 views; a
later checkpoint-loader gate will bind these views to the dense tensors from
the real qpack/model.safetensors payload.

## Exit gate

OSM-25B is GREEN when:

- Windows and Linux build/test;
- Linux executes the full sparse block with real Vulkan;
- routed Top-K selection remains unchanged;
- shared MLP gate/up/down weights stay resident across repeated tokens;
- shared scalar gate matches the OSM-25A CPU oracle;
- shared output is accumulated on Vulkan without a dedicated host readback;
- only one final sparse-MoE output download is required;
- cold and warm routed-cache passes match an independent CPU oracle.

## Next

**OSM-26 — dense checkpoint binding**

Bind router, shared-expert and other always-resident model tensors from the
qpack dense `model.safetensors` payload into runtime views. That removes the
synthetic host-weight fixtures and prepares the first upstream-compatible Qwen
layer fixture.
