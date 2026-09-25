# OSM-24C — routed Top-K Vulkan accumulation

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Join the routing, streaming cache, resident expert, and weighted-accumulate
pieces into one correctness-first routed MoE path with minimal host/device
activation traffic.

OSM-23 used one host upload and one host download **per selected expert**.

OSM-24C changes the path to:

```text
hidden host
    │
    │ one upload
    ▼
shared Vulkan input
    │
CPU Top-K ids + weights
    │
    ├─ GPU expert cache -> expert A -> resident output ─┐
    │                                      weight A     │
    ├─ GPU expert cache -> expert B -> resident output ─┤
    │                                      weight B     │
    └─ ...                                               │
                                                        ▼
                                           Vulkan accumulator
                                                        │
                                                        │ one download
                                                        ▼
                                                routed MoE output
```

## New path

`run_weighted_routed_moe_vulkan_accum(...)` performs:

1. CPU router oracle: projection, softmax, Top-K, optional renormalization;
2. one Vulkan input-buffer allocation/upload for the hidden vector;
3. one Vulkan accumulator allocation initialized to zero;
4. selected expert resolution through `VulkanResidentExpertCache`;
5. complete gate/up/SwiGLU/down expert execution from the shared input buffer;
6. device-side `acc += weight * expert_output`;
7. one final accumulator download.

No selected expert output is downloaded individually.

## Cache behavior

The first execution may stream experts from qpack through the host cache and
promote them to Vulkan.

A repeated execution with enough Vulkan cache budget should:

- hit every selected resident expert;
- avoid host-cache access;
- avoid qpack/SSD reads;
- still use one shared hidden upload and one final routed-result readback.

## Certification

The fixture selects Top-2 from three routed experts and uses:

- Q4 gate/up/down expert weights;
- BF16 affine metadata;
- one host expert-cache slot;
- two Vulkan resident-expert slots.

Expected output is built independently from CPU router semantics and fully
dequantized CPU expert MLPs.

The test verifies both cold and warm routed passes.

## Exit gate

OSM-24C is GREEN when:

- Windows and Linux compile/test;
- Linux executes the integrated path with real Vulkan;
- router ids/weights remain unchanged from OSM-22;
- first pass loads exactly selected experts;
- warm pass is served entirely by the Vulkan expert cache;
- one hidden upload is used for all selected experts in a pass;
- no per-expert output readback occurs in the implementation path;
- only one final routed-output download occurs;
- output matches the independent CPU oracle.

## Remaining Qwen sparse block work

OSM-24C covers the **routed experts** only.

Qwen3-Next also adds:

```text
shared_expert(x) * sigmoid(shared_expert_gate(x))
```

to the routed result.

The next semantic gate should therefore add the shared expert before the sparse
MoE block can be called architecture-complete.
