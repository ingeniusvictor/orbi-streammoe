# OSM-24B — Vulkan weighted accumulation

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Add the device-side primitive needed to combine routed expert outputs without
downloading each expert result to the host.

The operation is:

```text
accumulator[i] += routing_weight * expert_output[i]
```

Both source and accumulator are existing `VulkanFloatBuffer` objects.

## Why this matters

OSM-23 proved routed Top-K correctness, but its execution path still did:

```text
expert output -> host
expert output -> host
...
weighted host accumulation
```

OSM-24A made a complete expert output remain in Vulkan.

OSM-24B now provides the missing device-side accumulation primitive:

```text
resident expert output
        │
        ▼
weighted_accumulate.comp
        │
        ▼
shared Vulkan accumulator
```

## Shader contract

The compute shader uses 64 threads per workgroup and two storage buffers:

- binding 0: read-only expert/source output;
- binding 1: read/write accumulator.

Push constants:

- `count: uint32`;
- `weight: float`.

The shader is compiled to pinned SPIR-V and the Linux CI recompiles the GLSL
source and diffs it against the committed header.

## Synchronization

The dispatch establishes a pre-compute memory barrier for prior host writes and
compute-shader writes, then a post-compute barrier making the accumulator
available to later compute reads or a final host readback.

This is deliberately conservative for the correctness phase.

## Certification

The OSM-24B test creates persistent source and accumulator buffers, zeroes the
accumulator once, then executes:

```text
acc += 0.25 * source_A
acc += -0.5 * source_B
```

There is only one accumulator download, after both dispatches.

The result is compared element-by-element with a CPU oracle.

## Exit gate

OSM-24B is GREEN when:

- Windows and Linux compile/test;
- Linux executes the primitive with real Vulkan;
- the pinned SPIR-V exactly matches GLSL recompilation;
- multiple weighted dispatches accumulate rather than overwrite;
- only one final readback is required by the certification path;
- GPU result matches the CPU oracle.

## Next

**OSM-24C — routed Top-K Vulkan accumulation**

Combine OSM-22/21/24A/24B:

```text
hidden host
   │ one upload
   ▼
shared Vulkan input
   │
Top-K ids + weights
   │
   ├─ resident expert -> output buffer -> weighted accumulate
   ├─ resident expert -> output buffer -> weighted accumulate
   └─ ...
                                      │
                                      ▼
                           shared accumulator
                                      │
                                one download
```

That removes per-expert input uploads and per-expert output readbacks from the
routed MoE hot path.
