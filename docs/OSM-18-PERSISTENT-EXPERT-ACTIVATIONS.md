# OSM-18 — persistent expert activation pipeline

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Remove intermediate host readbacks from the complete streamed expert path.

OSM-17 introduced reusable Vulkan float buffers. OSM-18 teaches the Q4 GEMV
and SwiGLU stages to consume and produce those buffers directly.

## Buffer-native operations

New execution paths:

```text
run_vulkan_q4_gemv_buffers(...)
run_vulkan_swiglu_buffers(...)
```

Both validate that input/output buffers belong to the active Vulkan device.

### Q4 GEMV

Packed expert weights and their affine metadata are still staged from the
qpack/cache-owned host bytes for each projection. The activation vector is read
from a persistent Vulkan buffer and the result is written directly to another
persistent Vulkan buffer.

### SwiGLU

Gate and up activation buffers are consumed directly and the hidden activation
is written to a third Vulkan buffer.

## Complete streamed expert after OSM-18

```text
host input
    │ upload once
    ▼
input Vulkan buffer
    ├──────────────┐
    ▼              ▼
gate Q4 GEMV    up Q4 GEMV
    │              │
    ▼              ▼
gate buffer      up buffer
    └──────┬───────┘
           ▼
      Vulkan SwiGLU
           │
           ▼
      hidden buffer
           │
           ▼
      down Q4 GEMV
           │
           ▼
      output buffer
           │ download once
           ▼
       host output
```

No gate, up or hidden activation is read back to the host.

## Synchronization

Each buffer-native dispatch records a shader-write memory barrier and completes
through the existing queue/fence bootstrap contract before the next stage.

This is correctness-first synchronization. Command-buffer fusion and async
pipelining are later performance gates.

## Certification

The standalone persistent-ops test certifies:

- Q4 GEMV with Vulkan-buffer input/output versus CPU dequantize + matvec;
- SwiGLU with Vulkan-buffer input/output versus CPU SwiGLU;
- real Vulkan execution in Linux CI.

The existing complete streamed-expert test then certifies the whole
qpack/cache -> gate/up -> SwiGLU -> down path against its independent CPU oracle.

## Memory boundary

OSM-18 removes intermediate **activation** readbacks.

It does not yet make expert weights device-resident: Q4 weight/scales/biases are
still staged from the expert cache slot into Vulkan buffers for each projection.
That is the next major memory/throughput optimization boundary.

## Exit gate

OSM-18 is GREEN when:

- Windows and Linux build/test successfully;
- Linux executes buffer-native Q4 GEMV and SwiGLU;
- both standalone operations match their CPU oracles;
- the complete streamed expert remains correct;
- the expert pipeline uploads input once and downloads only final output.

## Next

OSM-19 should introduce a Vulkan-side expert weight slot or staging cache so one
streamed qpack expert can upload its packed gate/up/down weights once and reuse
them across all projection dispatches.

After that, router Top-K and weighted multi-expert accumulation can form the
first complete sparse MoE layer.
