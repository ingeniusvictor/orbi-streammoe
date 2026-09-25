# OSM-17 — reusable Vulkan float activation buffer

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Introduce the first reusable Vulkan storage object for activation tensors so
later expert stages do not need to allocate a new buffer for every operation.

OSM-16 moved all sparse-expert arithmetic to Vulkan, but each bootstrap kernel
still owns short-lived host-visible buffers internally.

OSM-17 establishes a reusable buffer lifetime independent of one kernel call.

## Contract

`VulkanFloatBuffer`:

- owns one Vulkan `VkBuffer` and its bound memory;
- is movable and non-copyable;
- is sized in float elements;
- supports repeated upload/download;
- exposes an opaque native buffer handle without leaking Vulkan headers;
- uses storage + transfer usage flags;
- prefers HOST_VISIBLE + HOST_COHERENT memory;
- handles explicit flush/invalidate when coherent memory is unavailable.

The creating `VulkanComputeContext` must outlive the buffer.

## Why this matters

The target streamed-expert path is moving from:

```text
kernel call
  allocate
  upload
  dispatch
  download
  destroy
```

toward:

```text
session / expert pipeline
  allocate activation buffers once
        ↓
  gate GEMV
        ↓
  up GEMV
        ↓
  SwiGLU
        ↓
  down GEMV
        ↓
  read only final output
```

OSM-17 is the ownership/lifetime foundation required for that transition.

## Certification

The test creates a 513-float activation buffer, then verifies:

1. creation and native handle validity;
2. exact upload/download round trip;
3. move semantics preserve ownership;
4. the same allocation can be reused with a second payload;
5. mismatched uploads are rejected.

Linux CI requires a real Vulkan compute context through Mesa llvmpipe when no
hardware GPU is available. Windows certifies the same portable code path at
build/test level.

## Exit gate

OSM-17 is GREEN when:

- Windows and Linux build/test successfully;
- Linux creates a real Vulkan buffer;
- repeated upload/download is exact;
- move semantics do not invalidate the Vulkan allocation;
- invalid-size writes fail safely.

## Next

OSM-18 should add buffer-native Q4 GEMV and SwiGLU dispatch APIs.

That will allow gate/up/intermediate/down activations to stay resident across a
complete expert pipeline, with host readback only after the final down
projection.
