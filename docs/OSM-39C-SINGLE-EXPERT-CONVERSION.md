# OSM-39C — bounded single-expert conversion pilot

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Convert one real routed expert from the pinned official
Qwen3-Next-80B-A3B-Instruct checkpoint into the OSM-39B QPACK expert layout
without downloading a full shard, layer, or model.

This is the first gate that exercises the complete path:

```text
official safetensors header
        ↓
strict HTTP byte ranges
        ↓
real BF16 gate/up/down matrices
        ↓
BF16 → F32 decode
        ↓
affine Q4 / group-size 64 quantization
        ↓
U32 packed weights + F32 scale/bias
        ↓
exact OSM-39B section offsets
        ↓
1 expert QPACK blob
```

## Pilot selection

The pinned pilot is:

```text
layer  = 0
expert = 0

model.layers.0.mlp.experts.0.gate_proj.weight
model.layers.0.mlp.experts.0.up_proj.weight
model.layers.0.mlp.experts.0.down_proj.weight
```

Official BF16 shapes:

```text
gate_proj  [512, 2048]  = 2,097,152 bytes
up_proj    [512, 2048]  = 2,097,152 bytes
down_proj  [2048, 512]  = 2,097,152 bytes
```

Total tensor payload fetched:

```text
6,291,456 bytes = 6 MiB
```

Safetensors JSON headers are fetched separately by the existing OSM-38C
bounded-range metadata gate.

## Quantization semantics

For each row and each 64-value group:

```text
bias  = min(group)
scale = (max(group) - min(group)) / 15
q     = clamp(round((x - bias) / scale), 0, 15)

x_hat = scale * q + bias
```

Eight Q4 values are packed into one little-endian U32 word, matching the
existing CPU/Vulkan dequantization contract.

Constant groups use scale=1 and bias=value, producing q=0 and exact
reconstruction.

The converter checks every source value is finite and proves each reconstructed
value stays within the nearest-level bound for its affine group.

## QPACK output

OSM-39B defines the exact destination geometry:

```text
expertStride = 1,966,080 bytes
```

OSM-39C writes one exact-size blob containing all nine sections:

```text
gate_proj.weight
gate_proj.scales
gate_proj.biases
up_proj.weight
up_proj.scales
up_proj.biases
down_proj.weight
down_proj.scales
down_proj.biases
```

No full layer file is allocated or downloaded.

## Integrity

The bounded fetch manifest records for each projection:

- tensor name;
- source shard;
- official shape;
- absolute source byte offset;
- exact source byte size;
- local filename;
- FNV-1a 64 checksum.

The C++ conversion gate rechecks file length and checksum before decoding.

## Exit gate

OSM-39C is GREEN when:

- local Q4 fixture matches the existing CPU dequantization oracle;
- affine group error bounds hold;
- official layer-0 expert-0 downloads exactly 6 MiB of tensor payload;
- all three official source shapes remain pinned as expected;
- output blob is exactly 1,966,080 bytes;
- Windows checkpoint workflow is GREEN;
- Ubuntu checkpoint workflow is GREEN;
- normal Windows/Ubuntu CI remains GREEN;
- tokenizer integration remains GREEN;
- Linux real Vulkan regression remains GREEN.

## Scope

This pilot deliberately converts **one expert only**. It does not yet generate a
full per-layer expert file and does not change runtime cache policy.

## Next

**OSM-39D — deterministic full-layer conversion**

Use the certified bounded converter repeatedly for all 512 experts of one layer,
write the exact fixed-stride `packed_experts/layer_00.bin`, verify offsets and
random expert readback, then establish resumable conversion before scaling to
all 48 layers.
