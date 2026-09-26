# OSM-40A — deterministic bounded dense/global conversion

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Begin execution of the OSM-39A non-expert conversion plan into the exact
`model.safetensors` contract already consumed by the decoder runtime.

OSM-40A deliberately certifies the output format and both conversion actions on
bounded tensors before production-scale global matrices are attempted.

## Supported actions

### BF16 → F32

Used for vector-like runtime tensors such as:

- `model.norm.weight`;
- layer RMSNorm weights;
- DeltaNet conv/bias/A_log/norm vectors;
- GQA q/k norm vectors.

### BF16 → affine Q4

A matrix is converted into the MLX-compatible triplet:

```text
<path>.weight  U32  [rows, cols/8]
<path>.scales  F32  [rows, cols/group_size]
<path>.biases  F32  [rows, cols/group_size]
```

The packing contract is identical to the already-certified OSM-39C affine Q4
quantizer and the `QpackMlxCheckpoint` runtime reader.

## Deterministic safetensors writer

OSM-40A writes tensors in lexicographic name order, uses deterministic JSON
metadata and 8-byte header padding, then reopens the result with the production
`SafetensorsReader`.

Repeated conversion of identical inputs must produce byte-identical output.

## Official bounded gate

The pinned Qwen3-Next checkpoint downloads only:

- `model.norm.weight` — 2,048 BF16 values;
- `model.layers.0.mlp.shared_expert_gate.weight` — 1×2,048 BF16.

Expected source payload: **8,192 bytes**.

Expected converted tensor payload:

- model norm F32: 8,192 bytes;
- shared gate Q4 weight: 1,024 bytes;
- shared gate scales: 128 bytes;
- shared gate biases: 128 bytes;
- total: **9,472 bytes**.

This proves one global F32 action and one layer-dense affine action against real
official weights without materializing large matrices.

## Scope boundary

OSM-40A does **not** yet convert:

- `model.embed_tokens.weight`;
- `lm_head.weight`;
- all dense matrices across 48 layers.

Those require a row/chunk streamed writer so memory remains bounded.

## Next

**OSM-40B — streamed production dense/global safetensors builder**

Add incremental deterministic safetensors layout planning and row/chunk
conversion so embedding, LM-head and all 48-layer dense tensors can be produced
without full matrix materialization.
