# OSM-40B — streamed production dense/global safetensors builder

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Remove full-matrix materialization from production conversion of large dense
Qwen tensors, especially:

- `model.embed_tokens.weight`;
- `lm_head.weight`;
- large layer-dense matrices.

OSM-40B extends the bounded OSM-40A conversion proof with a deterministic,
resumable safetensors file builder.

## Deterministic layout

All output tensor metadata is supplied before payload conversion begins.
Tensor names are sorted lexicographically, offsets are computed once, the
safetensors header is emitted once, and the final payload region is
preallocated.

No payload chunk can change the file layout.

## Row-streamed writes

Each output tensor is written as contiguous row chunks:

```text
source BF16 rows
      ↓
bounded BF16 decode
      ↓
affine Q4 conversion
      ↓
weight / scales / biases row chunks
      ↓
fixed safetensors offsets
```

Peak conversion memory therefore scales with the configured row chunk, not the
full vocabulary matrix.

## Resume journal

Every committed output chunk records:

- tensor name;
- first row;
- row count;
- FNV-1a checksum.

Resume re-verifies each certified chunk against the output file. New chunks
must begin exactly at the next uncommitted row. Exact replay of a committed
chunk is idempotent; gaps and conflicting replay are rejected.

## Finalization

A streamed file cannot finalize until every row of every planned tensor has
been committed. Finalization then re-verifies all chunks and opens the result
through the production `SafetensorsReader`.

## Official bounded gate

Official CI fetches only the first two BF16 rows of:

- `model.embed_tokens.weight`;
- `lm_head.weight`.

For the pinned Qwen3-Next checkpoint each row has 2,048 BF16 values, so CI
downloads exactly 16 KiB total.

The official rows are converted with the same 4-bit affine/group-64 path used
by production:

- packed columns per row: 256;
- groups per row: 32;
- two-row packed payload: 2,048 bytes;
- scales: 256 bytes;
- biases: 256 bytes.

This proves real checkpoint row/chunk conversion without downloading or
materializing either full vocabulary matrix.

## Next

**OSM-40C — full dense/global conversion orchestrator**

Combine the OSM-39A plan, streamed builder, official range-fetch tooling and
OSM-40A vector conversion into one restart-safe production workflow that emits
the complete `model.safetensors` inventory.
