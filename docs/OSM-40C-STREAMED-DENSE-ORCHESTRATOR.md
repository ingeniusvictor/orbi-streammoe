# OSM-40C — restart-safe dense/global conversion orchestrator

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Connect the complete OSM-39A dense/global conversion plan to the OSM-40B
streamed safetensors builder without ever materializing a production-scale
matrix in memory.

## Execution model

A verified source manifest describes:

- the source tensor name;
- the complete source shape;
- one or more local BF16 row-chunk files;
- exact row ranges;
- byte counts;
- FNV-1a checksums.

The orchestrator maps each source tensor through the OSM-39A plan and derives
the exact output inventory:

- `copy_bf16_to_f32` → one F32 target tensor;
- `affine_quantize` → U32 packed weight + F32 scales + F32 biases.

The resulting specs are opened by OSM-40B before conversion begins, so output
offsets remain deterministic.

## Restart behavior

Before any BF16 source file is read, the orchestrator inspects builder progress.

If an entire source chunk has already been certified, it is skipped without
reading or decoding the source payload.

Partial overlaps and gaps are rejected.

## Memory invariant

Peak conversion memory scales with one configured source row chunk plus its
converted output rows.

It does not scale with:

- vocabulary size;
- number of dense tensors;
- complete model.safetensors size.

## CI strategy

Portable fixture CI certifies the complete execution path with both:

- BF16 → F32 vector streaming;
- BF16 → affine-Q4 matrix streaming.

The fixture also certifies resume-without-source-read by deleting already
converted source chunks before reopening the builder.

Official checkpoint CI continues to use the real two-row embedding/LM-head
probe introduced in OSM-40B. This validates production Qwen shapes and Q4 row
conversion without forcing Actions to preallocate the complete production
`model.safetensors`.

## Next

**OSM-40D — dense/global missing-row fetch planner + production CLI**

Inspect the OSM-40B journal first, calculate only missing row intervals, fetch
those exact BF16 ranges from the pinned official shards, and feed OSM-40C
without redundant network traffic.
