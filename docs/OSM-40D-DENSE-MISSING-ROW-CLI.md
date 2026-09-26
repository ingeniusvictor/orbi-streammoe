# OSM-40D — dense/global missing-row fetch planner and production CLI

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Turn OSM-40C into a restart-safe production workflow that decides what BF16
data is still missing **before** making network requests.

The workflow keeps the canonical OSM-39A conversion plan as the source of truth
for which tensors are dense/global and how each tensor maps into
`model.safetensors`.

## Production workflow

```text
official config + safetensors index
        ↓
orbi_streammoe_qpack_list_dense_sources
        ↓
canonical dense source list
        ↓
fetch_qwen_shard_headers.py --tensor-list-json ...
        ↓
complete dense source shapes + offsets
        ↓
build_qwen_dense_stream_inventory.py
        ↓
OSM-40B resume journal
        ↓
plan_qwen_missing_dense_rows.py
        ↓
ONLY missing row chunks
        ↓
fetch_qwen_dense_row_plan.py
        ↓
OSM-40C dense-stream manifest
        ↓
orbi_streammoe_streamed_dense_convert
        ↓
streamed model.safetensors + journal
```

## Dense source inventory

`orbi_streammoe_qpack_list_dense_sources` reads the exact OSM-39A plan and
emits only:

- `global_dense`;
- `layer_dense`;

whose target is `model.safetensors`.

Routed experts and auxiliary MTP tensors cannot enter this inventory.

This prevents Python tooling from independently re-implementing the C++
checkpoint classification rules.

## Header enrichment

`fetch_qwen_shard_headers.py --tensor-list-json` can extend its bounded
selection with every source tensor emitted by the canonical dense inventory.

`build_qwen_dense_stream_inventory.py` then joins:

- canonical source tensor;
- source shard;
- conversion action;
- target path;
- complete BF16 shape;
- shard size/header size;
- exact safetensors data offsets.

Shard disagreement or BF16 geometry drift is rejected.

## Missing-row planner

`plan_qwen_missing_dense_rows.py` reads the OSM-40B journal, falling back to
`.bak` when needed.

For affine-Q4 tensors it requires equal progress across:

- `.weight`;
- `.scales`;
- `.biases`.

Divergent progress is treated as corruption rather than guessed around.

For every incomplete source tensor it emits exact BF16 byte ranges using:

```text
absolute offset
 = 8
 + safetensors header_size
 + tensor data_offset
 + first_row * source_row_bytes
```

`--max-chunks` bounds each operator batch.

## Fetch boundary

`fetch_qwen_dense_row_plan.py` accepts only plans whose model and snapshot
match the pinned official Qwen checkpoint.

It performs strict HTTP range requests and writes an OSM-40C manifest that
contains the **complete output inventory** while attaching source files only to
the currently fetched chunks.

Inventory tensors are therefore allowed to have zero source chunks in one
batch. This is required so OSM-40B can preplan the final safetensors layout once
and keep it stable across every restart.

## Native conversion CLI

`orbi_streammoe_streamed_dense_convert`:

1. rebuilds the canonical OSM-39A plan;
2. reads the complete OSM-40C manifest;
3. derives the deterministic target specs;
4. opens/reopens the OSM-40B builder;
5. executes only supplied missing chunks;
6. finalizes automatically only when every output row is certified.

## Correctness gates

Windows and Ubuntu CI certify:

- fresh missing-row planning;
- primary-journal resume;
- backup-journal recovery;
- affine triplet progress agreement;
- bounded `--max-chunks` batching;
- complete dense stream inventory construction;
- inventory-only OSM-40C manifests;
- native production CLI compilation;
- canonical dense source inventory compilation;
- all existing runtime and real-Vulkan regressions.

Official checkpoint CI additionally emits the dense source inventory directly
from the real Qwen OSM-39A plan and proves that:

- global and layer dense tensors are included;
- routed experts are excluded;
- only supported conversion actions appear;
- embedding, final norm and LM head are present.

## Operator loop

A full conversion can use repeated bounded batches:

```text
plan missing rows
→ fetch N chunks
→ convert N chunks
→ journal commit
→ repeat
```

A crash at any point restarts from the last verified OSM-40B row chunk.

## Next

**OSM-40E — production dense conversion controller**

Automate the plan/fetch/convert loop with disk-budget controls, per-batch
cleanup, progress reporting and a dry-run mode before authorizing complete
checkpoint conversion.
