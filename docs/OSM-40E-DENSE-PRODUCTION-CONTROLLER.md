# OSM-40E — production dense conversion controller

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Automate the restart-safe OSM-40D operator loop without moving conversion,
checkpoint parsing, or HTTP semantics into a new implementation.

OSM-40E is a thin production controller around the already-certified:

- OSM-40D missing-row planner;
- OSM-40D strict BF16 row fetcher;
- OSM-40D native streamed conversion CLI;
- OSM-40B journal and deterministic safetensors builder.

## Controller loop

```text
read OSM-40B journal
      ↓
plan exact missing rows
      ↓
apply chunk-count + source-byte budget
      ↓
check output/scratch disk budget
      ↓
fetch only this batch
      ↓
native OSM-40D conversion
      ↓
journal commits verified rows
      ↓
delete successful batch scratch
      ↓
repeat
```

## Dry-run

`--dry-run` performs planning, progress calculation, output-size estimation and
batch-budget selection without:

- creating the work directory;
- performing network fetches;
- invoking the native converter;
- changing the output or journal.

This is the required preflight mode before a production-scale conversion.

## Disk controls

Two independent controls are exposed:

- `--max-batch-source-bytes` caps downloaded BF16 scratch per batch;
- `--min-free-disk-bytes` preserves an operator-defined free-space reserve.

When the final output does not yet exist, the controller also reserves the
exact converted tensor payload estimate plus a conservative 16 MiB safetensors
header allowance before permitting the first batch.

## Restart-safe batch reuse

A batch directory is keyed by a deterministic SHA-256 digest of its exact row
tasks.

If a crash happens after fetch but before successful conversion, restart sees
the same missing-row plan and reuses the existing `dense-stream.json` instead
of downloading the batch again.

A successful conversion removes the batch directory by default. Use
`--keep-batches` only for debugging.

## Progress

Every planning pass reports:

- dense tensor count;
- total source rows;
- completed rows;
- missing rows;
- completion percentage;
- current batch task count;
- current batch BF16 bytes;
- estimated final output payload bytes;
- required free disk.

`--max-batches` permits controlled operator windows while preserving the same
journal for the next invocation.

## Correctness boundary

OSM-40E does not implement quantization or safetensors writes. It only
coordinates certified components. Any subordinate command failure aborts the
controller before scratch cleanup so the operator can inspect/reuse the batch.

## Exit gate

Windows and Ubuntu CI must prove:

- exact output payload estimation for F32 and affine-Q4 targets;
- source-byte batch limiting;
- rejection when one row chunk exceeds the configured budget;
- deterministic batch identity;
- progress accounting;
- immutable persisted batch plans;
- dry-run has zero fetch/converter side effects;
- all previous conversion, tokenizer, and real-Vulkan gates remain GREEN.

## Next

**OSM-40F — production package controller / full checkpoint readiness**

Combine the completed dense/global controller with the OSM-39 expert package
workflow, validate whole-package disk requirements and inventory closure, and
produce a single restart-safe production conversion readiness gate before a
full Qwen3-Next-80B-A3B conversion is authorized.
