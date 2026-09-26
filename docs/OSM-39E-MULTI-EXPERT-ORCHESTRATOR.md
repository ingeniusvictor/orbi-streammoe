# OSM-39E — bounded multi-expert production orchestrator

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Connect the certified OSM-39C single-expert BF16→Q4 converter to the OSM-39D
resumable fixed-stride layer writer.

The orchestrator converts a requested expert interval one expert at a time and
never materializes a whole routed-expert layer in memory.

## Flow

```text
verified expert manifests
        ↓
requested [first,end) range
        ↓
journal says complete? ── yes → verify + skip source read
        │ no
        ↓
OSM-39C convert one expert
        ↓
one QPACK expert blob
        ↓
OSM-39D write + readback + journal
        ↓
next expert
```

## Memory invariant

Peak conversion memory scales with one BF16 expert plus one QPACK expert blob,
not with 512 experts or a complete model layer.

For the pinned official geometry each source expert is three 2 MiB BF16
matrices (6 MiB total), while one QPACK expert blob is 1,966,080 bytes.

## Resume semantics

Before reading any source tensor payload, the orchestrator checks the OSM-39D
journal. A completed expert is verified and skipped. Therefore restart does not
re-download/re-quantize already certified experts when the tooling layer only
fetches missing manifests.

## Network boundary

The C++ core does not perform HTTP. Official bounded ranges are fetched by
`scripts/fetch_qwen_expert_range.py`, preserving the existing auditable
range-request tooling boundary.

## CI gate

Portable CI certifies:

- bounded expert interval selection;
- sequential one-expert-at-a-time conversion;
- exact fixed-stride commits;
- restart skips completed experts;
- aggregate progress/source-byte accounting;
- missing-manifest guard.

Official checkpoint CI additionally downloads experts 0 and 1 only and proves:

- exactly 12,582,912 BF16 source bytes are fetched;
- both experts convert into the production QPACK geometry;
- the production-size layer file is preallocated exactly;
- restart skips both experts without reading source BF16 again.

This remains intentionally far below a full 512-expert layer conversion.

## Next

**OSM-39F — production conversion CLI / missing-expert fetch planner**

Expose layer/range selection, inspect journals before network fetch, generate
the exact missing-expert download set, and provide restart-safe operator
progress for full offline layer conversion.
