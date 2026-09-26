# OSM-39F — production conversion CLI and missing-expert planner

Status: **IMPLEMENTED / CI CERTIFICATION PENDING**

## Goal

Turn the certified OSM-39E orchestration primitives into an operator-facing,
restart-safe conversion workflow.

## Workflow

```text
existing layer journal
      ↓
plan_qwen_missing_experts.py
      ↓
exact missing expert IDs
      ↓
fetch_qwen_expert_range.py --experts ...
      ↓
verified local expert manifests/payloads
      ↓
orbi_streammoe_qpack_convert_range
      ↓
OSM-39E sequential conversion
      ↓
OSM-39D fixed-stride layer + journal
```

## Safety properties

- journal is consulted before network fetch;
- already-completed experts are excluded from the missing set;
- arbitrary non-contiguous missing expert IDs can be fetched;
- native conversion CLI uses the same certified OSM-39E core;
- a complete full-layer request finalizes only after all experts verify;
- partial ranges remain resumable and do not claim finalization.

## Production geometry

For Qwen3-Next-80B-A3B:

- 48 layers;
- 512 routed experts per layer;
- QPACK expert stride 1,966,080 bytes;
- layer file 1,006,632,960 bytes (960 MiB).

The planner allows interrupted conversions to fetch only experts that are not
already journal-certified.

## Next

**OSM-39G — operator checkpoint packaging / layer manifest finalization**

Add deterministic generation of QPACK `layout.json`, layer inventory and
conversion provenance so converted layers can be assembled into a complete
runtime checkpoint without manual metadata editing.
