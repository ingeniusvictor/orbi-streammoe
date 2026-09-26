# OSM-39F — production conversion CLI and missing-expert planner

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

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

- primary journal is consulted before network fetch;
- missing primary journals fall back to the OSM-39D `.bak` recovery journal;
- already-completed experts are excluded from the missing set;
- arbitrary non-contiguous missing expert IDs can be fetched;
- shard-header probing accepts dynamic layer/expert selections;
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
already journal-certified. Payload downloads are therefore limited to the
missing expert set. Header probing remains bounded to the shards needed by the
selected tensor inventory.

## Certification gate

Windows and Ubuntu CI must prove:

- missing-expert planning from a fresh conversion;
- resume planning from a primary journal;
- recovery planning from a backup journal;
- rejection of out-of-range journal entries;
- explicit non-contiguous expert selection without list mutation;
- native production CLI compilation;
- all existing runtime and real-Vulkan regressions remain GREEN.

Official checkpoint CI additionally probes expert 2 through the dynamic header
selection path without downloading its full payload. This proves the production
header selector is not limited to the fixed experts 0 and 1 used by OSM-39E.

## Next

**OSM-39G — operator checkpoint packaging / layer manifest finalization**

Add deterministic generation of QPACK `layout.json`, layer inventory and
conversion provenance so converted layers can be assembled into a complete
runtime checkpoint without manual metadata editing.
