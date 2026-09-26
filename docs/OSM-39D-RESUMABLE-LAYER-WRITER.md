# OSM-39D — resumable fixed-stride QPACK layer writer

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Provide the deterministic storage primitive required to scale the certified
OSM-39C single-expert converter to all experts of one layer without risking
silent corruption or restarting a long conversion from zero.

OSM-39D does **not** download hundreds of experts in CI. It certifies the layer
assembly/resume contract independently from the already-certified real-expert
quantizer.

## Runtime-compatible layout

The existing QPACK reader addresses routed experts as:

```text
offset = expert_index * expertStride
```

and requires:

```text
layer_file_size = expertCount * expertStride
```

OSM-39D writes exactly that contract.

For the pinned official OSM-39B geometry:

```text
expertStride = 1,966,080 bytes
expertCount  = 512
layer bytes  = 1,006,632,960 bytes
             = 960 MiB
```

## Resume journal

Each layer has a sidecar journal containing:

- schema version;
- layer index;
- expert count;
- expert stride;
- completed expert IDs;
- FNV-1a 64 checksum for every committed expert.

The journal is rewritten through a temporary file and backup rotation. On open,
a missing primary journal can be recovered from the backup.

## Commit semantics

For one expert:

```text
exact-size expert blob
      ↓
seek expert_id * expertStride
      ↓
write + flush
      ↓
read back exact expert
      ↓
FNV checksum
      ↓
update resume journal
```

A completed expert is immutable:

- replaying identical bytes is idempotent;
- replaying different bytes is rejected;
- resume re-verifies every committed expert against the layer file.

This deliberately favors correctness over conversion throughput.

## Exit gate

OSM-39D is GREEN when Windows and Ubuntu CI prove:

- exact preallocated layer size;
- fixed-stride expert offsets;
- experts can be committed out of order;
- incomplete layers cannot finalize;
- restart restores completed-expert state;
- identical replay is idempotent;
- different replay is rejected;
- backup journal recovery works;
- expert readback checksums match;
- on-disk corruption is detected during resume;
- previous OSM-39C conversion and Vulkan regressions remain GREEN.

## Scope

The gate uses a tiny deterministic layer fixture. It does not make CI download
the ~3 GiB BF16 source payload required to convert all 512 experts of an
official layer.

## Next

**OSM-39E — bounded multi-expert production orchestrator**

Connect the OSM-39C official expert converter to the OSM-39D layer writer with
a configurable expert range, resumable downloads, per-expert progress, and a
small official multi-expert pilot before authorizing a full 512-expert layer
conversion outside CI.
