# OSM-41C — production expert-phase controller

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Turn the OSM-39 production expert primitives into one restart-safe controller
that can process the complete routed-expert scope authorized by OSM-41A and be
executed as the `expert_conversion` command under OSM-41B.

## Journal-first rule

For every layer the controller reads:

```text
packed_experts/layer_XX.progress.json
```

before any network fetch.

Already-certified experts are excluded from the download plan. Missing experts
are split into **contiguous bounded batches**, so each invocation of
`orbi_streammoe_qpack_convert_range` receives an exact interval for which all
required source manifests exist.

## Batch flow

```text
read layer journal
      ↓
missing expert IDs
      ↓
contiguous batches (max N experts)
      ↓
dynamic shard-header manifest
      ↓
bounded BF16 range fetch
      ↓
OSM-39F range converter
      ↓
verify journal contains every batch expert
      ↓
clean batch scratch
```

## Resume behavior

If conversion stops:

- completed experts remain certified in the OSM-39D journal;
- the next controller run plans only missing IDs;
- a previously downloaded complete batch can be reused;
- source blobs are independently checksum-verified by OSM-39C;
- converter success is accepted only after the journal contains the batch.

## Memory and disk

The controller never requests more than `--max-experts-per-batch` source
experts. Per batch it reserves a new layer file when needed, BF16 source bytes
for the batch, and the OSM-41A minimum free-space reserve.

The canonical output remains:

```text
output/packed_experts/layer_XX.bin
output/packed_experts/layer_XX.progress.json
```

## Operator controls

- `--dry-run`: no network or subprocess execution;
- `--max-experts-per-batch`: bounded source batch;
- `--max-batches`: explicit operator checkpoint;
- `--max-layers`: bounded operator scope;
- `--keep-batches`: retain downloaded source inputs.

A limited run is never reported complete unless every expert journal across
every layer is complete.

## OSM-41B integration

The controller command includes the exact OSM-41A-declared converter path, so
OSM-41B's component-binding gate can authorize the wrapper without weakening
the immutable converter binding.

## Exit gate

OSM-41C is GREEN when Windows and Ubuntu prove:

- journal-first planning;
- holes become separate contiguous batches;
- bounded source-byte accounting;
- new/existing layer disk reserve logic;
- downloaded-batch reuse detection;
- dry-run performs no network execution;
- bounded one-batch execution;
- converter success must durably journal every requested expert;
- inherited checkpoint and real-Vulkan regressions remain GREEN.

## Next

**OSM-41D — operator command builder / production phase wiring**

Generate the exact OSM-41B commands for expert conversion, expert
finalization, dense conversion and full checkpoint finalization directly from
the immutable OSM-41A manifest.
