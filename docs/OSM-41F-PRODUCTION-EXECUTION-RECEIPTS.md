# OSM-41F — production execution receipt / audit ledger

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Bind the restart-safe OSM-41B operator state and immutable OSM-41D command plan
to durable evidence that can be audited after a long production conversion.

OSM-41F does not replace conversion journals or decide phase order. Those remain
authoritative. It records immutable evidence after the authoritative state says
a phase completed successfully.

## Receipt layout

By default, a state file named operator-state.json uses this sibling directory:

    operator-state.json.receipts/
      00-expert_conversion.json
      01-expert_finalization.json
      02-dense_conversion.json
      03-full_checkpoint.json
      completion.json

The --receipt-dir option can select another directory.

Each phase receipt binds:

- execution ID;
- immutable execution-manifest SHA-256;
- immutable command-plan SHA-256;
- exact shell-free argv and its canonical SHA-256;
- completed phase index/name;
- exit code;
- attempt count;
- interrupted retry count;
- SHA-256 of the previous phase receipt.

Receipts are write-once. Byte-identical recovery is accepted; conflicting
evidence is rejected.

## Crash reconciliation

A phase can finish and its OSM-41B state transition can be durable immediately
before the receipt is written.

On the next next or all invocation, OSM-41F first reads the authoritative
operator state and reconstructs any missing receipt in the completed phase
prefix before allowing the next phase to run.

This makes the ledger recoverable without making it the execution state
machine.

## Completion evidence

When full_checkpoint completes, OSM-41F requires an OSM-40F package with
packageStage = full-checkpoint.

The completion receipt binds:

- all four phase receipts and the chain-tail hash;
- final operator-state SHA-256;
- execution manifest and command-plan SHA-256;
- package manifest.json;
- dense conversion journal;
- every file declared by the QPACK package manifest;
- declared and actual byte counts;
- streaming SHA-256 for every declared file;
- one canonical aggregate package digest.

All large file hashes are computed in bounded chunks. No expert layer or
model.safetensors payload is loaded wholesale into RAM.

The package source checkpoint and pinned source snapshot must match the
immutable production execution manifest.

## Resume versus audit

Ordinary idempotent resume validates the existing immutable completion receipt
and its bindings without re-reading tens of gigabytes of package payload.

Explicit deep verification is available through:

    python scripts/launch_qwen_production.py \
      --manifest execution.json \
      --state operator-state.json \
      --bin-dir build/Release \
      audit

The audit action rehashes the completed package. A same-size payload mutation
therefore fails against the immutable completion receipt.

The lower-level audit entrypoint is also available:

    python scripts/record_qwen_production_receipts.py \
      --manifest execution.json \
      --state operator-state.json \
      --command-plan operator-state.json.commands.json \
      --verify-package

## Safety boundary

OSM-41F preserves:

- OSM-41A immutable authorization;
- OSM-41B durable phase state and restart recovery;
- OSM-41D exact argv / shell=false;
- OSM-41E explicit preview, next, and bounded/unbounded all;
- lower-level expert and dense resume journals;
- bounded conversion memory;
- bounded audit hashing memory.

Preview remains read-only and does not create receipts.

## Exit gate

OSM-41F is GREEN when Windows and Ubuntu prove:

- completed phases emit immutable receipts;
- exact argv hashes bind to the command plan;
- receipt hashes form the ordered phase chain;
- a missing receipt is reconstructed from authoritative completed state;
- completion requires all four phase receipts;
- the final package manifest is full-checkpoint;
- all declared package files are size-checked and streaming SHA-256 hashed;
- normal completed resume is idempotent and does not rewrite evidence;
- explicit deep audit detects same-size package tampering;
- receipt tampering/conflicts are rejected;
- inherited OSM-41E, checkpoint and real-Vulkan regressions remain GREEN.

The gate still does **not** claim that CI converted the complete 80B
checkpoint. Full official conversion remains a production operator action.

## Next

**OSM-42A — official production rehearsal / dry-run evidence**

Exercise the full OSM-41A→41F operator path against the pinned official Qwen
metadata and bounded official source slices, producing a reproducible rehearsal
bundle before authorizing the first full 80B conversion on Windows hardware.
