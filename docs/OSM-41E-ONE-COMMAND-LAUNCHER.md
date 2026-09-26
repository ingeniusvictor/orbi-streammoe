# OSM-41E — one-command production launcher

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Provide the operator-facing entrypoint that combines OSM-41A immutable
production authorization, OSM-41B durable phase state and OSM-41D exact
phase-command generation.

OSM-41E does not replace lower-level tools. It selects the next authorized
phase and delegates execution through the certified OSM-41B state machine.

## Actions

### preview

Build and optionally persist the immutable OSM-41D command plan, then report
execution ID, state presence, completion state, next phase, exact phase argv
and exact OSM-41B wrapper argv.

Preview does not create or mutate the operator state.

### next

Initialize durable OSM-41B state if necessary and execute exactly one next
authorized phase. This is the checkpoint-friendly production workflow.

### all

Resume and execute every remaining authorized phase. The max-phases option
provides an explicit bounded checkpoint when an operator does not want to
cross more than N phase boundaries in one invocation.

## Restart semantics

Every actual phase still passes through OSM-41B. Phase order, durable status,
interrupted retries, manifest SHA-256 binding and lower-level expert/dense
journals therefore remain authoritative.

A completed execution makes next an idempotent no-op.

## Safety boundary

OSM-41E never constructs shell command strings. It consumes OSM-41D argv
vectors and uses the OSM-41B shell-false subprocess path.

## Exit gate

OSM-41E is GREEN when Windows and Ubuntu prove:

- preview does not initialize operator state;
- immutable command-plan reuse;
- next executes exactly one authorized phase;
- subsequent invocation resumes at the next phase;
- bounded all checkpointing;
- unbounded all completes remaining phases;
- completed preview reports no next command;
- completed next is idempotent;
- inherited checkpoint and real-Vulkan regressions remain GREEN.

## Next

**OSM-41F — production execution receipt / audit ledger**

Record immutable phase receipts, command hashes, execution ID, output package
hashes and final completion evidence without weakening restart-safe journals.
