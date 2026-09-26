# OSM-41B — restart-safe operator runner / phase state machine

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Consume the immutable OSM-41A production execution manifest and provide a
durable operator state machine around the already-certified expert, dense and
full-checkpoint tools.

OSM-41B does not hide long-running conversion work behind an opaque shell. Each
production phase is explicit, ordered and restartable.

## Phase order

```text
expert_conversion
        ↓
expert_finalization
        ↓
dense_conversion
        ↓
full_checkpoint
        ↓
complete
```

The runner rejects attempts to skip ahead or rerun an already-completed
execution.

## Durable state

The operator state is bound to:

- OSM-41A `execution_id`;
- SHA-256 of the immutable execution manifest;
- fixed phase order.

Each phase records:

- status: `pending | running | failed | completed`;
- attempt count;
- last subprocess exit code;
- exact last command;
- interrupted-running retry count.

State writes use temporary-file + backup rotation. If the primary state file is
lost while a backup survives, the runner restores the backup before continuing.

## Restart semantics

The runner persists `running` **before** launching a phase command.

If the host/process dies after that write, the next invocation may rerun the
same phase. Such a retry is counted as an interrupted retry.

This is safe because the lower layers are already restart-safe:

- expert conversion: OSM-39D/39E/39F journals;
- dense conversion: OSM-40B/40C/40D/40E journals;
- package finalization: deterministic OSM-39G/40F gates.

## Command binding

A phase command must reference the exact component declared by the immutable
OSM-41A manifest. This prevents accidentally substituting a different
converter/finalizer while reusing an authorized execution state.

Commands are executed as argument vectors with `subprocess.run(..., shell=False)`.

## Dry run

`run --dry-run` validates:

- manifest binding;
- phase order;
- component binding;
- command shape;

without mutating operator state or launching the command.

## Exit gate

OSM-41B is GREEN when Windows and Ubuntu prove:

- idempotent initialization;
- immutable manifest SHA binding;
- phase-order enforcement;
- dry-run non-mutation;
- failed subprocess persistence and retry;
- interrupted-running recovery;
- backup state recovery;
- command/component binding;
- final completion gate;
- inherited Vulkan/checkpoint/tokenizer regressions remain GREEN.

## Next

**OSM-41C — production expert-phase controller**

Use the OSM-41B phase runner with a dedicated full-layer expert controller that
plans missing experts before network fetch, processes all layers under bounded
scratch/disk budgets, and can resume from existing OSM-39 journals.
