# OSM-41D — production operator command builder

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Eliminate hand-written production phase commands.

OSM-41D converts the immutable OSM-41A execution manifest into the exact
argument vectors consumed by the OSM-41B restart-safe operator runner.

## Command plan

The generated plan binds all four phases:

```text
expert_conversion
  -> run_qwen_expert_conversion.py
  -> manifest-declared qpack range converter

expert_finalization
  -> manifest-declared expert package finalizer

dense_conversion
  -> manifest-declared dense controller
  -> manifest-declared streamed dense converter

full_checkpoint
  -> manifest-declared full-checkpoint finalizer
```

Every phase also receives a complete OSM-41B wrapper argv:

```text
python run_qwen_production_operator.py
  --manifest <immutable manifest>
  --state <durable state>
  run --phase <phase> --
  <exact phase argv...>
```

No shell string is generated. Paths remain individual argv entries.

## Expert batch derivation

OSM-41A already binds the maximum BF16 batch source budget. OSM-41D derives the
expert controller batch size from that immutable budget:

```text
source bytes per expert
  = 3 * hidden_size * moe_intermediate_size * sizeof(BF16)

max experts per batch
  = min(8, floor(max_batch_source_bytes / source_bytes_per_expert))
```

A manifest that cannot fit one source expert is rejected before execution.

## Immutability

The optional command-plan output is write-once:

- byte-identical regeneration is accepted;
- any conflicting regeneration is rejected.

The plan records the OSM-41A manifest SHA-256 and execution ID.

## Exit gate

OSM-41D is GREEN when Windows and Ubuntu prove:

- exact four-phase ordering;
- deterministic manifest -> argv construction;
- shell-free command vectors;
- paths containing spaces remain single argv entries;
- exact component-name binding;
- derived expert batch budget;
- dense chunk/batch parameters are inherited from OSM-41A;
- OSM-41B wrapper commands contain the exact phase argv;
- immutable command-plan behavior;
- inherited checkpoint and real-Vulkan regressions remain GREEN.

## Next

**OSM-41E — one-command production launcher**

Use the OSM-41D command plan to initialize OSM-41B state and execute or resume
the next authorized phase automatically while preserving explicit checkpoints
and dry-run visibility.
