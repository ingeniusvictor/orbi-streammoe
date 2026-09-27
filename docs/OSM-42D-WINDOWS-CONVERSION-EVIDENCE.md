# OSM-42D — full official Windows conversion execution / resumable evidence

Status: **IMPLEMENTED / CI CONTRACT CERTIFICATION PENDING**

## Goal

Provide the final evidence gate for the first real full Qwen3-Next-80B-A3B
conversion on the designated Windows machine.

OSM-42D does not perform the large conversion in GitHub Actions. The actual
conversion remains a production operator action on the authorized Windows host.
This milestone defines how completion becomes machine-verifiable and
non-ambiguous after all four phases finish.

## Real Windows production flow

The designated Windows machine first completes OSM-42B authorization and the
OSM-42C guarded first start. After that, resume only through the existing
OSM-41E launcher:

    python scripts/launch_qwen_production.py ^
      --manifest D:\orbi\qwen80b\execution.json ^
      --state D:\orbi\qwen80b\operator-state.json ^
      --bin-dir build\Release ^
      next

Use repeated `next` invocations for explicit phase boundaries, or a bounded:

    all --max-phases 1

The authoritative order remains:

    expert_conversion
      -> expert_finalization
      -> dense_conversion
      -> full_checkpoint

OSM-39/40 journals, OSM-41B durable state and OSM-41F receipts remain the
restart sources of truth. A process interruption does not require restarting
completed conversion work.

## Final certification

Only after OSM-41B reports every phase completed and OSM-41F deep audit verifies
the package, run:

    python scripts/certify_qwen_windows_conversion.py ^
      --authorization D:\orbi\qwen80b\conversion-authorization.json ^
      --manifest D:\orbi\qwen80b\execution.json ^
      --state D:\orbi\qwen80b\operator-state.json ^
      --command-plan D:\orbi\qwen80b\operator-state.json.commands.json ^
      --output D:\orbi\qwen80b\windows-conversion-certification.json

The certification requires:

- the immutable OSM-42B machine authorization;
- the pinned official model and snapshot;
- exact execution-manifest and command-plan hashes;
- completed OSM-41B state for all four phases;
- zero exit code for every completed phase;
- the complete OSM-41F receipt chain;
- an OSM-41F deep package re-audit;
- a `packageStage = full-checkpoint` package;
- streaming hashes for all declared package files;
- the aggregate package digest.

## What OSM-42D is allowed to claim

A valid certification may claim:

- full official checkpoint conversion completed;
- all four production phases completed;
- full checkpoint package deep-audited;
- package digest frozen as immutable evidence.

It deliberately does **not** claim:

- that the resulting checkpoint has loaded in the ORBI runtime;
- prompt-to-token end-to-end inference;
- multi-token generation;
- RAM/VRAM baseline during inference;
- tokens per second;
- Windows hardware inference certification.

Those belong to the next runtime-validation milestone.

## CI boundary

Windows and Ubuntu CI validate the certification logic using bounded fixtures.
CI must never be described as having converted the actual 80B model.

The real OSM-42D exit gate is satisfied only when the designated Windows host
produces a valid `windows-conversion-certification.json` from a real completed
production execution.

## Next

**OSM-43A — official full-checkpoint runtime load gate**

Load the OSM-42D-certified package into the ORBI StreamMoE runtime and prove the
first real checkpoint-backed model initialization before claiming prompt
inference.
