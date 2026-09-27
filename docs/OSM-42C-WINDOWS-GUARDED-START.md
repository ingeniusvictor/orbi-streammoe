# OSM-42C — first full official Windows conversion guarded start

Status: **IMPLEMENTED / CI CERTIFICATION PENDING**

## Goal

Bridge the machine-specific OSM-42B authorization pack into the first real,
checkpointed OSM-41E production action without weakening restart safety.

OSM-42C does **not** claim that the full Qwen3-Next-80B-A3B checkpoint has been
converted. It authorizes and starts exactly the first durable production phase.

## Fresh host evidence

Immediately before start, capture a new host probe on the designated Windows
machine using the exact output/work paths authorized by OSM-42B:

    python scripts/probe_qwen_windows_host.py ^
      --target-output D:\orbi\qwen80b\checkpoint ^
      --expert-work D:\orbi\qwen80b\expert-work ^
      --dense-work D:\orbi\qwen80b\dense-work ^
      --output D:\orbi\qwen80b\windows-host-probe-now.json

The fresh probe may have different available RAM or free-disk numbers, but must
still satisfy the authorized floors. Stable machine bindings must remain the
same: Windows architecture, total physical RAM, production volume IDs and
Vulkan summary/adapter evidence.

## Verify-only

Before mutating durable state:

    python scripts/guard_qwen_windows_start.py ^
      --authorization D:\orbi\qwen80b\conversion-authorization.json ^
      --current-host-probe D:\orbi\qwen80b\windows-host-probe-now.json ^
      --state D:\orbi\qwen80b\operator-state.json ^
      --bin-dir build\Release ^
      --verify-only

This verifies:

- authorization-pack digest and official model/snapshot pin;
- OSM-42A rehearsal bundle SHA-256;
- OSM-41A execution manifest SHA-256;
- OSM-41D command-plan SHA-256;
- original OSM-42B host-probe SHA-256;
- current RAM and disk gates;
- stable machine, volume and Vulkan bindings;
- absence of existing OSM-41B state or backup.

Verify-only never initializes operator state.

## Guarded first start

Run the same command without `--verify-only`:

    python scripts/guard_qwen_windows_start.py ^
      --authorization D:\orbi\qwen80b\conversion-authorization.json ^
      --current-host-probe D:\orbi\qwen80b\windows-host-probe-now.json ^
      --state D:\orbi\qwen80b\operator-state.json ^
      --bin-dir build\Release

After all guards pass, OSM-42C delegates to OSM-41E `execute_next`. Therefore
it initializes durable OSM-41B state and executes **exactly one** authorized
phase: `expert_conversion`.

It stops at the next explicit checkpoint. Subsequent recovery/resume uses the
existing OSM-41E `next` or bounded `all --max-phases N` commands; OSM-39/40
journals and OSM-41 state/receipts remain authoritative.

A second guarded *first start* is rejected when operator state or its backup
already exists. This prevents an authorization wrapper from silently becoming a
parallel resume mechanism.

## Safety boundary

OSM-42C preserves:

- no shell interpolation;
- immutable authorization/rehearsal/manifest/command bindings;
- fresh RAM/disk verification immediately before start;
- exact OSM-41D argv;
- OSM-41B durable phase order;
- OSM-41F receipt/audit integration;
- first-phase-only checkpointed start;
- explicit resume after interruption.

CI certifies the contract with fixtures on Windows and Ubuntu. GitHub-hosted
runners are not the final ORBI Windows hardware pilot.

## Exit gate

OSM-42C is GREEN when Windows and Ubuntu prove:

- verify-only is non-mutating;
- current host gates are re-evaluated;
- stable machine/Vulkan/volume bindings are enforced;
- authorization, rehearsal, manifest and command-plan tampering is rejected;
- exactly the first authorized phase executes;
- durable state is initialized only after all guards pass;
- repeated first-start attempts are rejected;
- inherited OSM-39 through OSM-42B gates remain GREEN.

## Next

**OSM-42D — full official Windows conversion execution / resumable evidence**

Run the authorized conversion on the designated Windows hardware through all
four production phases, preserving checkpoints and OSM-41F receipts, then
certify the resulting full checkpoint package before any runtime inference
claim.
