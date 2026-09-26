# OSM-42B — Windows full-conversion hardware preflight / authorization pack

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Convert the reproducible OSM-42A official rehearsal into an explicit,
machine-specific authorization decision for the first full official
Qwen3-Next-80B-A3B conversion on Windows.

OSM-42B does not start conversion. It answers a narrower question:

> Is this exact Windows host currently authorized to begin the already rehearsed
> production execution?

## Host probe

Run on the intended Windows machine:

    python scripts/probe_qwen_windows_host.py ^
      --target-output D:\orbi\qwen80b\checkpoint ^
      --expert-work D:\orbi\qwen80b\expert-work ^
      --dense-work D:\orbi\qwen80b\dense-work ^
      --output D:\orbi\qwen80b\windows-host-probe.json

The probe records:

- Windows version, architecture and Python version;
- total and currently available physical RAM;
- target/work path existence and writable nearest ancestor;
- filesystem/drive identity and current free/total bytes;
- `vulkaninfo --summary` evidence, SHA-256 and adapter marker lines.

Execution uses argv arrays with `shell=False`. No shell command is constructed.

A previously captured Vulkan summary can be supplied with
`--vulkan-summary-file`. That is useful for controlled evidence collection,
but the authorization pack still binds the summary bytes by SHA-256.

## Authorization

The machine authorization step is:

    python scripts/authorize_qwen_windows_conversion.py ^
      --rehearsal-bundle D:\orbi\qwen80b\rehearsal\rehearsal-bundle.json ^
      --host-probe D:\orbi\qwen80b\windows-host-probe.json ^
      --min-available-memory-bytes <operator-policy> ^
      --output D:\orbi\qwen80b\conversion-authorization.json

The RAM floor is intentionally explicit operator policy rather than an invented
project constant. The effective minimum can never be below the immutable
`max_batch_source_bytes` in the OSM-41A production manifest.

The disk gate uses the manifest's existing
`peak_required_free_bytes` requirement. For correctness-first authorization,
every distinct volume used by the output/expert/dense paths must independently
have at least that much free space. This is conservative when work directories
are split across drives, but cannot understate the preflight requirement.

## Cryptographic bindings

The immutable authorization pack binds:

- official model and pinned snapshot;
- OSM-42A rehearsal bundle SHA-256;
- OSM-41A execution ID and manifest SHA-256;
- OSM-41D exact command-plan SHA-256;
- Windows host-probe SHA-256;
- RAM and disk gate values;
- filesystem/drive IDs;
- Vulkan summary SHA-256 and adapter markers.

Any change to the rehearsal manifest/command plan or host probe requires a new
authorization pack.

## Resume/path semantics

Existing output or work directories are allowed because OSM-39/40/41 already
provide restart-safe journals and durable operator recovery.

OSM-42B does not delete, truncate or initialize those directories. It verifies
only that an existing target is a directory and that the nearest existing
ancestor is writable. Actual journals remain authoritative when production
execution begins.

## Safety boundary

An OSM-42B pack explicitly records:

- `conversion_started = false`;
- `operator_state_initialized = false`;
- `authorization_is_machine_specific = true`;
- `full_inference_claimed = false`.

CI may certify the probing/authorization logic and exercise native Windows
memory/disk APIs, but a GitHub-hosted runner is **not** the ORBI production
machine and must not be presented as the final hardware authorization.

## Exit gate

OSM-42B is GREEN when Windows and Ubuntu prove the portable contract tests and
Windows additionally proves the native host probe, while inherited production,
checkpoint and Vulkan gates remain GREEN.

The contract tests must prove:

- OSM-42A bundle digest and official source pin validation;
- execution-manifest and exact command-plan hash binding;
- Windows-only authorization;
- available RAM enforcement;
- peak free-disk enforcement;
- writable target/work paths;
- Vulkan evidence requirement;
- immutable authorization output;
- tamper/conflict rejection.

## Next

**OSM-42C — first full official Windows conversion runbook / guarded start**

Consume a real OSM-42B authorization pack on the designated Windows host and
provide a guarded start command for the OSM-41E launcher. The start must verify
that host/rehearsal/manifest bindings have not changed immediately before
initializing durable operator state, then execute through explicit resumable
checkpoints rather than claiming success before the full package exists.
