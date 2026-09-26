# OSM-41A — immutable production execution manifest

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Create the single immutable operator preflight that must exist before a
full-scale Qwen3-Next-80B-A3B conversion is authorized.

OSM-41A binds the certified expert, dense and final-package workflows to one
source snapshot and one set of operational budgets.

## Manifest contract

The execution manifest records:

- model and pinned snapshot;
- SHA-256 of source `config.json`;
- SHA-256 of the canonical dense stream inventory;
- output, expert-work and dense-work paths;
- affine-Q4 group size;
- complete layer and expert ranges;
- expert geometry and exact final expert payload bytes;
- dense tensor count and estimated final dense payload bytes;
- dense chunk size/count and BF16 batch budget;
- minimum free disk reserve;
- peak required free disk;
- exact planner/fetcher/converter/finalizer component names;
- deterministic execution ID.

## Full-scope guard

This gate is specifically for authorizing a **full production conversion**.

It rejects partial expert or layer ranges. Bounded pilots remain valid in the
earlier OSM-39/40 gates but cannot be mislabeled as the production execution
manifest.

## Immutability

If the requested manifest path already exists:

- byte-identical content is accepted as idempotent restart;
- any change is rejected.

Changing snapshot, disk budget, chunk size, output path or any other bound
execution parameter therefore creates a new operator plan rather than silently
mutating an authorized run.

## Disk model

The preflight reserves:

```text
all routed-expert QPACK payload
+ estimated dense/global payload
+ conservative safetensors header reserve
+ maximum one-batch BF16 scratch
+ operator minimum free-space reserve
```

For the pinned production model the expert component is independently
certified by OSM-39G as 48,318,382,080 bytes.

## Dependency gate

OSM-41A is built directly on the OSM-40F canonical full-checkpoint readiness
gate and must retain all inherited Windows, Ubuntu, tokenizer, checkpoint and
real-Vulkan regressions.

## Next

**OSM-41B — operator runner / phase state machine**

Consume the immutable OSM-41A manifest and execute the expert, dense and final
package phases with restart-safe phase transitions and explicit operator
checkpoints.
