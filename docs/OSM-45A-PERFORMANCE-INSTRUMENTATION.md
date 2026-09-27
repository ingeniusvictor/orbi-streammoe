# OSM-45A — runtime performance instrumentation contract

Status: **IMPLEMENTED / STACKED CI CERTIFICATION PENDING**

## Goal

Add portable measurements around the already-certified OSM-44C chat generation
probe without turning one cold run into a performance claim.

OSM-45A measures evidence. It does not optimize the runtime.

## Measurements

The runner records:

- end-to-end wall time using a monotonic clock;
- sampled child-process RSS;
- generated-token count;
- prompt-token count;
- cold end-to-end generated tokens per second;
- cold end-to-end milliseconds per generated token;
- host expert-cache budget;
- Vulkan expert-cache resident bytes;
- Vulkan expert-cache budget.

Process RSS sampling uses native OS facilities:

- Windows: `GetProcessMemoryInfo`;
- Linux/Android-like hosts: `/proc/<pid>/status`.

No third-party monitoring package is required.

## Metric semantics

`cold_end_to_end_generated_tokens_per_second` includes:

- process startup;
- checkpoint/runtime initialization;
- prompt rendering/tokenization inside the probe;
- prompt teacher forcing;
- bounded generation.

It is intentionally **not** equivalent to steady-state decode tokens/s.

## Evidence chain

The measurement requires an immutable successful OSM-44C certification for the
same checkpoint directory and invokes the production probe with `shell=False`.

## Claim boundary

A successful real OSM-45A run may claim:

- cold wall time measured;
- process RSS sampled;
- cache residency reported;
- a cold end-to-end output rate measured.

It does not claim:

- representative steady-state performance;
- a target tokens/s value;
- total device VRAM usage;
- that any optimization target has been met.

## CI boundary

CI exercises the instrumentation with a real child process that allocates
memory and sleeps long enough for RSS sampling. CI does not benchmark Qwen 80B.

## Next

**OSM-45B — phase-level latency instrumentation**

Instrument checkpoint load, prompt prefill and decode phases separately so
optimization work can target the actual bottleneck rather than the cold-run
aggregate.
