# OSM-46B — Windows pilot evidence seal

Status: **IMPLEMENTED / CI CONTRACT CERTIFICATION PENDING**

## Goal

Seal the output of a completed OSM-46A designated-host Windows pilot into one
immutable, auditable certification.

OSM-46B executes no model inference. It consumes evidence already produced by
the real OSM-46A run.

## Required chain

The seal requires the OSM-46A state to report all nine phases complete, in the
canonical order:

1. OSM-42D Windows conversion certification;
2. OSM-43A runtime load;
3. OSM-43B first token;
4. OSM-43C text first token;
5. OSM-43D bounded generation;
6. OSM-44C chat generation;
7. OSM-45A performance instrumentation;
8. OSM-45B phase latency;
9. OSM-45C cache attribution.

Each phase file is bound by absolute path, stage and SHA256.

## Cross-evidence invariants

The seal verifies:

- official model: `Qwen/Qwen3-Next-80B-A3B-Instruct`;
- pinned snapshot: `f5e99a3698d364cf77584543481b778afee26177`;
- the OSM-46A state SHA binding to the exact execution pack;
- exact nine-phase completion;
- checkpoint directory consistency;
- full package SHA256 consistency;
- final cache evidence path consistency.

## Metric summary

The final seal carries observed values from the real pilot:

- cold end-to-end wall time;
- sampled process peak RSS;
- generated/prompt token counts;
- cold end-to-end output rate;
- prefill and decode latency;
- host/GPU cache phase counters and hit-rate evidence.

These are observations from that pilot only.

## Claim boundary

A valid real OSM-46B seal may claim:

- the designated Windows pilot completed;
- all nine phase artifacts are sealed;
- the official model/snapshot and checkpoint package are bound;
- performance, phase latency and cache activity were observed.

It does **not** claim:

- representative steady-state throughput;
- portability of measured performance to other hosts;
- a target tokens/s value was achieved;
- that the cache policy is optimal.

## CI boundary

CI uses synthetic evidence fixtures to validate schema, hash binding,
idempotence, tamper rejection and claim boundaries. CI does not claim a real
80B Windows pilot occurred.

## Next

**OSM-46C — designated-host pilot review / optimization baseline**

Consume the real OSM-46B seal and derive a human-readable baseline identifying
where time and cache pressure were observed, without inferring causality beyond
the recorded evidence.
