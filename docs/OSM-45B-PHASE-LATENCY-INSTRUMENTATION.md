# OSM-45B — phase-level latency instrumentation

Status: **IMPLEMENTED / STACKED CI CERTIFICATION PENDING**

## Goal

Split OSM-45A's cold end-to-end measurement into runtime phases that correspond
to real `QwenGreedyTokenSession` execution boundaries.

## Definitions

### Prompt prefill

Every `step_greedy(...)` invoked while teacher-forcing a prompt token belongs
to prefill.

The prediction produced by the final prompt step becomes generated token 0.

### Decode

Decode starts only when a generated token is fed back into `step_greedy(...)`
to produce a later generated token.

For `G` generated tokens there are therefore `G - 1` measured decode steps.

## Runtime evidence

`QwenGreedySessionResult` records:

- `prompt_prefill_ns`;
- `decode_ns`;
- one timing per prompt step;
- one timing per decode step.

The text-session facade propagates these fields unchanged and the production
chat probe emits them as `phase_latency`.

## Certification

OSM-45B requires:

    len(prompt_step_durations_ns) == prompt_token_count
    len(decode_step_durations_ns) == generated_token_count - 1
    sum(prompt_step_durations_ns) == prompt_prefill_ns
    sum(decode_step_durations_ns) == decode_ns

It derives average prefill-step and decode-step latency, but does not reinterpret
those numbers as a portable performance target.

## Claim boundary

A successful real run may claim phase latency was measured and accounted.

It still does not claim:

- representative steady-state throughput;
- a target tokens/s value;
- hardware portability of the observed latency;
- that optimization goals have been met.

## Next

**OSM-45C — cache hit/miss phase attribution**

Attribute host/Vulkan expert-cache activity to prefill vs decode so optimization
can distinguish I/O pressure from compute latency.
