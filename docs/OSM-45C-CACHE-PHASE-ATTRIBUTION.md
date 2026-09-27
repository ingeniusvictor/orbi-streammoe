# OSM-45C — cache hit/miss phase attribution

Status: **IMPLEMENTED / STACKED CI CERTIFICATION PENDING**

## Goal

Attribute routed-expert cache activity to the two runtime phases already timed by
OSM-45B: prompt prefill and autoregressive decode.

OSM-45C changes no cache policy. It only adds accounting.

## Runtime accounting

Before prompt execution, the token session snapshots:

- host expert-cache cumulative hits/misses;
- Vulkan resident expert-cache cumulative hits/misses/loads/evictions.

After prefill it records the delta from the initial snapshot.

After generation stops it records the delta from the post-prefill snapshot.

The result therefore contains:

    prefill_cache
    decode_cache

for these monotonic counters:

- host hits;
- host misses;
- GPU hits;
- GPU misses;
- GPU loads;
- GPU evictions.

Resident entries/bytes remain endpoint state in the existing probe output and
are deliberately not treated as additive phase deltas.

## Reconciliation gate

Because the production chat probe constructs fresh caches immediately before the
session, OSM-45C requires:

    prefill counter + decode counter == final probe counter

for every attributed counter.

The certification also derives phase-local host/GPU hit rates when the
denominator is non-zero.

## Evidence chain

OSM-45C consumes:

- OSM-45A performance evidence containing the measured chat probe;
- OSM-45B phase-latency certification bound to that exact OSM-45A file.

This joins time and cache activity without inventing a causal performance
conclusion.

## Claim boundary

A successful real run may claim:

- cache activity is attributed to prefill/decode;
- phase totals reconcile with final counters;
- phase-local hit rates are observed.

It does not claim:

- the cache policy is optimal;
- misses are the dominant latency cause;
- a performance target is met.

## Next

**OSM-46A — real Windows pilot execution pack**

Once the contract stack is canonical, package the exact commands and evidence
paths required to execute OSM-42D through OSM-45C on the designated Windows
machine without hand-editing intermediate JSON.
