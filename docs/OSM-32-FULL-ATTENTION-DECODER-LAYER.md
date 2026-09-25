# OSM-32 — complete checkpoint-bound full-attention decoder layer

Status: **IMPLEMENTED / CLEAN CI CERTIFICATION PENDING**

## Goal

Compose the GQA residual branch and sparse-MoE residual branch into the second
complete Qwen3-Next decoder-layer variant:

```text
h   = x + GQA(RMSNorm(x))
out = h + SparseMoE(RMSNorm(h))
```

Together with OSM-30, this gives ORBI both layer families needed by the
Qwen3-Next alternating decoder topology.

## Runtime

`QwenCheckpointFullAttentionDecoderLayer` owns:

- OSM-31C checkpoint input-RMSNorm + gated-GQA residual sublayer;
- OSM-28 checkpoint post-attention-RMSNorm + sparse-MoE residual sublayer.

The routed expert cache remains external so hot experts can persist across
tokens/layers according to the existing cache policy.

## State

The full-attention branch owns persistent, growing decode state:

- position;
- K cache;
- V cache.

`reset_state()` clears that state for a fresh sequence.

## Current backend boundary

This is correctness-first:

- both RMSNorm stages execute on Vulkan;
- routed/shared MoE uses the existing Vulkan path;
- GQA uses the OSM-31A CPU semantic oracle;
- residual boundaries cross host memory.

## Exit gate

OSM-32 is GREEN when an end-to-end qpack fixture proves:

- both checkpoint norm vectors are used;
- Q/K/V/O and Q/K norm tensors come from the checkpoint;
- K/V cache persists across two tokens;
- sparse-MoE routed experts stream/cache for the full-attention layer;
- complete output matches explicit composition of independent OSM-31C + OSM-28
  references;
- reset reproduces fresh-session behavior;
- invalid hidden geometry is rejected;
- Linux CI executes the Vulkan RMSNorm/MoE portions with real Vulkan.

## Next

**OSM-33 — unified alternating decoder layer/runtime**

Introduce one layer wrapper that selects:

- Gated DeltaNet for 36 linear-attention layers;
- gated GQA for 12 full-attention layers;

according to `full_attention_interval`.

That becomes the building block for an actual multi-layer Qwen3-Next decoder.


## Clean-base certification

After OSM-31C was squash-merged, this branch was rebuilt directly from the new
canonical `main`. The final CI run therefore validates only the OSM-32
full-attention decoder-layer delta.
