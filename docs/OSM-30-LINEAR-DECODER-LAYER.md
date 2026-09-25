# OSM-30 — complete checkpoint-bound linear decoder layer

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Compose the two already-defined Qwen3-Next residual sublayers into one complete
single-token **linear-attention decoder layer**:

```text
x
│
├─────────────────────────────────────┐
│                                     │
▼                                     │
input RMSNorm                         │
│                                     │
▼                                     │
Gated DeltaNet                        │
│                                     │
└──────── residual add ───────────────┘
              │
              ▼
              h
              │
├─────────────────────────────────────┐
│                                     │
▼                                     │
post-attention RMSNorm                │
│                                     │
▼                                     │
checkpoint sparse MoE                 │
│                                     │
└──────── residual add ───────────────┘
              │
              ▼
             out
```

The exact semantic contract is:

```text
h   = x + DeltaNet(RMSNorm(x))
out = h + SparseMoE(RMSNorm(h))
```

## Runtime

`QwenCheckpointLinearDecoderLayer` owns:

- OSM-29C input-RMSNorm + stateful DeltaNet residual sublayer;
- OSM-28 post-attention-RMSNorm + sparse-MoE residual sublayer.

The routed expert cache remains external so hot experts can persist across
tokens and layers according to the existing cache policy.

## State

Only the Gated DeltaNet branch owns token-to-token recurrent state:

- causal Conv1D tail;
- gated-delta recurrent matrix.

`reset_state()` clears that state for a fresh sequence.

## Current backend boundary

This gate is correctness-first:

- both RMSNorm operations execute with Vulkan;
- routed/shared MoE execution uses the existing Vulkan path;
- Gated DeltaNet math remains the OSM-29A CPU oracle;
- residual composition currently crosses host memory.

The layer is therefore semantically complete before performance fusion.

## Exit gate

OSM-30 is GREEN when an end-to-end qpack fixture proves:

- both checkpoint norm vectors are used;
- DeltaNet state persists across at least two tokens;
- sparse MoE routed experts remain streamed/cached;
- complete output equals explicit composition of independently certified
  OSM-29C + OSM-28 components;
- reset reproduces fresh-session behavior;
- invalid hidden geometry is rejected;
- Linux CI executes both Vulkan norm/MoE stages with real Vulkan.

## Next

Two major tracks become possible after OSM-30:

1. **OSM-31 — full-attention/GQA decoder branch**, required for the 12 full
   attention layers in Qwen3-Next-80B-A3B.
2. **Vulkan DeltaNet optimization**, replacing the CPU correctness path while
   retaining OSM-29A as the semantic oracle.

Only after both DeltaNet and GQA decoder branches exist should a multi-layer
Qwen3-Next model runtime be considered complete.
