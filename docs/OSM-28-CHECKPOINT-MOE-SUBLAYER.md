# OSM-28 — checkpoint-bound post-attention MoE sublayer

Status: **IMPLEMENTED / FULL CHECKPOINT CERTIFICATION PENDING**

## Goal

Assemble the first decoder sublayer from exact Qwen3-Next checkpoint bindings:

```text
residual
   │
   ├─────────────────────────────────────┐
   │                                     │
   ▼                                     │
post_attention_layernorm.weight          │
   │                                     │
Vulkan RMSNorm                           │
   │                                     │
   ▼                                     │
OSM-27 checkpoint-bound sparse MoE       │
   │                                     │
   └────────────── residual add ◄────────┘
                   │
                   ▼
                 output
```

## Runtime object

`QwenCheckpointMoeSublayer` owns:

- the checkpoint post-attention RMSNorm vector;
- RMS epsilon;
- the complete `QwenCheckpointSparseMoeLayer` from OSM-27.

The routed expert cache remains externally owned so it can be shared across
decoder execution and retain hot experts across tokens.

## Correctness boundary

The current path is correctness-first:

1. RMSNorm executes on Vulkan and returns the normalized host vector;
2. OSM-27 executes the checkpoint-bound sparse MoE;
3. residual addition is performed on the host.

This deliberately introduces host traffic between the normalization and MoE
stages. A later persistent-activation gate should fuse this boundary after the
decoder semantics are complete.

## Exit gate

OSM-28 code is structurally complete when:

- the post-attention norm comes from the same exact checkpoint layer binding;
- malformed norm geometry and invalid epsilon are rejected;
- the MoE runtime is created from the same binding;
- execution is exactly `residual + MoE(RMSNorm(residual))`;
- residual shape mismatch is rejected.

Full GREEN certification requires an end-to-end qpack fixture test reusing the
OSM-27 checkpoint/expert fixture. That test should compare the whole sublayer to
an independent CPU oracle on Windows/Linux and real Vulkan on Linux.

## Next

After full OSM-28 certification:

**OSM-29 — input RMSNorm + attention/DeltaNet residual branch**

That adds the first half of the decoder layer:

```text
h = x + attention_or_deltanet(RMSNorm(x))
out = h + MoE(RMSNorm(h))
```

and brings ORBI StreamMoE to a complete single-layer Qwen3-Next execution
contract.
