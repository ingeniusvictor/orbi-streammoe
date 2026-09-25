# OSM-29C — checkpoint input-RMSNorm + DeltaNet residual sublayer

Status: **IMPLEMENTED / CLEAN CI CERTIFICATION PENDING**

## Goal

Assemble the first half of a Qwen3-Next linear-attention decoder layer:

```text
x
│
├──────────────────────────┐
│                          │
▼                          │
input_layernorm.weight     │
│                          │
Vulkan RMSNorm             │
│                          │
▼                          │
OSM-29B checkpoint DeltaNet│
│                          │
└────── residual add ◄─────┘
           │
           ▼
           h
```

The semantic contract is:

`h = x + DeltaNet(RMSNorm(x))`.

## State

The sublayer owns the OSM-29B persistent DeltaNet state:

- causal Conv1D tail;
- gated-delta recurrent matrix.

`reset_state()` starts a new decode session.

## Correctness boundary

This is still a correctness-first mixed backend:

- input RMSNorm executes through Vulkan;
- the checkpoint-bound DeltaNet branch currently executes through the OSM-29A
  CPU oracle;
- residual addition is on the host.

That boundary is intentional until the DeltaNet Vulkan kernels are certified.

## Exit gate

OSM-29C is GREEN when an end-to-end qpack fixture verifies:

- checkpoint `input_layernorm.weight`;
- Vulkan RMSNorm parity against CPU RMSNorm;
- two stateful checkpoint DeltaNet tokens;
- exact residual composition;
- state persistence across tokens;
- reset-state behavior;
- invalid residual shape rejection;
- explicit real Vulkan execution on Linux CI.

## Next

Once OSM-29C is GREEN, combine its output `h` with OSM-28:

```text
h   = x + DeltaNet(RMSNorm(x))
out = h + SparseMoE(RMSNorm(h))
```

That becomes the first complete **linear-attention Qwen3-Next decoder layer**
assembled from checkpoint-bound components.


## Clean-base certification

After OSM-29B was squash-merged, this branch was rebuilt directly from the new
canonical `main` so the final CI run validates only the OSM-29C delta rather
than inherited stacked commits.
