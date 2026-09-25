# OSM-31C — checkpoint input-RMSNorm + GQA residual sublayer

Status: **IMPLEMENTED / CLEAN CI CERTIFICATION PENDING**

## Goal

Assemble the first half of a Qwen3-Next full-attention decoder layer:

```text
x
│
├─────────────────────────┐
│                         │
▼                         │
input_layernorm.weight    │
│                         │
Vulkan RMSNorm            │
│                         │
▼                         │
OSM-31B gated GQA         │
│                         │
└───── residual add ◄─────┘
          │
          ▼
          h
```

Semantic contract:

`h = x + GQA(RMSNorm(x))`.

## State

The sublayer owns the persistent GQA state:

- decode position;
- K cache;
- V cache.

`reset_state()` starts a fresh sequence.

## Current backend boundary

This is correctness-first:

- input RMSNorm executes through Vulkan;
- checkpoint-bound GQA executes through the OSM-31A CPU oracle;
- residual addition is on the host.

## Exit gate

OSM-31C is GREEN when the full-attention qpack fixture proves:

- checkpoint input norm is used;
- Vulkan RMSNorm matches CPU RMSNorm;
- two sequential GQA tokens match a direct checkpoint-GQA reference;
- K/V cache and position persist exactly;
- reset reproduces fresh-session behavior;
- invalid hidden geometry is rejected;
- Linux CI executes the RMSNorm stage through real Vulkan.

## Next

**OSM-32 — complete full-attention decoder layer**

Combine this sublayer with OSM-28:

```text
h   = x + GQA(RMSNorm(x))
out = h + SparseMoE(RMSNorm(h))
```

OSM-32 will give ORBI both decoder-layer variants needed by the alternating
Qwen3-Next topology.


## Clean-base certification

After OSM-31B was squash-merged, this branch was rebuilt directly from the new
canonical `main`. The final CI run therefore validates only the OSM-31C
full-attention residual-sublayer delta.
