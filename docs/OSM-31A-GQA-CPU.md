# OSM-31A — Qwen3-Next gated GQA CPU decode semantics

Status: **IMPLEMENTED / CLEAN CI CERTIFICATION PENDING**

## Goal

Freeze the exact single-token full-attention behavior used by every fourth
Qwen3-Next decoder layer before checkpoint binding and Vulkan optimization.

Qwen3-Next-80B-A3B alternates 36 Gated DeltaNet layers with 12 full
grouped-query-attention layers.

## Frozen semantics

The CPU oracle matches the frozen Swiftlet path:

1. `q_proj` emits `[query | output_gate]` per query head;
2. `k_proj` and `v_proj` emit KV-head tensors;
3. per-head Q/K RMSNorm with checkpoint norm vectors;
4. partial NeoX-style RoPE on the first configured rotary dimensions;
5. append K/V to the persistent decode cache;
6. causal scaled-dot-product attention over all cached positions;
7. grouped-query mapping `query_head -> kv_head`;
8. elementwise `sigmoid(output_gate)` on the attended value;
9. output projection back to hidden size.

## State

`QwenGqaState` contains:

- current decode position;
- K cache in `[position, kv_head, head_dim]`;
- V cache in the same layout.

Unlike DeltaNet's fixed recurrent state, full attention grows with context.

## Certification

A deterministic two-token vector checks:

- token 1 output;
- token 2 output after attending over both cached positions;
- exact K cache after partial RoPE;
- exact V cache;
- position advancement;
- grouped-query head validation;
- max-context rejection.

The expected numeric vectors were generated independently from the frozen
Swiftlet equations.

## Exit gate

OSM-31A is GREEN when Windows and Linux compile/test and all output/cache
reference vectors match within tolerance.

## Next

**OSM-31B — checkpoint-bound GQA**

Extend the checkpoint config contract with RoPE/context fields, bind the exact
`self_attn.q_proj/k_proj/v_proj/o_proj` plus q/k norm tensors, and execute this
stateful GQA oracle directly from qpack checkpoint data.


## Clean-base certification

After OSM-30 was squash-merged, this branch was rebuilt directly from the new
canonical `main`. The final CI run validates only the OSM-31A GQA semantic
oracle delta.
