# OSM-31B — checkpoint-bound gated GQA

Status: **IMPLEMENTED / CLEAN CI CERTIFICATION PENDING**

## Goal

Bind exact Qwen3-Next full-attention checkpoint tensors to the stateful OSM-31A
gated GQA CPU oracle.

## Config extension

`Qwen3NextDenseConfig` now carries the fields required by attention decode:

- `partial_rotary_factor`;
- `rope_theta`;
- `rms_norm_eps`;
- `max_position_embeddings`.

The parser validates RoPE geometry, positive context length, finite norm
epsilon, and query-head / KV-head divisibility. Backward-compatible fixture
defaults match the Qwen3-Next family values used by the frozen runtime.

## Checkpoint binding

`QwenCheckpointGqa::create(...)` consumes a full-attention
`QwenDenseLayerBinding` and resolves:

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
self_attn.q_norm.weight
self_attn.k_norm.weight
```

The Q/K/V/O affine modules are dequantized once through the already-certified
MLX affine CPU bridge. The resulting runtime owns persistent K/V decode state.

## Exit gate

OSM-31B is GREEN when a qpack attention-layer fixture proves:

- exact checkpoint Q/K/V/O tensors are consumed;
- RoPE/context fields come from `config.json`;
- two sequential tokens match the direct OSM-31A oracle;
- K/V caches and position match after each token;
- reset reproduces the fresh first token;
- DeltaNet layer bindings are rejected.

## Next

**OSM-31C — input RMSNorm + GQA residual sublayer**

Compose:

`h = x + GQA(RMSNorm(x))`

from exact checkpoint tensors, mirroring OSM-29C for the 12 full-attention
layers.


## Clean-base certification

After OSM-31A was squash-merged, this branch was rebuilt directly from the new
canonical `main`. The final CI run therefore validates only the OSM-31B
checkpoint-GQA delta.
