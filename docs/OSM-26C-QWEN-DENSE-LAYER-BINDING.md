# OSM-26C — Qwen3-Next dense layer binding

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Bind the generic qpack/MLX checkpoint reader to the exact dense tensor names and
shapes used by Qwen3-Next decoder layers.

This is the first gate where the runtime understands a checkpoint as a
**Qwen3-Next layer**, rather than as unrelated tensors.

## Exact names

Every layer uses:

```text
model.layers.N.input_layernorm.weight
model.layers.N.post_attention_layernorm.weight

model.layers.N.mlp.gate
model.layers.N.mlp.shared_expert.gate_proj
model.layers.N.mlp.shared_expert.up_proj
model.layers.N.mlp.shared_expert.down_proj
model.layers.N.mlp.shared_expert_gate
```

DeltaNet layers additionally use the fused Qwen3-Next layout:

```text
linear_attn.conv1d.weight
linear_attn.dt_bias
linear_attn.A_log
linear_attn.norm.weight
linear_attn.out_proj
linear_attn.in_proj_qkvz
linear_attn.in_proj_ba
```

Full-attention layers use:

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
self_attn.q_norm.weight
self_attn.k_norm.weight
```

These names match the frozen Swiftlet Qwen3-Next loader.

## Shape validation

The binder derives and validates the architecture geometry from `config.json`.

Examples:

```text
router            [num_experts, hidden]
shared gate/up    [shared_intermediate, hidden]
shared down       [hidden, shared_intermediate]
shared gate scalar[1, hidden]

Delta qkvz        [2*key_dim + 2*value_dim, hidden]
Delta ba          [2*num_value_heads, hidden]
Delta out         [hidden, value_dim]

Attention q       [2*num_attention_heads*head_dim, hidden]
Attention k/v     [num_kv_heads*head_dim, hidden]
Attention o       [hidden, num_attention_heads*head_dim]
```

Plain recurrent/norm tensors are also checked by element count.

## Residency boundary

OSM-26C binds one layer at a time. It does not load all 48 layers into RAM.
That preserves the ability to control the dense working set while the routed
experts remain streamed separately.

## Exit gate

OSM-26C is GREEN when:

- Windows and Linux compile/test;
- config-driven linear/full-attention classification is correct;
- exact upstream Qwen tensor names resolve;
- fused DeltaNet module geometry is validated;
- GQA module geometry is validated;
- routed/shared-MoE dense modules resolve through the MLX affine binder;
- out-of-range and malformed layer geometry are rejected.

## Next

**OSM-27 — checkpoint-bound sparse MoE execution**

Use the bound router/shared-expert modules from a qpack dense fixture to create
the actual Vulkan resident shared expert and routed path, eliminating the
remaining hand-constructed dense Q4 arrays from the sparse-MoE certification.
