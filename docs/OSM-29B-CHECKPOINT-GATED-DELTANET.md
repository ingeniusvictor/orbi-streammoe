# OSM-29B — checkpoint-bound Gated DeltaNet

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Bind the exact Qwen3-Next DeltaNet tensors loaded by OSM-26C to the stateful
OSM-29A CPU semantic oracle.

The runtime now consumes `QwenDeltaDenseBinding` rather than hand-constructed
projection matrices.

## Construction

`QwenCheckpointGatedDeltaNet::create(...)` consumes one linear
`QwenDenseLayerBinding` and:

- dequantizes `linear_attn.in_proj_qkvz`;
- dequantizes `linear_attn.in_proj_ba`;
- dequantizes `linear_attn.out_proj`;
- keeps checkpoint Conv1D / dt_bias / A_log / norm tensors;
- derives head and state geometry from `config.json`;
- creates an initially empty persistent decode state.

The three affine modules use the generic MLX Q4/Q8 CPU bridge already certified
by the checkpoint sparse-MoE path.

## Execution

Input to this object is already input-RMS-normalized:

```text
normalized hidden
      ↓
checkpoint fused projections
      ↓
causal depthwise conv + persistent tail
      ↓
gated-delta recurrent state
      ↓
gated RMSNorm
      ↓
checkpoint out_proj
      ↓
DeltaNet branch output
```

## State

The wrapper exposes read-only state for certification and a `reset_state()`
operation for a new session.

## Exit gate

OSM-29B is GREEN when a qpack fixture proves:

- exact DeltaNet tensors are consumed from the checkpoint binding;
- two sequential tokens match the direct OSM-29A oracle;
- Conv1D and recurrent state persist across tokens;
- reset reproduces the fresh-session first-token output;
- non-linear layer bindings are rejected.

## Next

**OSM-29C — input RMSNorm + DeltaNet residual branch**

Use checkpoint `input_layernorm.weight`, feed the normalized token into this
runtime, and apply:

`h = x + DeltaNet(RMSNorm(x))`.

That will complete the first half of a linear-attention Qwen3-Next decoder layer.
