# OSM-29A — Qwen3-Next Gated DeltaNet CPU decode semantics

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Freeze the exact single-token decode semantics of the Qwen3-Next
Gated DeltaNet branch before introducing a Vulkan implementation.

This covers the attention branch used by 36 of the 48 layers in
Qwen3-Next-80B-A3B.

## Frozen upstream behavior

The implementation follows the frozen Swiftlet / MLX Qwen3-Next path:

1. fused `in_proj_qkvz` and `in_proj_ba`;
2. per-key-head fused-interleaved Q/K/V/Z/B/A unpacking;
3. causal depthwise Conv1D with persistent `kernel-1` tail;
4. per-head Q/K RMSNorm;
5. Q scaling by `1/dk` and K scaling by `1/sqrt(dk)`;
6. recurrent gated-delta update:
   `g = exp(-exp(A_log) * softplus(a + dt_bias))`;
7. `beta = sigmoid(b)`;
8. persistent recurrent matrix update;
9. gated RMSNorm `RMSNorm(out) * SiLU(z)`;
10. output projection back to hidden size.

## State

`QwenGatedDeltaNetState` owns two persistent decode states:

```text
conv_tail
  shape = [kernel - 1, conv_dim]

recurrent
  shape = [num_value_heads, value_head_dim, key_head_dim]
```

An empty state is interpreted as a fresh zero-initialized session.

## Certification vector

The test runs two deterministic decode tokens through the same state and checks:

- first-token output against an independently generated numeric reference;
- second-token output against an independently generated numeric reference;
- final Conv1D tail;
- final recurrent delta state;
- invalid value-head/key-head ratio rejection.

The numeric vectors were generated independently from the frozen Swiftlet/MLX
equations rather than from the production C++ function.

## Exit gate

OSM-29A is GREEN when Windows and Linux compile/test and both token/state
reference vectors match within the specified floating-point tolerance.

## Next

**OSM-29B — checkpoint-bound Gated DeltaNet**

Consume `QwenDeltaDenseBinding` from OSM-26C, dequantize the three affine
projection modules through the existing MLX affine bridge, and execute the
OSM-29A stateful decode step from actual qpack checkpoint tensors.

After that, the branch can be moved to Vulkan while retaining OSM-29A as the
semantic oracle.
