# OSM-27 — checkpoint-bound sparse MoE execution

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Eliminate the remaining hand-constructed dense router/shared-expert arrays from
the sparse-MoE runtime path.

OSM-26C can bind exact Qwen3-Next tensors from qpack `model.safetensors`.
OSM-27 turns those bindings directly into executable sparse-MoE runtime state.

## Runtime construction

`QwenCheckpointSparseMoeLayer::create(...)` consumes one
`QwenDenseLayerBinding`.

It performs:

```text
model.safetensors + config.json
        │
        ▼
QwenDenseLayerBinding
        │
        ├─ router Q8 affine ── CPU dequant once ── resident router matrix
        │
        ├─ shared gate Q4 ─┐
        ├─ shared up   Q4 ─┼─ VulkanResidentSharedExpert
        ├─ shared down Q4 ─┘
        │
        └─ shared scalar gate Q8 ─ CPU dequant once
```

The routed experts are still not materialized here. They remain in qpack
`packed_experts/layer_XX.bin` and enter through the existing two-level
host/Vulkan expert cache only when selected.

## Q8 dense bridge

Real Qwen3-Next MLX-4bit checkpoints use 8-bit affine overrides for:

- `mlp.gate` — the 512-way router;
- `mlp.shared_expert_gate` — the scalar shared-expert gate.

The generic CPU affine reference already supports both Q4 and Q8, so OSM-27
dequantizes these two relatively small always-resident matrices once during
runtime construction.

This is a correctness-first boundary. A later GPU-router gate can keep the
router Q8 and execute it directly on Vulkan.

## Shared expert

The shared expert gate/up/down projections must remain Q4 for the current
Vulkan shared-expert backend. They are passed directly from
`MlxAffineModule` into persistent Vulkan Q4 projection buffers without a
full float expansion.

## Execution path

After construction:

```text
hidden
  │
  ├─ checkpoint router -> softmax -> Top-K
  │                         │
  │                         └─ streamed routed experts
  │                                 │
  │                        Vulkan weighted accumulate
  │
  └─ checkpoint-bound resident shared expert
             │
             └─ sigmoid scalar gate -> Vulkan accumulate
                                   │
                                   ▼
                         one final MoE download
```

## Exit gate

OSM-27 is GREEN when:

- Windows and Linux compile/test;
- Linux executes the checkpoint-bound sparse path with real Vulkan;
- router Q8 values are read from `model.safetensors` and dequantized once;
- shared scalar gate Q8 is read/dequantized from the checkpoint;
- shared Q4 gate/up/down remain packed until Vulkan upload;
- routed experts still stream from qpack expert files;
- cold pass loads only selected routed experts;
- warm pass hits the Vulkan expert cache and bypasses host/qpack reads;
- full routed + shared output matches an independent CPU oracle.

## Next

**OSM-28 — first checkpoint-bound decoder sublayer**

Use `post_attention_layernorm.weight` from the bound layer, run RMSNorm, then
feed the normalized activation into the complete checkpoint-bound sparse-MoE
block and apply the residual connection.

That creates the first real Qwen decoder sublayer assembled entirely from
qpack checkpoint data.
