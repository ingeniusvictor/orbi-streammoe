# OSM-39A — deterministic QPACK conversion-plan contract

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Freeze the mapping from the official Qwen3-Next sharded checkpoint inventory to
the ORBI QPACK container before downloading or converting large payloads.

OSM-39A is metadata-only.

## Plan invariants

Every source tensor from `model.safetensors.index.json` appears exactly once in
the conversion plan.

The plan is sorted lexicographically by source tensor name, so output does not
depend on JSON object iteration order.

Each entry records:

- source tensor;
- source shard;
- tensor class;
- conversion action;
- target QPACK file;
- target QPACK path/section;
- layer index when applicable.

## Tensor classes

- `global_dense`: embedding, final norm and LM head;
- `layer_dense`: norms, routers, shared expert tensors, DeltaNet/GQA tensors;
- `routed_expert`: MoE expert payloads.

## Conversion actions

- `copy_bf16_to_f32` for vector/state tensors currently consumed as floats;
- `affine_quantize` for dense matrix modules;
- `split_packed_gate_up_experts` for official packed gate/up expert tensors;
- `split_packed_down_experts` for official packed down expert tensors;
- `direct_expert_quantize` for legacy/per-expert source layouts.

## Official Qwen3-Next checkpoint

The pinned official checkpoint currently uses one packed
`mlp.experts.gate_up_proj` tensor and one packed `mlp.experts.down_proj`
tensor per decoder layer.

OSM-39A therefore requires:

- 48 packed gate/up sources;
- 48 packed down sources;
- no mixed packed/direct expert representation.

## Target mapping

Dense/global tensors target `model.safetensors`.

Routed expert tensors target:

```text
packed_experts/layer_00.bin
...
packed_experts/layer_47.bin
```

with logical sections `gate_proj`, `up_proj`, and `down_proj`.

## Memory and network invariant

No tensor payload is downloaded by this feature. The official gate reuses only
the pinned `config.json` and `model.safetensors.index.json` metadata.

## Exit gate

OSM-39A is GREEN when:

- local packed-expert fixture is deterministic and complete;
- official checkpoint produces the expected packed expert inventory;
- Windows checkpoint-metadata workflow is GREEN;
- Ubuntu checkpoint-metadata workflow is GREEN;
- normal Windows/Ubuntu CI remains GREEN;
- tokenizer integration remains GREEN;
- Linux real Vulkan regression remains GREEN.

## Next

**OSM-39B — QPACK expert-layout geometry contract**

Derive exact per-expert section geometry, quantization group layout, expert
stride and per-layer output sizes from the official config before converting a
single full expert.
