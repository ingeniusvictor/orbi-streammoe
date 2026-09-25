# OSM-26B — MLX affine checkpoint binding

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Interpret the dense qpack `model.safetensors` using the same MLX affine
quantization contract that Swiftlet accepts.

OSM-26A provided file-backed safetensors access. OSM-26B adds module-level
quantization semantics from `config.json`.

## Quantization config

The parser supports:

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 8,
    "mode": "affine",
    "model.layers.0.mlp.shared_expert.gate_proj": {
      "group_size": 64,
      "bits": 4
    }
  }
}
```

Absent `mode` means `affine`, matching MLX/Swiftlet behavior.

Non-affine modes are rejected before tensor execution.

## Override resolution

Per-module overrides are matched by suffix compatibility. When more than one
key could match, ORBI sorts by longest key first so the most specific override
wins deterministically.

## Affine module contract

A quantized module is represented by:

```text
<path>.weight   U32 packed values
<path>.scales   F32/F16/BF16
<path>.biases   F32/F16/BF16
```

For 4-bit data, each U32 stores eight quantized values. For 8-bit data, each
U32 stores four.

OSM-26B verifies that the logical input dimension derived from packed words is
identical to the dimension implied by scale groups:

```text
packed_last_dim * (32 / bits)
==
scales_last_dim * group_size
```

## Output

`read_affine_module(path)` keeps weights quantized. It returns packed U32
words plus decoded scale/bias arrays and geometry. It does not expand the full
weight matrix to float.

That is the representation required by the existing Vulkan Q4 execution path.

## Exit gate

OSM-26B is GREEN when Windows and Linux compile/test, default quantization and
per-module overrides resolve correctly, missing mode defaults to affine,
non-affine configs are rejected, 4-bit and 8-bit geometry is validated, and
packed module values remain quantized.

## Next

**OSM-26C — Qwen dense tensor binding**

Resolve real Qwen layer names for router, shared expert, shared gate, norms,
attention/DeltaNet parameters and construct runtime-ready layer bindings from
qpack dense tensors.
