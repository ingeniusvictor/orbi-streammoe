# OSM-39G — deterministic QPACK expert-package finalization

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Finalize the routed-expert portion of a converted Qwen3-Next checkpoint into the
exact metadata contract consumed by `QpackReader`, without pretending that the
dense `model.safetensors` conversion is already complete.

OSM-39G therefore produces an **expert shell**, not a complete inference
checkpoint.

## Canonical operator layout

```text
<output>/
├── manifest.json
├── config.json
├── conversion-provenance.json
└── packed_experts/
    ├── layout.json
    ├── layer_00.bin
    ├── layer_00.progress.json
    ├── ...
    ├── layer_47.bin
    └── layer_47.progress.json
```

OSM-39F should write each layer and journal directly to these canonical paths.

## Finalization gate

Before metadata is emitted, every layer must:

- exist at the exact fixed-stride production size;
- have a primary journal or recoverable `.bak` journal;
- report all experts complete;
- pass OSM-39D readback verification for every expert.

Only then are metadata files generated.

## Generated layout

`packed_experts/layout.json` is derived from the same OSM-39B geometry used by
conversion:

- `expertCount`;
- `layerCount`;
- `expertStride`;
- all nine Q4 affine expert sections;
- the exact Qwen3-Next `linearLayers` map.

For the pinned model, `full_attention_interval=4`, therefore the map is:

```text
D D D G × 12
```

with 36 DeltaNet layers and 12 full-attention layers.

## Provenance

`conversion-provenance.json` records deterministically:

- source checkpoint and pinned snapshot;
- affine quantization contract;
- model/expert geometry;
- full-attention interval;
- every layer path and byte size;
- the 512 OSM-39D expert checksums for every finalized layer.

No timestamps are written, so identical inputs produce identical metadata.

## Runtime compatibility

After finalization, the generated expert shell is opened with the production
`QpackReader`. This validates the same manifest/layout/layer-size contract used
by runtime expert streaming.

The shell deliberately does not declare `model.safetensors`. Dense/global
conversion and final full-checkpoint certification remain a subsequent gate.

## Production scale

Pinned Qwen3-Next-80B-A3B expert geometry:

- 48 layers;
- 512 experts per layer;
- 1,966,080 bytes per QPACK expert;
- 1,006,632,960 bytes per layer;
- 48,318,382,080 routed-expert bytes total.

## Next

**OSM-40A — deterministic dense/global conversion plan execution**

Begin converting the OSM-39A dense/global tensor plan into the
`model.safetensors` payload required by the already-certified decoder runtime.
