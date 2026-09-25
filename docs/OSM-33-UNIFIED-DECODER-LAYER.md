# OSM-33 — unified alternating Qwen3-Next decoder layer

Status: **IMPLEMENTED / CLEAN CI CERTIFICATION PENDING**

## Goal

Replace explicit caller-side selection of the two decoder-layer families with
one checkpoint-bound runtime abstraction.

Qwen3-Next-80B-A3B uses:

- 36 Gated DeltaNet layers;
- 12 gated GQA layers;
- full attention every fourth decoder layer.

OSM-30 and OSM-32 already certify the two concrete layer families. OSM-33
provides the topology-aware dispatcher above them.

## Topology contract

`qwen_checkpoint_decoder_layer_kind(...)` maps the configured layer index to:

```text
layer 0  -> DeltaNet
layer 1  -> DeltaNet
layer 2  -> DeltaNet
layer 3  -> GQA
layer 4  -> DeltaNet
layer 5  -> DeltaNet
layer 6  -> DeltaNet
layer 7  -> GQA
...
```

For the 48-layer Qwen3-Next-80B topology this yields exactly:

```text
36 x Gated DeltaNet
12 x gated GQA
```

## Unified runtime

`QwenCheckpointDecoderLayer::create(...)` validates that the checkpoint
binding agrees with `full_attention_interval` and then owns exactly one of:

- `QwenCheckpointLinearDecoderLayer`;
- `QwenCheckpointFullAttentionDecoderLayer`.

The public `run(...)` contract is identical for both.

## State

The wrapper exposes only the state relevant to its selected family:

- `delta_state()` for Gated DeltaNet;
- `gqa_state()` for full attention.

The other pointer is null.

`reset_state()` dispatches to the selected concrete layer.

## Correctness boundary

OSM-33 does not duplicate numerical kernel certification. Its responsibility is
topology and dispatch correctness. Numerical behavior remains certified by:

- OSM-30 for complete DeltaNet decoder layers;
- OSM-32 for complete GQA decoder layers.

## Exit gate

OSM-33 is GREEN when CI proves:

- the 48-layer selector yields 36 DeltaNet + 12 GQA layers;
- every fourth layer selects GQA;
- out-of-range layer indices are rejected;
- layer-0 checkpoint binding creates the DeltaNet implementation;
- layer-3 checkpoint binding creates the GQA implementation;
- state exposure matches the selected implementation;
- checkpoint/config classification disagreement is rejected.

## Next

**OSM-34 — multi-layer decoder stack**

Instantiate a sequence of unified OSM-33 layers and execute hidden state through
the real D-D-D-G alternating order while sharing one routed-expert cache.


## Clean-base certification

After OSM-32 was squash-merged, this branch was rebuilt directly from the new
canonical `main`. The final CI run therefore validates only the OSM-33
topology/dispatch delta.
