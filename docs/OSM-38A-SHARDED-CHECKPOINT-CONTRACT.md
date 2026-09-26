# OSM-38A — sharded checkpoint manifest contract

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Validate a real Hugging Face Qwen3-Next sharded checkpoint before downloading,
converting, quantizing, or executing any tensor payload.

OSM-38A reads only:

- `config.json`;
- `model.safetensors.index.json`.

It therefore scales to the official Qwen3-Next-80B-A3B-Instruct checkpoint
without requiring the ~163 GB BF16 payload during CI.

## Contract

`inspect_qwen3_next_sharded_checkpoint(...)` certifies:

- `model_type=qwen3_next`;
- non-zero model/vocabulary/layer/expert geometry;
- complete repeating attention intervals;
- MoE top-k does not exceed expert count;
- non-zero safetensors metadata total size;
- non-empty tensor-to-shard weight map;
- canonical `model-xxxxx-of-yyyyy.safetensors` shard names;
- global embedding, final norm and LM-head tensors;
- one complete D-D-D-G block;
- final-layer presence to catch truncated indexes.

## Why metadata-first

The official checkpoint is sharded and large. The pilot must be able to reject
an incompatible or incomplete source before network/storage cost is paid.

This gate intentionally does not:

- download shard payloads;
- inspect tensor dtypes/shapes inside shards;
- convert experts to QPACK;
- quantize weights;
- execute inference.

Those are later OSM-38 stages.

## Official target

The current official Qwen3-Next-80B-A3B-Instruct release exposes 48 layers with
a 4-layer full-attention interval, 512 experts, 10 active experts per token,
vocabulary 151936, and 41 BF16 safetensors shards.

OSM-38A keeps the contract structural rather than hard-coding 41 shards, so a
compatible repack remains valid.

## Exit gate

OSM-38A is GREEN when Windows + Ubuntu certify:

- valid sharded metadata inspection;
- D-D-D-G tensor inventory;
- global tensor inventory;
- final-layer guard;
- shard inventory;
- wrong model, missing global tensor and malformed shard rejection.

No new numerical kernel is introduced.

## Next

**OSM-38B — authoritative official checkpoint metadata gate**

CI fetches only the pinned official `config.json` and
`model.safetensors.index.json` and requires OSM-38A to validate them,
recording exact shard/tensor/size metadata without downloading model weights.
