# OSM-38B — authoritative official checkpoint metadata gate

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Apply the OSM-38A sharded-checkpoint contract to the pinned official
Qwen3-Next-80B-A3B-Instruct metadata on both Windows and Ubuntu.

## Pinned source

Model:

`Qwen/Qwen3-Next-80B-A3B-Instruct`

Snapshot:

`f5e99a3698d364cf77584543481b778afee26177`

Only these files are downloaded:

- `config.json`;
- `model.safetensors.index.json`.

No model shard payload is downloaded.

## Authoritative assertions

The dedicated gate requires:

- hidden size 2048;
- vocabulary 151936;
- 48 decoder layers;
- full-attention interval 4;
- 512 routed experts;
- top-10 expert routing;
- 41 safetensors shards;
- checkpoint metadata size in the expected ~163 GB BF16 range;
- D-D-D-G selector for layers 0..3;
- final layer 47 is full gated attention.

The structural tensor inventory is still enforced by OSM-38A.

## Network invariant

This gate downloads only metadata, not the 41 BF16 shard files. It is therefore
safe to run routinely in CI and catches upstream/source-contract drift before a
large conversion job starts.

## Exit gate

OSM-38B is GREEN when the pinned metadata probe passes on:

- Windows;
- Ubuntu;

while the normal CI and tokenizer-integration workflows remain GREEN.

## Next

**OSM-38C — safetensors shard header/range pilot**

Fetch or stage the minimum shard byte ranges needed to inspect actual tensor
dtype/shape metadata without downloading full shard payloads, then map the
official BF16 tensor layout into the ORBI QPACK conversion plan.
