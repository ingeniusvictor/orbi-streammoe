# OSM-38C — safetensors shard header/range pilot

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Inspect real tensor dtype, shape and byte ranges from the pinned official
Qwen3-Next-80B-A3B-Instruct checkpoint without downloading any complete model
shard.

OSM-38A/38B proved the sharded index contract. OSM-38C crosses the next boundary:
the actual safetensors shard headers.

## Strict network contract

For every representative shard, the fetcher performs exactly two HTTP range
requests:

1. bytes 0-7 — the safetensors little-endian JSON header length;
2. bytes 8 through the exact end of the JSON header.

HTTP 206 Partial Content is mandatory. If the remote endpoint ignores the Range
request and returns 200, the script aborts before reading the response body.
This prevents an accidental multi-gigabyte shard download.

The header size is capped at 64 MiB.

## Representative official tensors

The pilot resolves representative tensors through the authoritative
model.safetensors.index.json weight map:

- model.embed_tokens.weight;
- layer 0 input RMSNorm;
- layer 0 DeltaNet in_proj_qkvz;
- layer 0 packed expert gate_up_proj;
- layer 3 GQA q_proj;
- layer 3 packed expert down_proj;
- model.norm.weight;
- lm_head.weight.

Only the unique shard headers containing these tensors are fetched.

## C++ range manifest contract

The generated manifest records:

- pinned model and snapshot;
- selected tensor names;
- shard filename and total remote file size;
- safetensors header size;
- exact bytes fetched;
- tensor count in each parsed header;
- selected tensor dtype, shape and payload-relative data offsets.

The C++ validator proves:

- fetched bytes are exactly 8 + header_size;
- dtype/shape agree with tensor byte ranges;
- ranges fit inside the remote shard payload size;
- each selected tensor is present;
- each tensor was found in the exact shard named by the checkpoint index;
- duplicate or malformed probe data is rejected.

## Official gate

The pinned official gate additionally requires:

- all selected tensors are BF16;
- embedding shape is [151936, 2048];
- final RMSNorm shape is [2048];
- LM-head shape is [151936, 2048];
- layer-0 input RMSNorm shape is [2048];
- total downloaded header bytes stay below 64 MiB.

## Memory / storage invariant

OSM-38C does not materialize tensor payloads and does not download complete
shards. It is a metadata/range pilot only.

## Exit gate

OSM-38C is GREEN when:

- local malformed-range contract tests pass in normal Windows + Ubuntu CI;
- official pinned range fetch passes on Windows + Ubuntu;
- official header metadata validates against OSM-38A/38B;
- existing numerical, tokenizer and real-Vulkan gates remain GREEN.

## Next

**OSM-38D — conversion slice pilot**

Use the certified shard offsets to range-fetch one bounded real tensor slice and
feed it into an offline BF16 to ORBI/QPACK conversion path. Keep the first slice
small and correctness-first; do not begin a full 80B conversion yet.
