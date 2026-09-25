# OSM-35A — model-global checkpoint contract

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Bind the tensors that surround the Qwen3-Next decoder stack without eagerly
materializing the two vocabulary-sized matrices.

The frozen Swiftlet model shell uses:

```text
model.embed_tokens
model.norm.weight
lm_head
```

and uses `model.embed_tokens` as the LM head when
`tie_word_embeddings=true`.

## Config

`Qwen3NextDenseConfig` now exposes:

- `vocab_size`;
- `tie_word_embeddings`.

They are backward-compatible for older layer-only fixtures: `vocab_size`
defaults to zero until a model-global binding is requested, and embedding tying
defaults to false.

## Memory contract

`QwenGlobalCheckpointBinding` does **not** dequantize or copy the complete
embedding or LM-head matrix.

Instead it records a checked descriptor:

```text
checkpoint path
rows
logical columns
quantized?
quantization spec
```

Only `model.norm.weight` is loaded eagerly because it is one hidden-size
vector.

This keeps OSM-35 compatible with the project's bounded-memory goal. The next
stage can implement one-row embedding reads and chunked vocabulary projection
directly from safetensors ranges.

## Exit gate

OSM-35A is GREEN when CI proves:

- `vocab_size` and `tie_word_embeddings` parse correctly;
- quantized embedding geometry is resolved without full dequantization;
- final norm is loaded and shape checked;
- untied `lm_head` is independently resolved;
- tied mode points the LM-head descriptor at `model.embed_tokens`;
- vocabulary/hidden geometry mismatch is rejected.

## Next

**OSM-35B — streamed embedding row + chunked LM head**

Use safetensors byte-range reads to fetch exactly one embedding row and process
the vocabulary projection in bounded chunks, preserving the low-memory
architecture instead of expanding the full matrices to FP32.
