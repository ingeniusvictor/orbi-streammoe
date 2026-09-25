# OSM-35B — streamed global vocabulary matrices

Status: **IMPLEMENTED / STACKED CI CERTIFICATION PENDING**

## Goal

Complete the bounded-memory path for the two vocabulary-sized matrices around
the decoder stack.

The model shell needs:

```text
token id
  -> model.embed_tokens row
  -> decoder stack
  -> model.norm.weight
  -> lm_head rows
  -> logits / greedy token
```

Neither `model.embed_tokens` nor `lm_head` may be expanded to a complete
FP32 matrix during normal decode.

## Bounded safetensors reads

`SafetensorsReader` and `QpackDenseReader` now expose typed element-range
reads for:

- F32/F16/BF16 values;
- U32 packed affine weights.

The range is validated before I/O and only the requested bytes are copied.

## Matrix-row streaming

`read_qwen_matrix_rows(...)` supports both checkpoint layouts:

- plain F32/F16/BF16 matrices;
- affine Q4/Q8 matrices.

For affine matrices, only the requested row range of weight/scales/biases is
read and dequantized.

## Embedding

`read_qwen_embedding_row(...)` reads exactly one
`model.embed_tokens` row for one token ID.

Memory is therefore O(hidden_size), not O(vocab_size * hidden_size).

## LM head

`project_qwen_lm_head_chunk(...)` projects one hidden vector against a bounded
contiguous vocabulary range.

`greedy_qwen_lm_head_streaming(...)` scans those chunks while retaining only
the current best token/logit, so greedy decode does not need a full-vocabulary
logits allocation.

Tied embeddings automatically use the same matrix descriptor.

## Exit gate

OSM-35B is GREEN when CI proves:

- bounded F32 and U32 safetensor reads match full reads;
- selected quantized embedding rows equal full-dequantization references;
- bounded LM-head logits equal full-matrix projection slices;
- streaming greedy argmax equals full-vocabulary argmax;
- tied LM head uses the embedding matrix correctly;
- token/range/chunk guards reject invalid inputs.

## Memory invariant

No OSM-35B production path calls `read_affine_module(...)` for the complete
embedding or LM-head matrix.

The full dequantization used in the test exists only as an independent oracle
for the tiny synthetic fixture.

## Next

**OSM-35C — checkpoint model shell**

Combine OSM-34 + OSM-35A/B:

```text
token id
  -> streamed embedding row
  -> alternating decoder stack
  -> final RMSNorm
  -> streamed LM-head projection
  -> next-token result
```

That will be the first end-to-end autoregressive model step in ORBI StreamMoE.
