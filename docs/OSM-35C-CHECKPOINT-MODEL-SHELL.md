# OSM-35C — end-to-end checkpoint model shell

Status: **IMPLEMENTED / STACKED CI CERTIFICATION PENDING**

## Goal

Close the first complete autoregressive Qwen3-Next inference step around the
OSM-34 decoder stack while preserving bounded memory for vocabulary matrices.

```text
token id
  ↓
stream one embedding row
  ↓
alternating decoder stack
  ↓
checkpoint final RMSNorm
  ↓
stream LM-head chunks
  ↓
greedy next token
```

## Runtime

`QwenCheckpointModelShell` owns:

- the OSM-35A global checkpoint binding;
- the OSM-34 stateful decoder stack;
- model configuration needed by final RMSNorm.

The qpack checkpoint and routed-expert cache remain external runtime resources.

## Step contract

`step_greedy(...)` performs exactly one incremental token step:

1. read one quantized/plain embedding row through OSM-35B;
2. execute every decoder layer with persistent DeltaNet/GQA state;
3. execute checkpoint `model.norm.weight` through Vulkan RMSNorm;
4. scan `lm_head` in bounded chunks;
5. return the greedy token/logit.

The full vocabulary matrix is never expanded to FP32 and the full logits vector
is not required for greedy decode.

## State

Decoder sequence state persists between calls:

- DeltaNet Conv1D tails and recurrent matrices;
- GQA position and K/V caches.

`reset_state()` clears the sequence while leaving immutable checkpoint
metadata intact.

## Exit gate

OSM-35C is GREEN when a four-layer D-D-D-G + tiny-vocabulary qpack fixture
proves:

- embedding row matches independent full-dequant oracle;
- all four decoder layers execute;
- final Vulkan RMSNorm matches CPU reference;
- streamed LM-head greedy token/logit equals full LM-head projection;
- a second token advances recurrent/KV state correctly;
- reset reproduces a fresh first-token result;
- invalid token IDs and zero LM-head chunk size fail safely;
- Linux CI exercises the Vulkan stages with real Vulkan.

## Next

**OSM-36 — autoregressive session loop**

Add a session-level loop around repeated `step_greedy(...)`, stop-token
handling, decode limits and tokenizer integration boundaries.
