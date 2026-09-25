# OSM-34 — multi-layer alternating decoder stack

Status: **IMPLEMENTED / CLEAN CI CERTIFICATION PENDING**

## Goal

Execute hidden state through a sequence of unified OSM-33 decoder layers while
sharing one routed-expert cache.

This is the first runtime object that represents an actual Qwen3-Next decoder
stack rather than one isolated layer.

## Construction

`QwenCheckpointDecoderStack::create(...)` receives:

- one `QpackMlxCheckpoint`;
- one parsed `Qwen3NextDenseConfig`;
- one Vulkan compute context.

It binds every checkpoint layer in order and constructs one OSM-33 unified
layer per index.

The topology is therefore derived directly from
`full_attention_interval` rather than caller-side branching.

## Execution

```text
hidden_0
   │
   ▼
layer 0  DeltaNet + MoE
   │
   ▼
layer 1  DeltaNet + MoE
   │
   ▼
layer 2  DeltaNet + MoE
   │
   ▼
layer 3  GQA + MoE
   │
   ▼
hidden_4
```

All layers receive the same `VulkanResidentExpertCache`. Cache keys already
include `(layer, expert)`, so expert residency remains layer-safe while one
bounded budget is shared across the whole stack.

## State

Each layer retains its own sequence state:

- DeltaNet layers keep Conv1D tail + recurrent delta state;
- GQA layers keep position + growing K/V cache.

`reset_state()` clears every layer.

## Exit gate

OSM-34 is GREEN when a four-layer D-D-D-G qpack fixture proves:

- stack construction yields exactly 3 DeltaNet + 1 GQA;
- layer order is D-D-D-G;
- one token executes through all four checkpoint-bound layers;
- a second token reuses each layer's persistent sequence state;
- one shared expert cache services every layer;
- GQA position reaches two after two tokens;
- every DeltaNet layer owns non-empty recurrent state after execution;
- reset clears all layer states;
- malformed input hidden size is rejected;
- Linux CI executes the Vulkan portions through real Vulkan.

## Next

**OSM-35 — model shell**

Add checkpoint token embedding, final RMSNorm and LM head around OSM-34:

```text
token ids
  -> embedding
  -> 48-layer decoder stack
  -> final RMSNorm
  -> LM head
  -> logits
```

That is the bridge from a decoder engine to an actual autoregressive model.


## Clean-base certification

After OSM-33 was squash-merged, this branch was rebuilt directly from the new
canonical `main`. The final CI run therefore validates only the OSM-34
multi-layer decoder-stack delta.
