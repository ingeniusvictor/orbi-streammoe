# OSM-36A — autoregressive token-ID session

Status: **IMPLEMENTED / STACKED CI CERTIFICATION PENDING**

## Goal

Move ORBI StreamMoE from a single next-token step to a complete bounded greedy
decode loop over token IDs.

The session deliberately stops at the tokenizer boundary. Text/tokenizer
integration comes after the numerical decode loop is certified.

## Decode semantics

For prompt token IDs:

```text
[p0, p1, ... pN]
```

the runtime performs teacher-forced prompt consumption:

```text
step(p0) -> prediction ignored
step(p1) -> prediction ignored
...
step(pN) -> generated token g0
step(g0) -> g1
step(g1) -> g2
...
```

Generation stops when:

- a configured stop-token ID is produced; or
- `max_new_tokens` is reached.

## Runtime contract

`QwenGreedyTokenSession` owns one OSM-35C
`QwenCheckpointModelShell`.

Options expose:

- `max_new_tokens`;
- `lm_head_chunk_rows`;
- zero or more stop-token IDs;
- `reset_before_prompt`.

The returned result records:

- prompt IDs;
- generated IDs;
- generated greedy logits;
- number of model steps;
- explicit stop reason;
- diagnostic text.

## State semantics

With `reset_before_prompt=true`, each request starts from a fresh DeltaNet/GQA
state.

With `reset_before_prompt=false`, callers may feed additional token IDs into
the existing sequence state. This is the primitive needed for later interactive
or chat-session work.

## Exit gate

OSM-36A is GREEN when CI proves:

- multi-token teacher-forced prompt consumption;
- three-token greedy continuation matches direct OSM-35C stepping;
- model-step count is `prompt_size + generated_size - 1`;
- configured stop-token termination;
- max-new-token termination;
- session reset reproduces the same first continuation;
- invalid prompt token, invalid stop token, empty prompt and zero limits fail
  without corrupting runtime state;
- Linux CI reruns the session test with real Vulkan required.

## Next

**OSM-36B — tokenizer boundary**

Add tokenizer encode/decode integration around this token-ID core without
coupling tokenization to model execution or expert streaming.
