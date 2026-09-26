# OSM-36C — greedy text session facade

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Expose one stateful text-generation entry point over the already-certified
OSM-36A token session and OSM-36B tokenizer boundary.

```text
text prompt
   ↓
Tokenizer::encode
   ↓
QwenGreedyTokenSession
   ↓
generated token IDs
   ↓
Tokenizer::decode
   ↓
text result
```

## Contract

`QwenGreedyTextSession` owns the numerical token session and therefore owns
DeltaNet/GQA sequence state. It does **not** own or implement a tokenizer.
Callers inject any implementation of the OSM-36B `Tokenizer` interface.

The facade exposes:

- `create(...)`;
- `generate(...)`;
- `reset()`;
- model vocabulary inspection;
- underlying token-session inspection for diagnostics.

The result preserves:

- completion status;
- whether the token session completed;
- model stop reason;
- model-step count;
- encoded prompt IDs;
- generated IDs;
- generated greedy logits;
- decoded text;
- diagnostic text.

## Safety / state semantics

Tokenizer vocabulary metadata, when available, must exactly match the model
vocabulary before model state advances.

Encoding or validation failures happen before model execution.

A decode failure occurs **after** numerical generation and therefore cannot
roll back model state. The result preserves generated IDs/logits and reports
that the token session executed. Callers may reset or continue intentionally.

## Correctness gate

OSM-36C is GREEN when CI proves:

- text facade output matches direct OSM-36A token generation;
- prompt IDs pass through exactly;
- generated IDs/logits and model-step accounting are preserved;
- reset reproduces deterministic greedy output;
- tokenizer/model vocabulary mismatch is rejected before execution;
- encode failure is contained before execution;
- decode failure preserves already-generated token evidence;
- invalid generation options remain rejected by OSM-36A;
- Windows + Ubuntu CI remain GREEN;
- Linux real-Vulkan regression remains GREEN.

## Architectural invariant

No tokenizer algorithm, BPE merge table, platform JNI code, or text-processing
dependency enters the Vulkan/model execution core.

## Clean-base certification

This candidate is based directly on OSM-36B canonical `main`
`3c51f13e432c4d7e5831992f9dc3616b4f2ab6b2`.

## Next

**OSM-37A — Qwen tokenizer asset contract**

Inspect and bind the real tokenizer assets required by the target
Qwen3-Next-80B-A3B checkpoint before implementing a concrete tokenizer adapter.
