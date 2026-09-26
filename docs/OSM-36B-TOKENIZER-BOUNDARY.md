# OSM-36B — tokenizer boundary

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Define the portable text/token boundary around the already-certified OSM-36A
autoregressive token-ID session without coupling tokenizer implementation to
checkpoint execution, Vulkan, expert streaming, or cache policy.

## Architecture

```text
text prompt
   ↓
Tokenizer::encode(...)
   ↓
QwenPreparedTextPrompt
   ↓
OSM-36A QwenGreedyTokenSession
   ↓
generated token IDs
   ↓
Tokenizer::decode(...)
   ↓
text result
```

OSM-36B intentionally does **not** implement a Qwen-specific BPE engine inside
the numerical runtime. A concrete tokenizer may be provided by a portable
native component, platform adapter, JNI bridge, or another independently
certified implementation.

## Tokenizer contract

`Tokenizer` exposes:

- encode text to token IDs;
- decode token IDs to text;
- optional vocabulary size;
- optional EOS token ID;
- explicit diagnostics.

The boundary functions catch tokenizer exceptions so platform adapters cannot
unwind through the numerical runtime.

## Session integration

`prepare_qwen_text_prompt(...)`:

- preserves OSM-36A generation options;
- encodes the prompt;
- rejects empty encodings;
- validates IDs when tokenizer vocabulary size is known;
- optionally merges tokenizer EOS into the OSM-36A stop-token set;
- avoids duplicate EOS stop IDs.

`decode_qwen_generated_text(...)`:

- refuses unsuccessful token sessions;
- validates generated IDs when vocabulary size is known;
- decodes generated IDs;
- preserves the generated-token history alongside text.

## Exit gate

OSM-36B is GREEN when CI proves:

- special-token and raw encode paths;
- exact OSM-36A option passthrough;
- EOS stop-token merge and deduplication;
- generated-token decode;
- skip-special-token policy;
- encode/decode failure containment;
- empty/OOV token guards;
- exception containment;
- Windows + Ubuntu portability.

No real Vulkan rerun is required specifically for this boundary because OSM-36B
executes no numerical kernels. OSM-36A already certifies the downstream token
session with real Vulkan Linux execution.

## Clean-base certification

This candidate is built directly from OSM-36A canonical `main`.

## Next

**OSM-36C — concrete Qwen tokenizer adapter / text session façade**

Bind a real Qwen-compatible tokenizer implementation to this interface and
certify end-to-end text → tokens → autoregressive session → text while keeping
tokenization independently replaceable on Windows and Android.
