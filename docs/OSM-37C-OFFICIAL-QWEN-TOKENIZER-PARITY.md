# OSM-37C — official Qwen tokenizer parity

Status: **IMPLEMENTED / AUTHORITATIVE CI CERTIFICATION PENDING**

## Goal

Certify the concrete OSM-37B tokenizer adapter against the official
`Qwen/Qwen3-Next-80B-A3B-Instruct` tokenizer rather than only a synthetic
fixture.

## Authoritative source

Model:

```text
Qwen/Qwen3-Next-80B-A3B-Instruct
```

Pinned snapshot:

```text
f5e99a3698d364cf77584543481b778afee26177
```

Official `tokenizer.json` SHA256:

```text
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
```

The CI fixture is downloaded only for the tokenizer gate; the 80B model weights
are never downloaded.

## Differential oracle

CI installs pinned Python Hugging Face `tokenizers==0.22.1`.

For a multilingual and special-token prompt corpus it records:

- token IDs with `add_special_tokens=false`;
- token IDs with `add_special_tokens=true`;
- decode output preserving special tokens;
- decode output skipping special tokens.

The C++ OSM-37B adapter then loads the exact same official tokenizer assets and
must match every vector exactly.

## Corpus

The gate covers:

- ASCII punctuation;
- Spanish UTF-8;
- CJK text;
- newline-containing ORBI/Qwen text;
- Qwen chat special-token syntax.

## Exit gate

OSM-37C is GREEN only when the dedicated tokenizer workflow passes on both
Windows and Ubuntu with:

- pinned official model revision;
- exact tokenizer SHA256;
- official Python tokenizer reference generation;
- exact C++ encode parity;
- exact C++ decode parity with and without special tokens.

The normal Windows/Linux/Vulkan CI must remain GREEN independently.

## Next

**OSM-38 — real Qwen3-Next checkpoint pilot contract**

Connect the certified official tokenizer path to a real Qwen3-Next checkpoint
pilot while retaining bounded global matrices, streamed experts, host/GPU cache
policy and correctness-first execution.
