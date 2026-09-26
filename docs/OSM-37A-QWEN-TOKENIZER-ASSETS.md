# OSM-37A — Qwen tokenizer asset contract

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Bind the real tokenizer asset semantics required by Qwen3-Next before choosing
or implementing a concrete tokenizer engine.

OSM-37A does not tokenize text. It validates that the files handed to a future
adapter describe the tokenizer/model pair we actually intend to execute.

## Official target observations

For the official `Qwen/Qwen3-Next-80B-A3B-Instruct` assets reviewed during
this gate:

- model type is `qwen3_next`;
- model vocabulary size is 151936;
- tokenizer class is `Qwen2Tokenizer`;
- `tokenizer.json` is the canonical executable tokenizer artifact;
- tokenizer semantics are BPE + NFC normalization + Split/ByteLevel
  pretokenization + ByteLevel decoding;
- `<|im_end|>` is tokenizer EOS;
- `<|endoftext|>` is tokenizer PAD;
- generation stop IDs include 151645 and 151643;
- no BOS token is automatically prepended by tokenizer configuration.

The runtime contract is structural rather than hard-coded to these numeric
values so compatible checkpoints can still be validated.

## Required files

```text
config.json
tokenizer.json
tokenizer_config.json
generation_config.json
```

`vocab.json` and `merges.txt` are detected when present, but OSM-37A treats
`tokenizer.json` as the primary portable artifact.

## Validated invariants

`inspect_qwen_tokenizer_assets(...)` validates:

- `model_type == qwen3_next`;
- Qwen2-compatible tokenizer class;
- BPE model;
- NFC normalizer;
- Split + ByteLevel pretokenization;
- ByteLevel decoder;
- tokenizer EOS/PAD token-to-ID resolution;
- model/tokenizer EOS agreement;
- generation stop-token domain;
- generation/tokenizer PAD agreement;
- every declared special ID fits the model vocabulary.

## Padded vocabulary rule

The tokenizer token-ID domain is allowed to be **smaller than** the model
LM-head vocabulary:

```text
tokenizer_id_domain_size <= model_vocab_size
```

This matters because model vocabularies may contain padded/reserved output rows.
The inverse is forbidden: a tokenizer must never emit an ID for which the model
has no embedding/LM-head row.

The contract records `padded_model_vocab_rows` explicitly.

## OSM-36C compatibility correction

The text-session preflight follows the same rule. A tokenizer-reported ID domain
smaller than the model vocabulary is not rejected solely for being smaller.
Actual prompt/stop IDs are still checked against model vocabulary, and decode
continues to reject generated IDs the tokenizer cannot represent.

## Exit gate

OSM-37A is GREEN when Windows + Ubuntu certify:

- valid Qwen3-Next asset inspection;
- normalized generation stop-token arrays;
- EOS/PAD mapping;
- optional vocab/merges discovery;
- padded model-vocabulary compatibility;
- tokenizer domain overflow rejection;
- wrong model/tokenizer/BPE/normalizer/pretokenizer rejection;
- special-token range and agreement guards;
- missing/malformed asset guards.

No new numerical kernel is introduced by this gate.

## Next

**OSM-37B — concrete tokenizer engine evaluation + adapter**

Evaluate a tokenizer.json-compatible backend against the certified contract.
A leading candidate is MLC `tokenizers-cpp`, which already targets Windows,
Linux and Android, but dependency/build cost must be measured before adoption.
