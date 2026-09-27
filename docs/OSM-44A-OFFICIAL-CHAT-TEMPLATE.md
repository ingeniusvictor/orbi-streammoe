# OSM-44A — official Qwen chat-template contract

Status: **IMPLEMENTED / AUTHORITATIVE CI CERTIFICATION PENDING**

## Goal

Certify the exact conversation serialization supplied by the official
`Qwen/Qwen3-Next-80B-A3B-Instruct` tokenizer assets before ORBI StreamMoE
implements or executes any native chat renderer.

OSM-43D proves bounded text generation. It does not prove that a structured
conversation with `system`, `user`, and `assistant` roles is serialized
the way the official Qwen model expects.

## Authoritative source

Model:

    Qwen/Qwen3-Next-80B-A3B-Instruct

Pinned snapshot:

    f5e99a3698d364cf77584543481b778afee26177

The existing tokenizer workflow downloads `tokenizer_config.json` directly
from that immutable revision.

OSM-44A loads those local assets with a pinned Hugging Face Transformers
implementation and calls `apply_chat_template(...)` as the authoritative
reference renderer.

## Reference vectors

The gate records rendered UTF-8 text and exact token IDs for:

1. user-only prompt with assistant generation prompt;
2. system + user prompt;
3. multi-turn system/user/assistant/user history;
4. completed user/assistant exchange without a generation prompt.

Each vector records:

- structured messages;
- `add_generation_prompt`;
- rendered text;
- rendered SHA256;
- exact token IDs;
- token count.

The reference additionally binds:

- `tokenizer.json` SHA256;
- `tokenizer_config.json` SHA256;
- raw `chat_template` SHA256;
- official model/revision;
- pinned Transformers version.

## Double-path oracle

For every case:

    apply_chat_template(..., tokenize=True)

must equal:

    encode(
      apply_chat_template(..., tokenize=False),
      add_special_tokens=False
    )

This catches accidental double-special-token insertion and serialization/token
disagreement.

## Claim boundary

OSM-44A may claim the official chat-template semantics are captured and
cryptographically bound.

It does **not** claim:

- a native ORBI chat renderer exists;
- native rendering matches the official template;
- chat-formatted model inference has run;
- conversational quality;
- throughput or memory performance.

## CI

The dedicated tokenizer workflow:

1. downloads the pinned official tokenizer assets;
2. installs pinned `transformers`;
3. generates the OSM-44A authoritative vectors;
4. validates the contract.

Windows and Ubuntu must both pass.

## Next

**OSM-44B — native Qwen chat renderer parity**

Implement the smallest portable native renderer necessary for the certified
official system/user/assistant contract and prove byte-for-byte rendered text
and token-ID parity against OSM-44A.
