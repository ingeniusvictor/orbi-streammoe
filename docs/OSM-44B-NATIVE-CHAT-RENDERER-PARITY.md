# OSM-44B — native Qwen chat renderer parity

Status: **IMPLEMENTED / STACKED CI CERTIFICATION PENDING**

## Goal

Implement the smallest portable native renderer that exactly reproduces the
OSM-44A official Qwen chat-template contract for the role/content subset ORBI
needs before chat-formatted inference.

## Supported subset

OSM-44B supports:

- string content;
- roles `system`, `user`, `assistant`;
- multi-turn histories;
- `add_generation_prompt`;
- plain assistant-role string content exactly as rendered by the pinned official template.

It deliberately rejects or leaves for later milestones:

- `tool` role;
- tool calls;
- tool responses;
- tools schema injection;
- multimodal content arrays;
- explicit structured `reasoning_content` fields.

## Parity gate

The dedicated tokenizer workflow first generates OSM-44A vectors using the
official Hugging Face renderer.

The native C++ test then requires, for every vector:

1. byte-for-byte rendered UTF-8 equality;
2. exact token-ID equality after encoding with the certified
   `QwenTokenizersCppTokenizer`.

This avoids hard-coding expected token IDs in the native test while still
binding it to the pinned official snapshot.

## Architectural boundary

The renderer has no Vulkan/model dependency. It converts structured messages
into text only.

This keeps chat semantics independently testable from:

- model execution;
- expert streaming;
- KV/DeltaNet state;
- performance instrumentation.

## Claim boundary

OSM-44B may claim native rendering parity for the certified
system/user/assistant subset.

It does not yet claim:

- tool calling parity;
- structured reasoning-field semantics beyond ordinary string content;
- chat-formatted model inference;
- conversational quality;
- tokens/s or memory performance.

## Next

**OSM-44C — chat-formatted bounded generation gate**

Feed native OSM-44B rendered messages through the OSM-43D bounded generation
path and certify the first real structured conversation on the official
checkpoint.
