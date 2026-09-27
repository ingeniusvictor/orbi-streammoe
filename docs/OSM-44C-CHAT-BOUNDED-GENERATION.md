# OSM-44C — chat-formatted bounded generation gate

Status: **IMPLEMENTED / STACKED CI CERTIFICATION PENDING**

## Goal

Take structured `system/user/assistant` messages through the OSM-44B native
renderer and then through the already-certified OSM-43D bounded generation path.

This is the first gate that joins conversation semantics and checkpoint
execution without introducing tool calling or performance claims.

## Production path

    messages.json
      -> OSM-44B native Qwen chat renderer
      -> official Qwen tokenizer (add_special_tokens=false)
      -> bounded teacher forcing
      -> 48-layer decoder
      -> streamed QPACK experts
      -> bounded autoregressive generation (2..8 tokens)
      -> decoded continuation

## Input contract

The messages file is:

    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
      ]
    }

Only the OSM-44B-certified string-content roles are accepted.

Explicit limits:

- max messages;
- max prompt tokens;
- 2..8 new tokens;
- LM-head chunk rows;
- host expert-cache bytes;
- Vulkan expert-cache bytes.

## Evidence chain

OSM-44C binds:

- immutable OSM-43D bounded-generation certification;
- OSM-44A authoritative chat-template reference;
- exact tokenizer/config/chat-template hashes;
- messages-file SHA256;
- native probe output.

## Claim boundary

A successful real run may claim:

- native chat renderer used;
- official tokenizer used;
- structured chat prompt executed;
- full 48-layer checkpoint inference;
- bounded multi-token chat continuation.

It does not claim:

- tool calling;
- tools schema rendering;
- representative conversation quality;
- tokens per second;
- RAM/VRAM performance.

## CI boundary

Portable CI validates the certification contract and compiles the probe.
The tokenizer workflow builds the real probe with tokenizers-cpp enabled.

Real 80B OSM-44C certification still requires the designated Windows machine
and the previously certified checkpoint package.

## Next

**OSM-45A — runtime performance instrumentation contract**

After chat semantics are certified, add measurements for wall time, token
latency, host RAM, cache residency and Vulkan memory without yet claiming a
portable performance target.
