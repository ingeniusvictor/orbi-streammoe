# OSM-43D — bounded multi-token official generation gate

Status: **IMPLEMENTED / CI CONTRACT CERTIFICATION PENDING**

## Goal

Advance from OSM-43C's first continuation token to a small, explicitly bounded
autoregressive sequence using the same official tokenizer, full checkpoint,
Vulkan runtime and streamed-expert cache path.

## Bounds

The production gate requires:

- non-empty UTF-8 prompt file;
- explicit `max_prompt_tokens`;
- `2 <= max_new_tokens <= 8`;
- explicit LM-head chunk rows;
- explicit host expert-cache bytes;
- explicit Vulkan expert-cache bytes.

The gate must actually produce at least two tokens. Early EOS before token 2
does not certify OSM-43D.

## Runtime accounting

For `P` prompt tokens and `G` generated tokens:

    model_steps = P + G - 1

The evidence records every generated token ID and greedy logit, decoded output,
stop reason, cache statistics, Vulkan device evidence and all declared bounds.

## Parent evidence

OSM-43D is immutably bound to the OSM-43C text-first-token certification and
therefore transitively to OSM-43B, OSM-43A and OSM-42D.

## Claim boundary

A successful real run may claim bounded official multi-token generation.

It does not yet claim:

- chat-template correctness;
- assistant-role formatting;
- sustained/representative throughput;
- tokens per second;
- RAM or VRAM performance measurements;
- interactive usability.

## CI boundary

Normal CI certifies the evidence contract. The dedicated tokenizer workflow
builds the real native probe with tokenizers-cpp enabled on Windows and Ubuntu.

A real 80B OSM-43D certification still requires the designated Windows host and
the actual previously-certified checkpoint package.

## Next

**OSM-44A — official chat-template contract**

Bind Qwen's official chat-template semantics without changing the numerical
runtime. Performance instrumentation follows only after prompt/chat semantics
are certified.
