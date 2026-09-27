# OSM-43C — official text-prompt to first-token gate

Status: **IMPLEMENTED / CI CONTRACT CERTIFICATION PENDING**

## Goal

Advance from OSM-43B's token-ID probe to a real UTF-8 text prompt using the
already-certified official Qwen tokenizer.

Production path:

    UTF-8 prompt file
      -> official tokenizer.json / tokenizers-cpp
      -> bounded prompt token IDs
      -> teacher-forced checkpoint execution
      -> 48-layer D-D-D-G decoder
      -> streamed QPACK experts
      -> final Vulkan RMSNorm
      -> streamed LM head
      -> first continuation token
      -> official tokenizer decode

## Hard bindings

OSM-43C requires:

- OSM-43B immutable first-token certification;
- the same checkpoint directory/package;
- model `Qwen/Qwen3-Next-80B-A3B-Instruct`;
- snapshot `f5e99a3698d364cf77584543481b778afee26177`;
- official `tokenizer.json` SHA256
  `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`;
- explicit `max_prompt_tokens`;
- explicit LM-head chunk rows;
- explicit host and Vulkan cache budgets.

## Native probe

The production executable is built with:

    -DORBI_STREAMMOE_ENABLE_TOKENIZERS_CPP=ON

and invoked as:

    orbi_streammoe_probe_text_first_token.exe ^
      <checkpoint_dir> ^
      <tokenizer_asset_dir> ^
      <prompt_file> ^
      <max_prompt_tokens> ^
      <lm_head_chunk_rows> ^
      <host_cache_bytes> ^
      <gpu_cache_bytes>

The prompt is file-backed so arbitrary UTF-8 text does not depend on shell
quoting semantics.

## Claim boundary

A successful real OSM-43C run can claim:

- official tokenizer used;
- real text prompt encoded;
- bounded teacher forcing through the official checkpoint;
- all 48 layers executed;
- first continuation token generated;
- generated token decoded by the official tokenizer.

It still does **not** claim:

- multi-token generation;
- chat-template correctness;
- conversational quality;
- sustained throughput;
- tokens per second.

## CI boundary

Normal CI certifies the Python evidence contract. The dedicated tokenizer
workflow builds the native probe with tokenizers-cpp enabled and keeps official
tokenizer parity GREEN.

Neither CI path can claim a real 80B text-prompt inference run. That exit gate
requires the OSM-43B-certified package on the designated Windows host.

## Next

**OSM-43D — bounded multi-token official generation gate**

Run a small explicitly bounded continuation, preserve exact token/logit/model
step evidence, and only then begin latency/RAM/VRAM measurement work.
