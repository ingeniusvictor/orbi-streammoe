# OSM-43B — official first-token inference gate

Status: **IMPLEMENTED / CI CONTRACT CERTIFICATION PENDING**

## Goal

Execute one real checkpoint-backed Qwen3-Next-80B-A3B token-to-next-token step
after OSM-43A has certified that the full package loads in ORBI StreamMoE.

The native path is:

    token ID
      -> streamed embedding row
      -> 48-layer D-D-D-G decoder
      -> routed QPACK experts through bounded host/Vulkan caches
      -> final Vulkan RMSNorm
      -> streamed LM-head chunks
      -> greedy next-token ID

## Production probe

    orbi_streammoe_probe_first_token.exe ^
      <checkpoint_dir> ^
      <token_id> ^
      <lm_head_chunk_rows> ^
      <host_cache_bytes> ^
      <gpu_cache_bytes>

Cache and LM-head bounds are mandatory. OSM-43B deliberately has no guessed
production memory defaults.

## Certification

    python scripts/certify_qwen_first_token.py ^
      --runtime-load-certification D:\orbi\qwen80b\runtime-load-certification.json ^
      --checkpoint-dir D:\orbi\qwen80b\checkpoint ^
      --probe-executable build\Release\orbi_streammoe_probe_first_token.exe ^
      --token-id <TOKEN_ID> ^
      --lm-head-chunk-rows <ROWS> ^
      --host-cache-bytes <BYTES> ^
      --gpu-cache-bytes <BYTES> ^
      --output D:\orbi\qwen80b\first-token-certification.json

The certification is cryptographically bound to the immutable OSM-43A
certification file and therefore transitively to OSM-42D conversion evidence.

## Claims

A successful real production run may claim:

- one official checkpoint-backed model step executed;
- all 48 decoder layers executed;
- QPACK experts were streamed;
- final RMSNorm executed;
- LM head was streamed in bounded chunks;
- one greedy token ID was produced;
- explicit host/GPU cache limits were used.

OSM-43B does **not** claim:

- text prompt tokenization;
- multi-token generation;
- conversation quality;
- sustained throughput;
- tokens per second.

## CI boundary

Portable CI certifies the evidence contract and builds the real native probe.
Fixture success is not a real Qwen3-Next-80B-A3B first token.

The real OSM-43B exit gate requires the actual OSM-43A-certified Windows
checkpoint package and execution of the native probe on that host.

## Next

**OSM-43C — official text-prompt to first-token gate**

Bind the already-certified official tokenizer to the production checkpoint,
encode a real text prompt, teacher-force its token IDs through the runtime, and
produce the first continuation token without yet claiming sustained generation.
