# OSM-43A — official full-checkpoint runtime load gate

Status: **IMPLEMENTED / CI CONTRACT CERTIFICATION PENDING**

## Goal

Prove that an OSM-42D-certified full official Qwen3-Next-80B-A3B package can be
opened by the real ORBI StreamMoE runtime and can instantiate the checkpoint
model shell on Vulkan.

This milestone is intentionally a **load gate**, not an inference gate.

## Native runtime probe

The new native executable:

    orbi_streammoe_probe_checkpoint_load <checkpoint_dir>

performs the real runtime construction path:

    QpackReader
      -> QpackMlxCheckpoint
      -> parse_qwen3_next_dense_config
      -> VulkanComputeContext
      -> QwenCheckpointModelShell::create

Success requires a valid Vulkan compute device and a valid checkpoint-backed
decoder stack.

The probe reports the runtime device plus model geometry and explicitly records
that no inference or token generation occurred.

## OSM-42D binding

The production entrypoint is:

    python scripts/certify_qwen_runtime_load.py ^
      --conversion-certification D:\orbi\qwen80b\windows-conversion-certification.json ^
      --checkpoint-dir D:\orbi\qwen80b\checkpoint ^
      --probe-executable build\Release\orbi_streammoe_probe_checkpoint_load.exe ^
      --output D:\orbi\qwen80b\runtime-load-certification.json

The gate refuses to run unless:

- the conversion evidence is an OSM-42D certification;
- the official model is Qwen/Qwen3-Next-80B-A3B-Instruct;
- the pinned snapshot is f5e99a3698d364cf77584543481b778afee26177;
- full official conversion is certified complete;
- the full package is deep-audited;
- the requested checkpoint directory exactly matches the OSM-42D package path;
- the native runtime probe succeeds;
- the loaded decoder stack exposes exactly 48 layers;
- Vulkan initialization and decoder-stack creation are confirmed.

## Claim boundary

OSM-43A may claim:

- the certified official full checkpoint opens in ORBI StreamMoE;
- QPACK and dense metadata bind successfully;
- Vulkan initializes;
- the checkpoint-backed 48-layer decoder shell is created.

It does **not** claim:

- prompt inference;
- LM-head execution;
- first-token generation;
- multi-token generation;
- RAM/VRAM performance;
- tokens per second.

## CI boundary

Portable CI validates the wrapper contract and compiles the real native probe.
The existing Linux real-Vulkan gate continues to validate Vulkan availability
for native runtime components.

Only a run against the actual OSM-42D-certified Windows package satisfies the
real production OSM-43A exit gate.

## Next

**OSM-43B — official first-token inference gate**

Execute one real token-to-next-token step through:

    streamed embedding
      -> 48-layer D-D-D-G decoder
      -> final Vulkan RMSNorm
      -> streamed LM head
      -> greedy next token

and record bounded RAM/VRAM/SSD evidence without yet claiming sustained
multi-token generation performance.
