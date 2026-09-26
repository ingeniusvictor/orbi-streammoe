# OSM-40F — full checkpoint package readiness

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Promote the two independently certified production conversion tracks into the
single runtime package contract required before a full Qwen3-Next-80B-A3B
conversion can be considered complete:

- OSM-39G routed-expert package;
- OSM-40B/40C/40D/40E dense/global streamed conversion.

OSM-40F does not add another quantizer or downloader. It is the final
correctness/readiness gate over already-converted artifacts.

## Promotion flow

```text
OSM-39G expert shell
    +
complete OSM-40B dense journal
    +
model.safetensors
        ↓
reopen + verify every dense journal chunk
        ↓
promote config quantization contract
        ↓
declare dense payload in manifest
        ↓
write full-checkpoint provenance
        ↓
reopen through QpackReader
        ↓
reopen through QpackMlxCheckpoint
        ↓
validate global Qwen inventory
        ↓
FULL CHECKPOINT READY
```

## Required runtime globals

The final gate requires:

- `model.norm.weight`;
- quantized `model.embed_tokens`;
- quantized `lm_head`.

The embedding and LM-head quantization metadata must agree with the requested
affine-Q4 contract.

## Dense journal invariant

The streamed dense output is not trusted because the file merely exists.

OSM-40F reconstructs the dense tensor specs from the production safetensors
header, reopens the OSM-40B journal and requires:

- every tensor row complete;
- every committed chunk checksum verified against disk;
- production `SafetensorsReader` inventory closure.

Only then can package metadata change from `expert-shell` to
`full-checkpoint`.

## Deterministic package promotion

OSM-40F adds:

- `model.safetensors` to `manifest.json`;
- runtime affine quantization metadata to `config.json`;
- `full-checkpoint-provenance.json`;
- `packageStage = full-checkpoint`.

Existing compatible metadata is idempotent. Conflicting quantization metadata
is rejected instead of overwritten silently.

## Exit gate

Windows and Ubuntu CI must prove:

- a valid OSM-39G expert shell can be promoted;
- an incomplete dense journal blocks promotion;
- complete dense chunks are reverified;
- manifest/config/provenance promotion is deterministic;
- conflicting quantization metadata is rejected;
- the resulting package reopens with production `QpackReader`;
- the resulting dense payload reopens with `QpackMlxCheckpoint`;
- required global Qwen tensor inventory is present;
- all prior tokenizer, checkpoint and real-Vulkan regressions remain GREEN.

## Production authorization boundary

Passing OSM-40F means the conversion **architecture and final package contract**
are certified for a complete production conversion. It does not claim that CI
has downloaded and converted the entire 80B checkpoint.

A real full conversion remains an operator action using the certified OSM-39F
expert workflow and OSM-40E dense controller with sufficient local storage.

## Next

**OSM-41A — production conversion preflight / operator execution manifest**

Generate one immutable operator manifest that records source snapshot, output
paths, disk budgets, expert/dense controller parameters and package finalizer
inputs before authorizing the first full-scale conversion run.
