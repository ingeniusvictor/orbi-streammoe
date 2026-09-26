# OSM-42A — official production rehearsal / dry-run evidence

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Exercise the complete production authorization and operator-planning boundary against
the pinned official Qwen3-Next-80B-A3B-Instruct checkpoint before any full 80B
conversion is authorized on Windows hardware.

OSM-42A deliberately does **not** execute the four production phases. It proves that
the official source identity, full-scope OSM-41A manifest, OSM-41D exact argv plan,
OSM-41E read-only preview and bounded official source evidence agree reproducibly.

## Official source pin

The rehearsal is bound to:

- model: `Qwen/Qwen3-Next-80B-A3B-Instruct`
- snapshot: `f5e99a3698d364cf77584543481b778afee26177`

The dense stream inventory must carry the same model and snapshot. Any drift aborts
before an operator state can be created.

## Rehearsal bundle

The command:

    python scripts/rehearse_qwen_production.py \
      --metadata-dir build/qwen-official-metadata \
      --dense-inventory build/qwen-official-dense-stream-inventory.json \
      --expert-slice-dir build/qwen-official-expert-range \
      --dense-slice-dir build/qwen-official-dense-pilot \
      --output-dir build/qwen-osm42a-rehearsal \
      --bin-dir build/cmake

produces immutable:

- `preflight-request.json`
- `execution.json`
- `operator-state.json.commands.json`
- `rehearsal-bundle.json`

The evidence bundle records:

- official model and snapshot;
- official metadata and dense-inventory SHA-256;
- model geometry;
- OSM-41A execution ID and manifest SHA-256;
- OSM-41D command-plan SHA-256 and ordered phases;
- OSM-41E first-phase preview;
- streaming SHA-256 and byte counts for bounded official expert and dense slices;
- explicit proof that no operator state or OSM-41F receipt directory was created.

## Safety boundary

OSM-42A is a production **rehearsal**, not a conversion.

It must not:

- initialize the durable OSM-41B state machine;
- emit OSM-41F phase receipts;
- download or dequantize the complete 80B checkpoint;
- claim successful prompt-to-token inference;
- claim Windows hardware performance.

The bounded source evidence directories are capped by an explicit byte budget and
are hashed in streaming chunks.

## Exit gate

OSM-42A is GREEN when Windows and Ubuntu prove:

- the pinned official metadata can generate a full-source OSM-41A manifest;
- the complete official dense source inventory can be bound to exact safetensors
  header geometry;
- the immutable OSM-41D command plan covers all four ordered phases;
- OSM-41E preview selects `expert_conversion` without creating operator state;
- bounded official expert and dense slices are present and SHA-256 evidenced;
- the bundle is byte-for-byte reproducible on idempotent rerun;
- model/snapshot drift is rejected;
- inherited production, checkpoint and real-Vulkan gates remain GREEN.

## Next

**OSM-42B — Windows full-conversion hardware preflight / authorization pack**

Turn the OSM-42A rehearsal evidence into an explicit machine-specific preflight for
the first full official conversion: free SSD budget, filesystem/path checks,
available RAM, Vulkan adapter evidence, resumable work directories and an operator
authorization record. It still must not claim full inference until the resulting
checkpoint has been converted, loaded and exercised end to end.
