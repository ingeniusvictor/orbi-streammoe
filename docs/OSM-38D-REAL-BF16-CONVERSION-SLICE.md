# OSM-38D — bounded real BF16 conversion slice

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Cross from metadata-only checkpoint inspection into the first real bounded tensor
payload conversion without beginning a full Qwen3-Next-80B-A3B conversion.

The pilot uses the official final RMSNorm tensor:

- tensor: model.norm.weight
- shape: [2048]
- source dtype: BF16
- source payload: 4096 bytes
- converted dtype: F32
- converted payload: 8192 bytes

## Source contract

OSM-38D consumes the OSM-38C range manifest rather than rediscovering tensor
locations.

The official fetcher therefore already knows:

- exact shard filename;
- remote shard file size;
- safetensors header size;
- payload-relative tensor offsets.

The absolute tensor range is:

8 + header_size + data_begin

through:

8 + header_size + data_end - 1

The network request still requires HTTP 206 Partial Content through the strict
OSM-38C range helper.

Exactly 4096 payload bytes are downloaded.

## Independent oracle

The Python fetcher converts each little-endian BF16 word into IEEE-754 F32 bytes
by placing the BF16 bits in the high 16 bits of a 32-bit float representation.

It does not write the oracle F32 artifact. It records only:

- source FNV-1a 64 checksum;
- expected converted F32 FNV-1a 64 checksum.

The C++ conversion path must independently:

1. validate the source artifact size and manifest;
2. validate the source checksum;
3. decode BF16 to float;
4. serialize portable little-endian F32 bytes;
5. match the Python oracle checksum;
6. write the resulting 8192-byte artifact.

## Local correctness fixture

The portable unit test uses exact BF16 encodings for:

- 1.0;
- -2.0;
- 0.5.

It certifies deterministic F32 output bytes and rejects source corruption.

## Official gate

The official pinned Qwen gate requires:

- model and snapshot identity unchanged;
- tensor name model.norm.weight;
- shape [2048];
- source bytes exactly 4096;
- fetched bytes exactly 4096;
- output bytes exactly 8192;
- 2048 finite converted values;
- exact Python/C++ FNV parity.

## Scope boundary

OSM-38D is deliberately not a full QPACK repacker.

The repository currently has a mature QPACK reader/runtime but no general
checkpoint writer. This pilot freezes the real remote-range and BF16 conversion
semantics first, so a later repacker can build on a certified byte-level path.

## Exit gate

OSM-38D is GREEN when:

- local conversion tests pass on Windows and Ubuntu;
- official conversion slice passes on Windows and Ubuntu;
- normal numerical CI remains GREEN;
- Linux real Vulkan regression remains GREEN;
- tokenizer-integration remains GREEN;
- OSM-38C header/range gates remain GREEN.

## Next

**OSM-39A — QPACK conversion-plan contract**

Define a deterministic conversion plan mapping official sharded safetensors
tensor names and shapes into the ORBI/Swiftlet-compatible QPACK dense and expert
layout before downloading or converting large expert payloads.
