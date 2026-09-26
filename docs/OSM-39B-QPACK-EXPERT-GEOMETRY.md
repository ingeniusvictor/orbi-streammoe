# OSM-39B — QPACK expert-layout geometry contract

Status: **IMPLEMENTED / CLEAN-BASE CI CERTIFICATION PENDING**

## Goal

Freeze the exact routed-expert byte geometry for the Qwen3-Next QPACK target
before converting a full expert payload.

OSM-39B remains metadata-only. It reads `config.json` and performs checked
integer arithmetic; it does not download or transform tensor payload.

## Target quantization

The current ORBI target is affine 4-bit with group size 64:

```text
8 logical values -> 1 U32 packed word
64 logical values -> 1 scale + 1 bias
```

The geometry API remains parameterized so test fixtures and future formats can
supply another valid `bits/group_size` contract explicitly.

## Projection shapes

For each routed expert:

```text
gate_proj: [moe_intermediate_size, hidden_size]
up_proj:   [moe_intermediate_size, hidden_size]
down_proj: [hidden_size, moe_intermediate_size]
```

Each projection is stored as three contiguous sections:

```text
.weight  U32 packed words
.scales  F32 affine scales
.biases  F32 affine biases
```

The nine sections are strictly contiguous. The end of the final section is the
exact `expertStride`.

## Official pinned Qwen3-Next-80B-A3B-Instruct geometry

From the pinned official config:

```text
hidden_size             = 2048
moe_intermediate_size   = 512
num_experts             = 512
num_hidden_layers       = 48
target bits             = 4
target group_size       = 64
```

Derived shapes:

```text
gate/up weight: [512, 256] U32
gate/up groups: [512, 32]  F32 scale + F32 bias

down weight:    [2048, 64] U32
down groups:    [2048, 8]  F32 scale + F32 bias
```

Each projection occupies 655,360 bytes, so:

```text
expertStride      = 1,966,080 bytes
per-layer experts = 1,006,632,960 bytes
48 expert layers  = 48,318,382,080 bytes (45 GiB)
```

This figure covers routed-expert QPACK payload only. Dense/global tensors,
metadata and filesystem overhead are separate.

## Safety invariants

The derivation rejects:

- zero or unsupported bit widths;
- bit widths that do not divide a U32 word;
- zero group size;
- projection widths not divisible by packed-word capacity;
- projection widths not divisible by group size;
- uint64 arithmetic overflow;
- malformed/zero checkpoint dimensions;
- non-contiguous or malformed section geometry.

## Exit gate

OSM-39B is GREEN when:

- local fixture derives exact packed/group shapes and byte offsets;
- official pinned metadata derives the expected 4-bit/group-64 geometry;
- official Windows checkpoint workflow is GREEN;
- official Ubuntu checkpoint workflow is GREEN;
- normal Windows/Ubuntu CI is GREEN;
- tokenizer integration remains GREEN;
- Linux real Vulkan regression remains GREEN.

## Next

**OSM-39C — bounded single-expert conversion pilot**

Fetch only the source ranges required for one official expert, convert its
gate/up/down matrices to the OSM-39B QPACK layout, and verify dequantized
numerical error against the BF16 source without materializing a full layer.
