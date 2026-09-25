# OSM-21 — Vulkan resident expert cache

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Add a second-level expert cache that keeps complete MoE experts resident in
reusable Vulkan buffers under an explicit logical byte budget.

The two cache levels are now:

```text
SSD / qpack
    ↓
L1 Host ExpertCache
    ↓
L2 VulkanResidentExpertCache
    ↓
VulkanResidentExpert
```

## Key

The GPU-side key is:

```text
(layer, expert)
```

matching the host expert cache and Qwen sparse-MoE routing identity.

## Hit path

```text
router/request
     ↓
(layer, expert)
     ↓
GPU cache HIT
     ↓
execute existing VulkanResidentExpert
```

A hit does not touch the host cache and does not read the qpack layer file.

## Miss path

```text
GPU cache MISS
     ↓
host ExpertCache.fetch(...)
     ↓
qpack / storage when host cache misses
     ↓
VulkanResidentExpert::create(...)
     ↓
GPU cache admission
```

## Eviction

OSM-21 mirrors the upstream-inspired policy already used by the host cache:

1. lowest historical access frequency;
2. oldest use tick as tie-break;
3. deterministic key tie-break.

Frequency history survives eviction so a repeatedly useful expert retains its
historical preference if it returns later.

## Budget

The cache uses a logical byte budget based on the reusable Vulkan resources
owned by one resident expert:

- packed Q4 gate/up/down bytes;
- decoded float scale/bias buffers;
- reusable input/gate/up/hidden/output activation buffers.

This is **not yet a claim about physical VRAM allocation**. Vulkan memory
alignment, heap granularity, driver bookkeeping, and unified-memory behavior can
make physical allocation differ from the accounted byte count.

The temporary creation of a new expert may also briefly coexist with old
residents before eviction. A later memory-planner gate can remove that transient
overlap if real-device measurements show it matters.

## Certification scenario

The test uses three experts and a budget for two resident experts:

```text
load E0  -> miss
load E1  -> miss
touch E0 -> hit
load E2  -> miss
```

Because E0 has frequency 2 while E1 has frequency 1, E1 must be evicted.

The host cache is intentionally limited to one slot. The certification also
checks that a Vulkan hit for E0 does not reload E0 into the host cache.

## Exit gate

OSM-21 is GREEN when:

- Windows and Linux compile/test;
- Linux executes the cache path with real Vulkan;
- hit/miss/load/eviction counters are deterministic;
- LFU + recency eviction chooses the expected victim;
- GPU hits bypass the host cache;
- resident-byte accounting stays within the configured budget;
- outputs from retained/new experts match independent CPU oracles.

## Next

**OSM-22 — router Top-K contract**

Implement the Qwen sparse router reference path and select the top routed
experts plus normalized routing weights. Then connect those selections to the
resident expert cache without yet combining expert outputs.
