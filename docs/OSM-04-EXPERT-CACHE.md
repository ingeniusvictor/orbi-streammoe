# OSM-04 — portable expert cache and storage adapter

Status: **IMPLEMENTED / CI PENDING**

Reference baseline:

`leonickson1/Swiftlet@909c04213c9deb369dac0679d0872512cf3ab32e`

## Objective

Separate expert placement policy from platform I/O and GPU buffers.

This gives Windows and Android one shared cache policy while allowing each platform/backend to choose the best storage and buffer implementation later.

## Storage contract

`ExpertStorage` remains the platform-neutral boundary:

```text
ExpertCache
    │
    ▼
ExpertStorage
    ├─ QpackExpertStorage (portable correctness path)
    ├─ Windows overlapped reader (next)
    └─ POSIX/Android pread reader (later)
```

`QpackExpertStorage` adapts the OSM-02 qpack reader to the storage interface. It is intentionally serial today.

## Cache policy

The portable `ExpertCache` reproduces the important Swiftlet semantics:

- bounded by an explicit byte budget;
- fixed-size expert slots;
- lazy allocation;
- frequency history per expert, including across evictions;
- LFU victim selection;
- recency tie-break;
- lower slot index as the final deterministic tie-break;
- all members of one requested batch are protected from eviction by later members of the same batch;
- cache misses are sent to storage as one batch;
- a failed batch read clears every newly filled key so unread/corrupt bytes are never advertised as resident.

## Important ORBI hardening

ORBI rejects a batch containing more distinct experts than the cache can hold **before mutating cache state**.

The frozen Swiftlet configuration avoids this in normal operation by enforcing enough slots for routed top-k, but making the invariant explicit is safer for a cross-platform runtime.

## Why this matters

For Qwen3-Next-80B-A3B, each layer selects 10 routed experts. The production runtime must keep those simultaneously addressable while misses are loaded.

The cache policy itself does not need to know whether the slot memory is:

- ordinary CPU memory;
- pinned/host-visible Vulkan memory;
- a Windows staging allocation;
- an Android Vulkan allocation.

That separation is the foundation for one model runtime across both platforms.

## Test coverage

The OSM-04 test validates:

- lazy growth;
- cache hits without storage reads;
- LFU eviction;
- recency tie-break;
- same-batch slot protection;
- one storage batch for several misses;
- failure cleanup;
- preservation of earlier resident keys;
- rejection of an oversized batch before mutation.

## Exit gate

OSM-04 closes when Windows and Linux CI pass the cache policy suite.

Optimized OS-specific positional I/O is the next storage gate.
