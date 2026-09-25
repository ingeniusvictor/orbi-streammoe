# OSM-23 — weighted routed-expert execution

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Connect the Qwen Top-K router from OSM-22 to the two-level streamed expert
cache from OSM-21 and produce the weighted routed-MoE output.

The correctness path is now:

```text
hidden state
    │
    ├──────────────► CPU router projection
    │                    │
    │                    ▼
    │               softmax + Top-K
    │                    │
    │          expert ids + weights
    │                    │
    ▼                    ▼
VulkanResidentExpertCache
    │
    ├─ expert A -> Vulkan expert MLP -> host output * weight A
    ├─ expert B -> Vulkan expert MLP -> host output * weight B
    └─ ...
                         │
                         ▼
                 host accumulation
                         │
                         ▼
                  routed MoE output
```

## Scope

OSM-23 is deliberately correctness-first.

It does **not** yet:

- execute the router projection on Vulkan;
- accumulate selected expert outputs on Vulkan;
- execute the Qwen shared expert;
- apply the shared-expert sigmoid gate.

Those are separate gates so numerical and cache correctness remain easy to
isolate.

## Routing semantics

The selected experts come from the OSM-22 contract:

1. router linear projection;
2. global softmax;
3. Top-K selection;
4. selected-probability renormalization when configured.

For Qwen3-Next-80B-A3B the target contract remains 512 experts / Top-10 with
`norm_topk_prob=true`.

## Expert execution

Each routed expert is resolved through:

`VulkanResidentExpertCache::run(layer, expert, hidden)`

Therefore a selected expert may be:

- a GPU-cache hit;
- loaded from host cache;
- streamed from qpack/SSD and then promoted to Vulkan.

The weighted accumulator is independent of where the expert came from.

## Certification

The OSM-23 test uses:

- 3 routed experts;
- Top-2 routing;
- a Vulkan cache budget for exactly 2 resident experts;
- a one-slot host cache;
- BF16 affine metadata;
- Q4 gate/up/down expert projections.

The router is constructed so expert 1 ranks first and expert 2 ranks second.

The test:

1. computes the CPU router result;
2. computes independent CPU expert outputs;
3. builds the expected weighted sum;
4. executes the routed MoE through Vulkan;
5. compares the weighted result to the independent CPU oracle;
6. repeats the same token and requires both selected experts to hit the Vulkan cache.

## Exit gate

OSM-23 is GREEN when:

- Windows and Linux compile/test;
- Linux executes the routed path with real Vulkan;
- selected expert ids and weights match the CPU router oracle;
- first execution loads exactly the selected experts;
- repeated execution reuses those Vulkan residents;
- weighted output matches the independent CPU oracle;
- no unselected expert is loaded.

## Next

**OSM-24 — Vulkan weighted expert accumulation**

Remove the per-expert output readback and host accumulation:

```text
resident expert output buffers
        ↓
weight + accumulate on Vulkan
        ↓
single final readback
```

This will make Top-10 routed execution much closer to the production path
needed by Qwen3-Next-80B-A3B.
