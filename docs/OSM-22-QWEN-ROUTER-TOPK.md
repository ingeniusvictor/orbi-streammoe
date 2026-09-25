# OSM-22 — Qwen router Top-K contract

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Freeze the exact sparse-router semantics required by Qwen3-Next before
introducing multi-expert weighted accumulation.

The reference path is:

```text
hidden state
    │
    ▼
router linear projection
[expert_count, hidden_size]
    │
    ▼
expert logits
    │
    ▼
global softmax
    │
    ▼
Top-K selection
    │
    ├─ expert id
    └─ probability
    │
    ▼
optional selected-weight renormalization
```

For the initial Qwen3-Next-80B-A3B compatibility target:

- hidden size: 2048
- routed experts: 512
- selected experts per token: 10
- `norm_topk_prob = true`

## Upstream compatibility

The frozen Swiftlet/Qwen reference computes the router projection, applies a
global softmax across all experts, selects Top-K, then divides each selected
probability by the selected probability sum when `norm_topk_prob` is enabled.

ORBI reproduces that behavior in the CPU correctness path.

The selected probability mass is retained separately for diagnostics. This is
useful because the normalized routed weights sum to one while the original
global-softmax mass generally does not.

## Determinism

ORBI uses a deterministic tie rule:

**equal probabilities prefer the lower expert id.**

This follows the behavior of the frozen Swiftlet CPU selection loop, which
visits experts in ascending id order and only replaces a selected candidate on
a strictly greater probability.

## API

`route_qwen_top_k_cpu(...)` receives:

- one hidden vector;
- one row-major router matrix `[experts, hidden]`;
- hidden size;
- expert count;
- Top-K;
- normalization policy.

It returns:

- raw router logits;
- ordered routed expert ids;
- routed weights;
- pre-renormalization selected probability mass.

## Certification

OSM-22 tests:

1. exact Top-2 order for a deterministic 4-expert router;
2. normalized routed weights sum to 1 when enabled;
3. non-normalized weights preserve global-softmax mass;
4. deterministic lower-id tie behavior;
5. target-scale 512-expert / Top-10 selection;
6. Qwen3-Next-80B architecture contract carries `norm_topk_prob=true`;
7. invalid router shapes are rejected.

## Boundary

OSM-22 is intentionally a CPU semantic oracle. It does not yet make the router
projection itself a Vulkan kernel.

The next gate can use these routed ids and weights to drive the
`VulkanResidentExpertCache` and combine multiple selected expert outputs.

## Next

**OSM-23 — weighted routed-expert execution**

For each selected expert:

```text
router pick
  -> resident expert cache
  -> expert output
  -> multiply by routing weight
  -> accumulate
```

The first gate should exclude the shared expert so routed-expert correctness is
isolated. Shared-expert + sigmoid gating can then be added separately.
