# OSM-25A — Qwen shared-expert semantics

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Freeze the exact Qwen3-Next shared-expert math before adding its Vulkan path.

The routed experts are already complete through OSM-24C. Qwen3-Next adds one
always-on shared expert:

```text
shared = down(SiLU(gate(x)) * up(x))
scale  = sigmoid(shared_expert_gate · x)
output = scale * shared
```

This result is added to the weighted routed-expert result.

## Why separate this gate

The shared expert is semantically different from routed experts:

- it is always executed;
- it is not selected by Top-K;
- it has a separate scalar gate vector;
- its weights should ultimately live with the resident dense model core rather
  than in the streamed routed-expert pool.

OSM-25A therefore defines only the portable CPU oracle and shape contract.

## API

`run_shared_expert_cpu(...)` accepts:

- hidden vector;
- gate/up/down row-major weights;
- one scalar gate vector;
- hidden/intermediate dimensions.

It validates all geometry before execution.

## Exit gate

OSM-25A is GREEN when:

- Windows and Linux compile/test;
- shared gate equals `sigmoid(dot(gate_vector, hidden))`;
- gate/up/SwiGLU/down math matches the independent reference calculation;
- malformed shapes fail cleanly.

## Next

**OSM-25B — Vulkan shared expert**

Keep shared gate/up/down weights resident in Vulkan, execute from the same shared
hidden buffer used by OSM-24C, compute/apply the sigmoid scalar gate, and add the
shared result into the existing Vulkan MoE accumulator before the one final
download.
