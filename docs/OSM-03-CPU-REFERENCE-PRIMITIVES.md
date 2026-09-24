# OSM-03 — CPU reference primitives

Status: **IMPLEMENTED / CI PENDING**

Reference baseline:

`leonickson1/Swiftlet@909c04213c9deb369dac0679d0872512cf3ab32e`

## Objective

Establish a small, portable CPU correctness oracle for the arithmetic that will later be implemented in Vulkan.

The reference path favors explicit behavior over speed.

## Implemented primitives

### MLX affine dequantization

Supported packed expert/dense formats:

- 4-bit affine;
- 8-bit affine.

The arithmetic matches Swiftlet's checkpoint reference:

```text
w[i] = scale[group(i)] * q[i] + bias[group(i)]
```

Packed values are read least-significant lane first from each uint32 word:

- 8 x 4-bit lanes per word;
- 4 x 8-bit lanes per word.

### Dense reference math

- row-major matrix-vector multiply;
- RMSNorm;
- softmax;
- sigmoid;
- SiLU;
- SwiGLU.

### MoE routing reference

`route_top_k`:

1. softmaxes router logits over all experts;
2. selects the highest `k` probabilities;
3. optionally renormalizes only the selected probabilities.

That reproduces the Qwen/Swiftlet `norm_topk_prob` behavior needed by the routed MoE path.

## Why this gate matters

Vulkan kernels cannot become the correctness oracle. Every accelerator implementation needs a deterministic CPU path to compare against.

OSM-03 therefore freezes the arithmetic before:

- streamed expert-cache integration;
- qpack expert dequantization;
- Vulkan GEMV;
- router/top-k GPU work;
- Gated DeltaNet kernels.

## Test coverage

The CI test validates:

- known Q4 affine bytes -> expected float values;
- known Q8 affine bytes -> expected float values;
- matrix-vector multiplication;
- weighted RMSNorm;
- raw top-k router probabilities;
- selected-probability renormalization;
- SwiGLU;
- rejection of unsupported affine bit widths.

## Exit gate

OSM-03 closes when the CPU reference primitive tests pass on both Windows and Linux.
