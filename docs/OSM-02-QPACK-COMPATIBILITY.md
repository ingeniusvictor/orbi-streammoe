# OSM-02 — qpack portable reader

Status: **IMPLEMENTED / CI PENDING**

Reference compatibility target:

`leonickson1/Swiftlet@909c04213c9deb369dac0679d0872512cf3ab32e`

## Objective

Read and validate Swiftlet qpack v1 containers from portable C++ without Swift, Foundation or Metal.

This gate intentionally does **not** implement optimized platform I/O or GPU execution. It freezes the on-disk compatibility contract first.

## Supported contract

ORBI StreamMoE now models:

- `manifest.json`
  - `magic == "QPACK"`
  - `version == 1`
  - model/source metadata
  - optional expert `quantBits` + `quantGroupSize`
  - declared file sizes
- `packed_experts/layout.json`
  - expert count
  - layer count
  - fixed expert stride
  - section name/dtype/shape/offset/size
  - linear/full-attention layer flags
- `packed_experts/layer_XX.bin`
  - exactly `expertCount * expertStride` bytes per layer
  - expert N starts at `N * expertStride`

## Correctness checks

Opening a container rejects:

- wrong qpack magic;
- unsupported manifest version;
- zero layer/expert/stride values;
- empty sections;
- section ranges outside the expert stride;
- mismatched `linearLayers` length;
- missing layer files;
- layer byte size disagreement;
- manifest-declared size disagreement;
- half-specified quantization metadata.

Expert reads reject:

- invalid layer index;
- invalid expert index;
- destination buffers smaller than one stride;
- short reads.

## Current I/O implementation

OSM-02 uses portable `std::ifstream + seekg + read` deliberately.

This is **not** the final streaming path.

OSM-04 will replace the storage implementation behind the portable contract with:

- Windows overlapped `ReadFile`;
- POSIX/Android `pread`/equivalent;
- batched reads;
- bounded cache slots.

Keeping OSM-02 simple gives us a clean byte-for-byte correctness oracle.

## CI fixture

The test creates a synthetic qpack with two layer files and three experts per layer, then verifies that `read_expert(layer=1, expert=2)` returns exactly the 16 bytes stored at its fixed-stride offset.

Negative tests cover corrupt magic, truncated layer files and bounds handling.

## Exit gate

OSM-02 closes when Windows and Linux CI both pass the qpack compatibility test. Android-specific I/O is not required to close this gate.
