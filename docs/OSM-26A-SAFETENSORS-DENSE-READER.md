# OSM-26A — safetensors dense qpack reader

Status: **IMPLEMENTED / CERTIFICATION PENDING**

## Goal

Begin replacing synthetic dense-weight fixtures with the real dense payload
inside a Swiftlet-compatible qpack container.

A qpack v1 contains a resident `model.safetensors` file in addition to the
streamed expert layer blobs. OSM-26A implements the file-format reader and qpack
binding for that dense payload.

## Safetensors contract

`SafetensorsReader` parses the 8-byte little-endian header length, the JSON
tensor directory, and the raw payload offsets.

Construction validates header bounds, tensor bounds, supported dtype widths,
shape-product overflow, and exact agreement between each tensor's dtype/shape
and its declared byte range.

## Large-file behavior

The dense file is not loaded into RAM during construction. Only the JSON header
is read. Tensor bytes are fetched explicitly by file offset when requested.

The reader also exposes `absolute_offset(name)` so later Windows, POSIX, and
Vulkan binders can map or stream large tensors without materializing the whole
dense checkpoint.

## Typed reads

OSM-26A supports typed conversion for F32, F16, BF16, and U32. Other common
integer dtypes are recognized for element-width validation but do not yet have
typed conversion helpers.

## qpack binding

`QpackDenseReader` requires `model.safetensors` to be declared in
`manifest.json`, checks its actual size, and opens it as a file-backed
safetensors payload.

Tensor resolution first tries the requested name directly and then the
`language_model.` prefix used by some Qwen checkpoint layouts.

## Exit gate

OSM-26A is GREEN when Windows and Linux compile/test, typed values decode
correctly, absolute offsets are correct, malformed tensor ranges are rejected,
qpack manifest size validation works, and the `language_model.` fallback is
certified.

## Next

OSM-26B will parse the MLX affine quantization block from `config.json` and
bind module triplets `.weight/.scales/.biases`, including per-module
quantization overrides.
