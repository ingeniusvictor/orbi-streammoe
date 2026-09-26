#!/usr/bin/env python3
import argparse
import json
import pathlib
import struct

from fetch_qwen_shard_headers import fetch_exact_range, shard_url

TENSOR_NAME = "model.norm.weight"
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(data: bytes) -> int:
    value = FNV_OFFSET
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def bf16_to_f32_bytes(data: bytes) -> bytes:
    if len(data) % 2:
        raise RuntimeError("BF16 source byte length must be even")
    out = bytearray()
    for offset in range(0, len(data), 2):
        word = struct.unpack_from("<H", data, offset)[0]
        out.extend(struct.pack("<I", word << 16))
    return bytes(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    range_manifest_path = pathlib.Path(args.range_manifest)
    range_manifest = json.loads(range_manifest_path.read_text(encoding="utf-8"))

    selected = None
    selected_shard = None
    selected_file_size = None
    selected_header_size = None

    for shard in range_manifest["shards"]:
        metadata = shard["selected_tensor_metadata"]
        if TENSOR_NAME in metadata:
            selected = metadata[TENSOR_NAME]
            selected_shard = shard["filename"]
            selected_file_size = int(shard["file_size"])
            selected_header_size = int(shard["header_size"])
            break

    if selected is None:
        raise RuntimeError(f"{TENSOR_NAME} is absent from range manifest")
    if selected["dtype"] != "BF16":
        raise RuntimeError(f"{TENSOR_NAME} must be BF16")
    if selected["shape"] != [2048]:
        raise RuntimeError(f"{TENSOR_NAME} official shape changed")

    begin, end = (int(v) for v in selected["data_offsets"])
    if begin >= end:
        raise RuntimeError("invalid source tensor data range")

    source_size = end - begin
    if source_size != 4096:
        raise RuntimeError(
            f"unexpected {TENSOR_NAME} BF16 byte size: {source_size}"
        )

    absolute_begin = 8 + selected_header_size + begin
    absolute_end = 8 + selected_header_size + end - 1

    source_bytes, remote_size = fetch_exact_range(
        shard_url(selected_shard),
        absolute_begin,
        absolute_end,
    )
    if remote_size != selected_file_size:
        raise RuntimeError("remote shard size disagrees with header manifest")
    if len(source_bytes) != source_size:
        raise RuntimeError("bounded tensor range returned wrong byte count")

    expected_f32 = bf16_to_f32_bytes(source_bytes)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = "model.norm.weight.bf16.bin"
    (output_dir / source_name).write_bytes(source_bytes)

    manifest = {
        "schema_version": 1,
        "model": range_manifest["model"],
        "snapshot": range_manifest["snapshot"],
        "tensor_name": TENSOR_NAME,
        "shard": selected_shard,
        "source_dtype": "BF16",
        "source_shape": [2048],
        "source_absolute_offset": absolute_begin,
        "source_byte_size": source_size,
        "fetched_bytes": len(source_bytes),
        "source_file": source_name,
        "source_fnv1a64": f"{fnv1a64(source_bytes):016x}",
        "output_dtype": "F32",
        "output_byte_size": len(expected_f32),
        "expected_f32_fnv1a64": f"{fnv1a64(expected_f32):016x}",
    }
    (output_dir / "conversion-slice.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"conversion slice: tensor={TENSOR_NAME} shard={selected_shard} "
        f"offset={absolute_begin} fetched={len(source_bytes)} "
        f"output={len(expected_f32)}"
    )
    print(f"source_fnv1a64={manifest['source_fnv1a64']}")
    print(f"expected_f32_fnv1a64={manifest['expected_f32_fnv1a64']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
