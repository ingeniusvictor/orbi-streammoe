#!/usr/bin/env python3
import argparse
import json
import pathlib

from fetch_qwen_conversion_slice import fnv1a64
from fetch_qwen_shard_headers import fetch_exact_range, shard_url

TENSORS = (
    "model.embed_tokens.weight",
    "lm_head.weight",
)


def locate(range_manifest, tensor_name):
    for shard in range_manifest["shards"]:
        meta = shard["selected_tensor_metadata"].get(tensor_name)
        if meta is not None:
            return shard, meta
    raise RuntimeError(f"tensor absent from range manifest: {tensor_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows", type=int, default=2)
    args = parser.parse_args()

    if args.rows <= 0:
        raise RuntimeError("--rows must be positive")

    manifest = json.loads(
        pathlib.Path(args.range_manifest).read_text(encoding="utf-8")
    )
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    result = []
    total = 0

    for tensor_name in TENSORS:
        shard, meta = locate(manifest, tensor_name)
        if meta["dtype"] != "BF16":
            raise RuntimeError(f"{tensor_name} must be BF16")
        shape = [int(v) for v in meta["shape"]]
        if len(shape) != 2 or shape[1] <= 0 or shape[0] < args.rows:
            raise RuntimeError(f"unexpected matrix shape for {tensor_name}: {shape}")

        row_bytes = shape[1] * 2
        begin, end = (int(v) for v in meta["data_offsets"])
        if end - begin != shape[0] * row_bytes:
            raise RuntimeError(f"source byte geometry mismatch: {tensor_name}")

        header_size = int(shard["header_size"])
        absolute_begin = 8 + header_size + begin
        absolute_end = absolute_begin + args.rows * row_bytes - 1

        data, remote_size = fetch_exact_range(
            shard_url(shard["filename"]),
            absolute_begin,
            absolute_end,
        )
        if remote_size != int(shard["file_size"]):
            raise RuntimeError(f"remote shard size changed: {shard['filename']}")
        if len(data) != args.rows * row_bytes:
            raise RuntimeError(f"short global row fetch: {tensor_name}")

        stem = tensor_name.replace(".", "_")
        filename = f"{stem}.rows0-{args.rows - 1}.bf16.bin"
        (output / filename).write_bytes(data)
        total += len(data)
        result.append({
            "source_tensor": tensor_name,
            "source_shape": shape,
            "rows": args.rows,
            "cols": shape[1],
            "source_file": filename,
            "source_byte_size": len(data),
            "source_fnv1a64": f"{fnv1a64(data):016x}",
            "shard": shard["filename"],
            "source_absolute_offset": absolute_begin,
        })
        print(
            f"{tensor_name}: shape={shape} rows={args.rows} "
            f"fetched={len(data)}"
        )

    payload = {
        "schema_version": 1,
        "model": manifest["model"],
        "snapshot": manifest["snapshot"],
        "rows": args.rows,
        "total_fetched_bytes": total,
        "tensors": result,
    }
    (output / "global-row-chunks.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"global row chunks: tensors={len(result)} total_fetched={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
