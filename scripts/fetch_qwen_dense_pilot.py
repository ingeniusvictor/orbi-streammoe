#!/usr/bin/env python3
import argparse
import json
import pathlib

from fetch_qwen_conversion_slice import fnv1a64
from fetch_qwen_shard_headers import fetch_exact_range, shard_url

TENSORS = (
    "model.norm.weight",
    "model.layers.0.mlp.shared_expert_gate.weight",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    range_manifest = json.loads(
        pathlib.Path(args.range_manifest).read_text(encoding="utf-8")
    )
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_by_name = {
        shard["filename"]: shard for shard in range_manifest["shards"]
    }
    items = []

    for tensor_name in TENSORS:
        selected = None
        selected_shard = None
        for shard in range_manifest["shards"]:
            metadata = shard["selected_tensor_metadata"]
            if tensor_name in metadata:
                selected = metadata[tensor_name]
                selected_shard = shard["filename"]
                break

        if selected is None or selected_shard is None:
            raise RuntimeError(
                f"dense pilot tensor absent from range manifest: {tensor_name}"
            )
        if selected["dtype"] != "BF16":
            raise RuntimeError(f"{tensor_name} must be BF16")

        begin, end = (int(v) for v in selected["data_offsets"])
        if begin >= end:
            raise RuntimeError(f"invalid source range: {tensor_name}")

        shard_meta = shard_by_name[selected_shard]
        absolute_begin = 8 + int(shard_meta["header_size"]) + begin
        absolute_end = 8 + int(shard_meta["header_size"]) + end - 1
        data, remote_size = fetch_exact_range(
            shard_url(selected_shard),
            absolute_begin,
            absolute_end,
        )
        if remote_size != int(shard_meta["file_size"]):
            raise RuntimeError(
                f"remote shard size changed for {selected_shard}"
            )
        if len(data) != end - begin:
            raise RuntimeError(f"short dense range: {tensor_name}")

        safe_name = tensor_name.replace(".", "_") + ".bf16.bin"
        (output_dir / safe_name).write_bytes(data)
        items.append(
            {
                "source_tensor": tensor_name,
                "source_shape": selected["shape"],
                "source_byte_size": len(data),
                "source_file": safe_name,
                "source_fnv1a64": f"{fnv1a64(data):016x}",
            }
        )
        print(
            f"dense pilot: tensor={tensor_name} shard={selected_shard} "
            f"bytes={len(data)}"
        )

    manifest = {
        "schema_version": 1,
        "model": range_manifest["model"],
        "snapshot": range_manifest["snapshot"],
        "tensors": items,
    }
    (output_dir / "dense-pilot.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
