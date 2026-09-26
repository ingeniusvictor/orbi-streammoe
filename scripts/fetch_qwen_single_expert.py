#!/usr/bin/env python3
import argparse
import json
import pathlib

from fetch_qwen_conversion_slice import fnv1a64
from fetch_qwen_shard_headers import fetch_exact_range, shard_url

LAYER = 0
EXPERT = 0
PROJECTIONS = {
    "gate_proj": [512, 2048],
    "up_proj": [512, 2048],
    "down_proj": [2048, 512],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    range_manifest = json.loads(
        pathlib.Path(args.range_manifest).read_text(encoding="utf-8")
    )
    shard_by_name = {
        shard["filename"]: shard for shard in range_manifest["shards"]
    }

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    total_fetched = 0

    for projection, expected_shape in PROJECTIONS.items():
        tensor_name = (
            f"model.layers.{LAYER}.mlp.experts.{EXPERT}."
            f"{projection}.weight"
        )

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
                f"expert pilot tensor absent from range manifest: {tensor_name}"
            )
        if selected["dtype"] != "BF16":
            raise RuntimeError(f"{tensor_name} must be BF16")
        if selected["shape"] != expected_shape:
            raise RuntimeError(
                f"{tensor_name} shape changed: {selected['shape']}"
            )

        shard_meta = shard_by_name[selected_shard]
        begin, end = (int(v) for v in selected["data_offsets"])
        if begin >= end:
            raise RuntimeError(f"invalid source range: {tensor_name}")

        source_size = end - begin
        expected_size = expected_shape[0] * expected_shape[1] * 2
        if source_size != expected_size:
            raise RuntimeError(
                f"{tensor_name} BF16 bytes changed: {source_size}"
            )

        header_size = int(shard_meta["header_size"])
        absolute_begin = 8 + header_size + begin
        absolute_end = 8 + header_size + end - 1
        data, remote_size = fetch_exact_range(
            shard_url(selected_shard),
            absolute_begin,
            absolute_end,
        )
        if remote_size != int(shard_meta["file_size"]):
            raise RuntimeError(
                f"remote shard size changed for {selected_shard}"
            )
        if len(data) != source_size:
            raise RuntimeError(f"short expert range: {tensor_name}")

        filename = f"{projection}.bf16.bin"
        (output_dir / filename).write_bytes(data)
        total_fetched += len(data)
        result[projection] = {
            "tensor_name": tensor_name,
            "shard": selected_shard,
            "shape": expected_shape,
            "source_absolute_offset": absolute_begin,
            "source_byte_size": source_size,
            "source_file": filename,
            "source_fnv1a64": f"{fnv1a64(data):016x}",
        }
        print(
            f"{projection}: shard={selected_shard} "
            f"offset={absolute_begin} bytes={len(data)}"
        )

    manifest = {
        "schema_version": 1,
        "model": range_manifest["model"],
        "snapshot": range_manifest["snapshot"],
        "layer_index": LAYER,
        "expert_index": EXPERT,
        "total_fetched_bytes": total_fetched,
        "projections": result,
    }
    (output_dir / "single-expert.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"single expert pilot: layer={LAYER} expert={EXPERT} "
        f"fetched={total_fetched}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
