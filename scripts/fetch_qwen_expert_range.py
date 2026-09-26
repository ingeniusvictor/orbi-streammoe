#!/usr/bin/env python3
import argparse
import json
import pathlib

from fetch_qwen_conversion_slice import fnv1a64
from fetch_qwen_shard_headers import fetch_exact_range, shard_url

PROJECTIONS = {
    "gate_proj": lambda hidden, inter: [inter, hidden],
    "up_proj": lambda hidden, inter: [inter, hidden],
    "down_proj": lambda hidden, inter: [hidden, inter],
}


def fetch_expert(range_manifest, output_root, layer, expert, hidden, intermediate):
    shard_by_name = {
        shard["filename"]: shard for shard in range_manifest["shards"]
    }
    output_dir = output_root / f"expert_{expert:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    total_fetched = 0

    for projection, shape_fn in PROJECTIONS.items():
        expected_shape = shape_fn(hidden, intermediate)
        tensor_name = (
            f"model.layers.{layer}.mlp.experts.{expert}."
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
            raise RuntimeError(f"tensor absent from range manifest: {tensor_name}")
        if selected["dtype"] != "BF16":
            raise RuntimeError(f"{tensor_name} must be BF16")
        if selected["shape"] != expected_shape:
            raise RuntimeError(
                f"{tensor_name} shape changed: {selected['shape']}"
            )

        shard_meta = shard_by_name[selected_shard]
        begin, end = (int(v) for v in selected["data_offsets"])
        source_size = end - begin
        expected_size = expected_shape[0] * expected_shape[1] * 2
        if begin >= end or source_size != expected_size:
            raise RuntimeError(f"invalid BF16 range: {tensor_name}")

        absolute_begin = 8 + int(shard_meta["header_size"]) + begin
        absolute_end = 8 + int(shard_meta["header_size"]) + end - 1
        data, remote_size = fetch_exact_range(
            shard_url(selected_shard),
            absolute_begin,
            absolute_end,
        )
        if remote_size != int(shard_meta["file_size"]):
            raise RuntimeError(f"remote shard size changed: {selected_shard}")
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

    manifest = {
        "schema_version": 1,
        "model": range_manifest["model"],
        "snapshot": range_manifest["snapshot"],
        "layer_index": layer,
        "expert_index": expert,
        "total_fetched_bytes": total_fetched,
        "projections": result,
    }
    (output_dir / "single-expert.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"expert range pilot: layer={layer} expert={expert} "
        f"fetched={total_fetched}"
    )
    return total_fetched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--first-expert", type=int, default=0)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    args = parser.parse_args()

    if args.layer < 0 or args.first_expert < 0 or args.count <= 0:
        raise RuntimeError("layer/first-expert/count must define a positive range")

    range_manifest = json.loads(
        pathlib.Path(args.range_manifest).read_text(encoding="utf-8")
    )
    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    experts = []
    total = 0
    for expert in range(args.first_expert, args.first_expert + args.count):
        fetched = fetch_expert(
            range_manifest,
            output_root,
            args.layer,
            expert,
            args.hidden_size,
            args.intermediate_size,
        )
        experts.append(expert)
        total += fetched

    summary = {
        "schema_version": 1,
        "layer_index": args.layer,
        "first_expert": args.first_expert,
        "count": args.count,
        "experts": experts,
        "total_fetched_bytes": total,
    }
    (output_root / "expert-range.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"bounded expert range: layer={args.layer} "
        f"experts={experts} total_fetched={total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
