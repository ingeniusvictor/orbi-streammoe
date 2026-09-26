#!/usr/bin/env python3
import argparse
import json
import pathlib


def locate_tensor(range_manifest: dict, tensor_name: str) -> tuple[dict, dict]:
    found = []
    for shard in range_manifest.get("shards", []):
        metadata = shard.get("selected_tensor_metadata", {})
        if tensor_name in metadata:
            found.append((shard, metadata[tensor_name]))
    if len(found) != 1:
        raise RuntimeError(
            f"expected exactly one header entry for {tensor_name}, got {len(found)}"
        )
    return found[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-sources-json", required=True)
    parser.add_argument("--range-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sources = json.loads(
        pathlib.Path(args.dense_sources_json).read_text(encoding="utf-8")
    )
    ranges = json.loads(
        pathlib.Path(args.range_manifest).read_text(encoding="utf-8")
    )

    if sources.get("schema_version") != 1:
        raise RuntimeError("unsupported dense source inventory schema")
    if ranges.get("schema_version") != 1:
        raise RuntimeError("unsupported range manifest schema")

    entries = sources.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("dense source inventory entries must be non-empty")

    tensors = []
    seen = set()
    for entry in entries:
        source_tensor = entry.get("source_tensor")
        source_shard = entry.get("source_shard")
        target_path = entry.get("target_path")
        action = entry.get("action")
        if (
            not isinstance(source_tensor, str)
            or not source_tensor
            or not isinstance(source_shard, str)
            or not source_shard
            or not isinstance(target_path, str)
            or not target_path
            or action not in ("copy_bf16_to_f32", "affine_quantize")
        ):
            raise RuntimeError("malformed dense source inventory entry")
        if source_tensor in seen:
            raise RuntimeError(f"duplicate dense source tensor: {source_tensor}")
        seen.add(source_tensor)

        shard, meta = locate_tensor(ranges, source_tensor)
        if shard.get("filename") != source_shard:
            raise RuntimeError(
                f"source shard drift for {source_tensor}: "
                f"{shard.get('filename')} != {source_shard}"
            )
        if meta.get("dtype") != "BF16":
            raise RuntimeError(f"{source_tensor} must be BF16")

        shape = [int(v) for v in meta.get("shape", [])]
        offsets = [int(v) for v in meta.get("data_offsets", [])]
        if not shape or any(v <= 0 for v in shape):
            raise RuntimeError(f"invalid source shape: {source_tensor}")
        if len(offsets) != 2 or offsets[0] < 0 or offsets[0] >= offsets[1]:
            raise RuntimeError(f"invalid data offsets: {source_tensor}")

        elements = 1
        for dim in shape:
            elements *= dim
        if offsets[1] - offsets[0] != elements * 2:
            raise RuntimeError(f"BF16 byte geometry drift: {source_tensor}")

        tensors.append(
            {
                "source_tensor": source_tensor,
                "target_path": target_path,
                "action": action,
                "tensor_class": entry.get("tensor_class"),
                "layer_index": entry.get("layer_index"),
                "source_shape": shape,
                "source_shard": source_shard,
                "shard_file_size": int(shard["file_size"]),
                "header_size": int(shard["header_size"]),
                "data_offsets": offsets,
            }
        )

    tensors.sort(key=lambda item: item["source_tensor"])
    payload = {
        "schema_version": 1,
        "model": ranges["model"],
        "snapshot": ranges["snapshot"],
        "source_checkpoint": sources.get("source_checkpoint"),
        "dense_tensor_count": len(tensors),
        "tensors": tensors,
    }

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"dense stream inventory: tensors={len(tensors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
