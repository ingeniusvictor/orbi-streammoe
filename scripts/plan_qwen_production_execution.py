#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib

from run_qwen_dense_conversion import (
    HEADER_RESERVE_BYTES,
    estimate_output_payload_bytes,
)

SCHEMA_VERSION = 1
DEFAULT_GROUP_SIZE = 64


def load_json(path: pathlib.Path) -> dict:
    root = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return root


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_positive_int(root: dict, key: str) -> int:
    value = int(root.get(key, 0))
    if value <= 0:
        raise RuntimeError(f"config field must be positive: {key}")
    return value


def affine_q4_matrix_bytes(rows: int, cols: int, group_size: int) -> int:
    if rows <= 0 or cols <= 0 or group_size <= 0:
        raise RuntimeError("invalid affine Q4 geometry")
    if cols % 8 != 0 or cols % group_size != 0:
        raise RuntimeError("affine Q4 columns are not pack/group aligned")
    packed = rows * (cols // 8) * 4
    aux = rows * (cols // group_size) * 4
    return packed + aux + aux


def expert_geometry(config: dict, group_size: int) -> dict:
    hidden = checked_positive_int(config, "hidden_size")
    intermediate = checked_positive_int(config, "moe_intermediate_size")
    experts = checked_positive_int(config, "num_experts")
    layers = checked_positive_int(config, "num_hidden_layers")

    stride = (
        affine_q4_matrix_bytes(intermediate, hidden, group_size)
        + affine_q4_matrix_bytes(intermediate, hidden, group_size)
        + affine_q4_matrix_bytes(hidden, intermediate, group_size)
    )
    layer_bytes = stride * experts
    total = layer_bytes * layers
    return {
        "hidden_size": hidden,
        "moe_intermediate_size": intermediate,
        "expert_count": experts,
        "layer_count": layers,
        "expert_stride": stride,
        "layer_bytes": layer_bytes,
        "total_expert_payload_bytes": total,
    }


def canonical_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = canonical_text(payload)
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current != rendered:
            raise RuntimeError(
                f"existing execution manifest conflicts with requested preflight: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def build_manifest(args) -> dict:
    metadata_dir = pathlib.Path(args.metadata_dir)
    config_path = metadata_dir / "config.json"
    dense_inventory_path = pathlib.Path(args.dense_inventory)

    config = load_json(config_path)
    dense_inventory = load_json(dense_inventory_path)

    if dense_inventory.get("schema_version") != 1:
        raise RuntimeError("unsupported dense inventory schema")
    if dense_inventory.get("snapshot") != args.snapshot:
        raise RuntimeError("dense inventory snapshot disagrees with requested snapshot")
    if dense_inventory.get("model") != args.model:
        raise RuntimeError("dense inventory model disagrees with requested model")

    geometry = expert_geometry(config, args.group_size)
    dense_payload = estimate_output_payload_bytes(
        dense_inventory,
        args.group_size,
    )

    if args.expert_first != 0 or args.expert_end != geometry["expert_count"]:
        raise RuntimeError(
            "OSM-41A full production manifest requires the complete expert range"
        )
    if args.layer_first != 0 or args.layer_end != geometry["layer_count"]:
        raise RuntimeError(
            "OSM-41A full production manifest requires the complete layer range"
        )
    if args.chunk_rows <= 0:
        raise RuntimeError("chunk_rows must be positive")
    if args.max_batch_source_bytes <= 0:
        raise RuntimeError("max_batch_source_bytes must be positive")
    if args.min_free_disk_bytes < 0:
        raise RuntimeError("min_free_disk_bytes cannot be negative")

    final_output_reserve = (
        geometry["total_expert_payload_bytes"]
        + dense_payload
        + HEADER_RESERVE_BYTES
    )
    peak_required_free = (
        final_output_reserve
        + args.max_batch_source_bytes
        + args.min_free_disk_bytes
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "production-execution-preflight",
        "source": {
            "model": args.model,
            "snapshot": args.snapshot,
            "metadata_dir": str(pathlib.Path(args.metadata_dir)),
            "config_sha256": sha256_file(config_path),
            "dense_inventory": str(dense_inventory_path),
            "dense_inventory_sha256": sha256_file(dense_inventory_path),
        },
        "target": {
            "output_dir": str(pathlib.Path(args.output_dir)),
            "expert_work_dir": str(pathlib.Path(args.expert_work_dir)),
            "dense_work_dir": str(pathlib.Path(args.dense_work_dir)),
            "dense_output": str(pathlib.Path(args.output_dir) / "model.safetensors"),
            "dense_journal": str(pathlib.Path(args.output_dir) / "model.progress.json"),
        },
        "quantization": {
            "mode": "affine",
            "bits": 4,
            "group_size": args.group_size,
        },
        "expert_conversion": {
            "layer_first": args.layer_first,
            "layer_end_exclusive": args.layer_end,
            "expert_first": args.expert_first,
            "expert_end_exclusive": args.expert_end,
            "geometry": geometry,
            "planner": "scripts/plan_qwen_missing_experts.py",
            "fetcher": "scripts/fetch_qwen_expert_range.py",
            "converter": "orbi_streammoe_qpack_convert_range",
            "finalizer": "orbi_streammoe_qpack_finalize_experts",
        },
        "dense_conversion": {
            "tensor_count": int(dense_inventory["dense_tensor_count"]),
            "estimated_payload_bytes": dense_payload,
            "header_reserve_bytes": HEADER_RESERVE_BYTES,
            "chunk_rows": args.chunk_rows,
            "max_chunks": args.max_chunks,
            "max_batch_source_bytes": args.max_batch_source_bytes,
            "min_free_disk_bytes": args.min_free_disk_bytes,
            "controller": "scripts/run_qwen_dense_conversion.py",
            "converter": "orbi_streammoe_streamed_dense_convert",
        },
        "full_checkpoint": {
            "finalizer": "orbi_streammoe_finalize_full_checkpoint",
            "required_package_stage": "full-checkpoint",
        },
        "disk": {
            "expert_payload_bytes": geometry["total_expert_payload_bytes"],
            "dense_payload_bytes": dense_payload,
            "dense_header_reserve_bytes": HEADER_RESERVE_BYTES,
            "final_output_reserve_bytes": final_output_reserve,
            "max_batch_source_bytes": args.max_batch_source_bytes,
            "minimum_free_reserve_bytes": args.min_free_disk_bytes,
            "peak_required_free_bytes": peak_required_free,
        },
        "authorization": {
            "full_source_scope": True,
            "immutable_preflight": True,
            "authorized": True,
        },
    }

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["execution_id"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:20]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--dense-inventory", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expert-work-dir", required=True)
    parser.add_argument("--dense-work-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--layer-first", type=int, default=0)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--expert-first", type=int, default=0)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--chunk-rows", type=int, required=True)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--max-batch-source-bytes", type=int, required=True)
    parser.add_argument("--min-free-disk-bytes", type=int, default=0)
    args = parser.parse_args()

    if args.group_size <= 0:
        raise RuntimeError("group_size must be positive")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise RuntimeError("max_chunks must be positive")

    payload = build_manifest(args)
    write_immutable(pathlib.Path(args.manifest), payload)
    print(canonical_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
