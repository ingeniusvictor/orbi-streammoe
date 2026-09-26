#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib

from fetch_qwen_checkpoint_metadata import MODEL, SNAPSHOT
from launch_qwen_production import preview, prepare_manifest


SCHEMA_VERSION = 1
DEFAULT_GROUP_SIZE = 64
DEFAULT_CHUNK_ROWS = 16
DEFAULT_MAX_BATCH_SOURCE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_EVIDENCE_BYTES = 512 * 1024 * 1024


def canonical_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = canonical_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing rehearsal evidence conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def collect_bounded_tree(root: pathlib.Path, max_total_bytes: int) -> dict:
    if max_total_bytes <= 0:
        raise RuntimeError("max evidence bytes must be positive")
    if not root.is_dir():
        raise RuntimeError(f"bounded source evidence directory is missing: {root}")

    files = []
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total += size
        if total > max_total_bytes:
            raise RuntimeError(
                f"bounded evidence exceeded {max_total_bytes} bytes: {root}"
            )
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError(f"bounded source evidence is empty: {root}")
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": total,
        "aggregate_sha256": aggregate,
        "files": files,
    }


def build_rehearsal(
    metadata_dir: pathlib.Path,
    dense_inventory_path: pathlib.Path,
    expert_slice_dir: pathlib.Path,
    dense_slice_dir: pathlib.Path,
    output_dir: pathlib.Path,
    bin_dir: pathlib.Path,
    scripts_dir: pathlib.Path,
    python_executable: str,
    group_size: int = DEFAULT_GROUP_SIZE,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    max_batch_source_bytes: int = DEFAULT_MAX_BATCH_SOURCE_BYTES,
    max_evidence_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
) -> dict:
    config = load_json(metadata_dir / "config.json")
    inventory = load_json(dense_inventory_path)

    if inventory.get("model") != MODEL or inventory.get("snapshot") != SNAPSHOT:
        raise RuntimeError("official dense inventory model/snapshot drift")
    if config.get("model_type") not in {"qwen3_next", "qwen3_moe"}:
        raise RuntimeError("unexpected official Qwen model_type")

    layers = int(config.get("num_hidden_layers", 0))
    experts = int(config.get("num_experts", 0))
    if layers <= 0 or experts <= 0:
        raise RuntimeError("official config is missing layer/expert geometry")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "execution.json"
    state = output_dir / "operator-state.json"
    command_plan = output_dir / "operator-state.json.commands.json"
    receipts = pathlib.Path(str(state) + ".receipts")
    preflight = output_dir / "preflight-request.json"

    request = {
        "metadata_dir": str(metadata_dir),
        "dense_inventory": str(dense_inventory_path),
        "model": MODEL,
        "snapshot": SNAPSHOT,
        "output_dir": str(output_dir / "checkpoint"),
        "expert_work_dir": str(output_dir / "expert-work"),
        "dense_work_dir": str(output_dir / "dense-work"),
        "group_size": group_size,
        "layer_first": 0,
        "layer_end": layers,
        "expert_first": 0,
        "expert_end": experts,
        "chunk_rows": chunk_rows,
        "max_batch_source_bytes": max_batch_source_bytes,
        "min_free_disk_bytes": 0,
    }
    write_immutable(preflight, request)
    prepare_manifest(manifest, preflight)

    dry_run = preview(
        manifest,
        state,
        bin_dir,
        scripts_dir,
        python_executable,
        command_plan,
    )

    if state.exists() or receipts.exists():
        raise RuntimeError("OSM-42A dry-run mutated operator state or receipts")
    if dry_run.get("state_initialized"):
        raise RuntimeError("OSM-42A preview unexpectedly initialized state")
    if dry_run.get("next_phase") != "expert_conversion":
        raise RuntimeError("OSM-42A preview selected the wrong first phase")
    if dry_run.get("mutated_state"):
        raise RuntimeError("OSM-42A preview reported state mutation")

    expert_evidence = collect_bounded_tree(
        expert_slice_dir, max_evidence_bytes
    )
    dense_evidence = collect_bounded_tree(
        dense_slice_dir, max_evidence_bytes
    )
    execution = load_json(manifest)
    command_payload = load_json(command_plan)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "stage": "official-production-rehearsal",
        "source": {
            "model": MODEL,
            "snapshot": SNAPSHOT,
            "metadata_dir": str(metadata_dir),
            "config_sha256": sha256_file(metadata_dir / "config.json"),
            "dense_inventory": str(dense_inventory_path),
            "dense_inventory_sha256": sha256_file(dense_inventory_path),
        },
        "geometry": {
            "layer_count": layers,
            "expert_count": experts,
            "hidden_size": int(config.get("hidden_size", 0)),
            "moe_intermediate_size": int(config.get("moe_intermediate_size", 0)),
        },
        "production_authorization": {
            "execution_id": execution["execution_id"],
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "command_plan": str(command_plan),
            "command_plan_sha256": sha256_file(command_plan),
            "authorized": bool(execution["authorization"]["authorized"]),
            "full_source_scope": bool(
                execution["authorization"]["full_source_scope"]
            ),
            "phase_order": list(command_payload["phase_order"]),
        },
        "dry_run": dry_run,
        "bounded_official_source_evidence": {
            "expert_slice": expert_evidence,
            "dense_slice": dense_evidence,
        },
        "safety": {
            "operator_state_created": state.exists(),
            "receipt_dir_created": receipts.exists(),
            "full_checkpoint_conversion_executed": False,
            "bounded_source_evidence_only": True,
        },
    }
    bundle["bundle_sha256"] = hashlib.sha256(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    bundle_path = output_dir / "rehearsal-bundle.json"
    write_immutable(bundle_path, bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--dense-inventory", required=True)
    parser.add_argument("--expert-slice-dir", required=True)
    parser.add_argument("--dense-slice-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bin-dir", required=True)
    parser.add_argument(
        "--scripts-dir",
        default=str(pathlib.Path(__file__).resolve().parent),
    )
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        "--max-batch-source-bytes",
        type=int,
        default=DEFAULT_MAX_BATCH_SOURCE_BYTES,
    )
    parser.add_argument(
        "--max-evidence-bytes",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_BYTES,
    )
    args = parser.parse_args()

    result = build_rehearsal(
        pathlib.Path(args.metadata_dir),
        pathlib.Path(args.dense_inventory),
        pathlib.Path(args.expert_slice_dir),
        pathlib.Path(args.dense_slice_dir),
        pathlib.Path(args.output_dir),
        pathlib.Path(args.bin_dir),
        pathlib.Path(args.scripts_dir),
        args.python_executable,
        args.group_size,
        args.chunk_rows,
        args.max_batch_source_bytes,
        args.max_evidence_bytes,
    )
    print(canonical_text(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
