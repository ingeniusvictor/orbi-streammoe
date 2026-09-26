#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import sys

from run_qwen_production_operator import PHASES, validate_manifest

SCHEMA_VERSION = 1
MAX_DEFAULT_EXPERT_BATCH = 8


def canonical_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


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


def component_path(bin_dir: pathlib.Path, declared: str) -> pathlib.Path:
    name = pathlib.Path(declared).name
    if not name:
        raise RuntimeError("manifest component name is empty")
    return bin_dir / name


def expert_source_bytes(manifest: dict) -> int:
    geometry = manifest["expert_conversion"]["geometry"]
    hidden = int(geometry["hidden_size"])
    intermediate = int(geometry["moe_intermediate_size"])
    if hidden <= 0 or intermediate <= 0:
        raise RuntimeError("invalid expert geometry")
    return 3 * hidden * intermediate * 2


def derived_expert_batch_size(manifest: dict) -> int:
    budget = int(manifest["disk"]["max_batch_source_bytes"])
    per_expert = expert_source_bytes(manifest)
    if budget <= 0 or budget < per_expert:
        raise RuntimeError(
            "manifest max_batch_source_bytes cannot fit one source expert"
        )
    return min(MAX_DEFAULT_EXPERT_BATCH, budget // per_expert)


def require_path_value(section: dict, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"manifest path value missing: {key}")
    return value


def build_phase_commands(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    bin_dir: pathlib.Path,
    scripts_dir: pathlib.Path,
    python_executable: str,
) -> dict:
    manifest = load_json(manifest_path)
    validate_manifest(manifest)

    source = manifest.get("source")
    target = manifest.get("target")
    expert = manifest.get("expert_conversion")
    dense = manifest.get("dense_conversion")
    full = manifest.get("full_checkpoint")
    disk = manifest.get("disk")
    if not all(
        isinstance(item, dict)
        for item in (source, target, expert, dense, full, disk)
    ):
        raise RuntimeError("production manifest sections are incomplete")

    metadata_dir = require_path_value(source, "metadata_dir")
    dense_inventory = require_path_value(source, "dense_inventory")
    model = require_path_value(source, "model")
    snapshot = require_path_value(source, "snapshot")
    output_dir = require_path_value(target, "output_dir")
    dense_output = require_path_value(target, "dense_output")
    dense_journal = require_path_value(target, "dense_journal")
    dense_work_dir = require_path_value(target, "dense_work_dir")

    expert_converter = component_path(bin_dir, expert["converter"])
    expert_finalizer = component_path(bin_dir, expert["finalizer"])
    dense_converter = component_path(bin_dir, dense["converter"])
    full_finalizer = component_path(bin_dir, full["finalizer"])

    expert_controller = scripts_dir / "run_qwen_expert_conversion.py"
    dense_controller = scripts_dir / pathlib.Path(dense["controller"]).name
    operator_runner = scripts_dir / "run_qwen_production_operator.py"

    batch_size = derived_expert_batch_size(manifest)

    phase_commands = {
        "expert_conversion": [
            python_executable,
            str(expert_controller),
            "--manifest",
            str(manifest_path),
            "--converter",
            str(expert_converter),
            "--max-experts-per-batch",
            str(batch_size),
        ],
        "expert_finalization": [
            str(expert_finalizer),
            metadata_dir,
            output_dir,
            model,
            snapshot,
        ],
        "dense_conversion": [
            python_executable,
            str(dense_controller),
            "--metadata-dir",
            metadata_dir,
            "--inventory",
            dense_inventory,
            "--output",
            dense_output,
            "--journal",
            dense_journal,
            "--work-dir",
            dense_work_dir,
            "--converter",
            str(dense_converter),
            "--chunk-rows",
            str(int(dense["chunk_rows"])),
            "--max-batch-source-bytes",
            str(int(dense["max_batch_source_bytes"])),
            "--min-free-disk-bytes",
            str(int(dense["min_free_disk_bytes"])),
        ],
        "full_checkpoint": [
            str(full_finalizer),
            output_dir,
            dense_journal,
        ],
    }

    max_chunks = dense.get("max_chunks")
    if max_chunks is not None:
        value = int(max_chunks)
        if value <= 0:
            raise RuntimeError("dense max_chunks must be positive")
        phase_commands["dense_conversion"].extend(
            ["--max-chunks", str(value)]
        )

    operator_commands = {}
    for phase in PHASES:
        argv = [
            python_executable,
            str(operator_runner),
            "--manifest",
            str(manifest_path),
            "--state",
            str(state_path),
            "run",
            "--phase",
            phase,
            "--",
            *phase_commands[phase],
        ]
        operator_commands[phase] = argv

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "production-operator-command-plan",
        "execution_id": manifest["execution_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "state_path": str(state_path),
        "bin_dir": str(bin_dir),
        "scripts_dir": str(scripts_dir),
        "python_executable": python_executable,
        "derived": {
            "source_bytes_per_expert": expert_source_bytes(manifest),
            "max_experts_per_batch": batch_size,
        },
        "phase_order": list(PHASES),
        "phase_commands": phase_commands,
        "operator_commands": operator_commands,
        "shell": False,
    }


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = canonical_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                f"existing command plan conflicts with requested plan: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--bin-dir", required=True)
    parser.add_argument(
        "--scripts-dir",
        default=str(pathlib.Path(__file__).resolve().parent),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = build_phase_commands(
        pathlib.Path(args.manifest),
        pathlib.Path(args.state),
        pathlib.Path(args.bin_dir),
        pathlib.Path(args.scripts_dir),
        args.python_executable,
    )
    if args.output:
        write_immutable(pathlib.Path(args.output), payload)
    print(canonical_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
