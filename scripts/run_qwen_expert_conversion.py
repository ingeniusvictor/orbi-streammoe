#!/usr/bin/env python3
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass

from plan_qwen_missing_experts import parse_journal

SOURCE_BYTES_PER_VALUE = 2


@dataclass(frozen=True)
class ExpertBatch:
    layer: int
    experts: tuple[int, ...]

    @property
    def first(self) -> int:
        return self.experts[0]

    @property
    def end_exclusive(self) -> int:
        return self.experts[-1] + 1

    @property
    def csv(self) -> str:
        return ",".join(str(v) for v in self.experts)


def load_json(path: pathlib.Path) -> dict:
    root = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return root


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported production manifest schema")
    if manifest.get("stage") != "production-execution-preflight":
        raise RuntimeError("manifest stage mismatch")
    auth = manifest.get("authorization")
    if not isinstance(auth, dict) or auth.get("authorized") is not True:
        raise RuntimeError("production manifest is not authorized")

    expert = manifest.get("expert_conversion")
    source = manifest.get("source")
    target = manifest.get("target")
    disk = manifest.get("disk")
    if not all(isinstance(item, dict) for item in (expert, source, target, disk)):
        raise RuntimeError("production manifest sections are missing")

    geometry = expert.get("geometry")
    if not isinstance(geometry, dict):
        raise RuntimeError("expert geometry is missing")

    for key in (
        "hidden_size",
        "moe_intermediate_size",
        "expert_count",
        "layer_count",
        "expert_stride",
        "layer_bytes",
    ):
        if int(geometry.get(key, 0)) <= 0:
            raise RuntimeError(f"invalid expert geometry field: {key}")

    if int(expert.get("layer_first", -1)) != 0:
        raise RuntimeError("production expert controller requires layer_first=0")
    if int(expert.get("layer_end_exclusive", -1)) != int(geometry["layer_count"]):
        raise RuntimeError("production expert controller requires all layers")
    if int(expert.get("expert_first", -1)) != 0:
        raise RuntimeError("production expert controller requires expert_first=0")
    if int(expert.get("expert_end_exclusive", -1)) != int(geometry["expert_count"]):
        raise RuntimeError("production expert controller requires all experts")

    for section, key in (
        (source, "metadata_dir"),
        (target, "output_dir"),
        (target, "expert_work_dir"),
        (expert, "converter"),
    ):
        value = section.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"manifest value missing: {key}")


def expert_source_bytes(geometry: dict) -> int:
    hidden = int(geometry["hidden_size"])
    intermediate = int(geometry["moe_intermediate_size"])
    return 3 * hidden * intermediate * SOURCE_BYTES_PER_VALUE


def layer_name(layer: int) -> str:
    return f"layer_{layer:02d}"


def layer_paths(output_dir: pathlib.Path, layer: int) -> tuple[pathlib.Path, pathlib.Path]:
    packed = output_dir / "packed_experts"
    stem = layer_name(layer)
    return packed / f"{stem}.bin", packed / f"{stem}.progress.json"


def completed_experts(
    journal_path: pathlib.Path,
    layer: int,
    expert_count: int,
) -> set[int]:
    root, _ = parse_journal(journal_path)
    if not root:
        return set()
    if int(root.get("layer_index", -1)) != layer:
        raise RuntimeError("expert journal layer mismatch")
    if int(root.get("expert_count", -1)) != expert_count:
        raise RuntimeError("expert journal expert_count mismatch")
    completed = {int(item["expert"]) for item in root["completed"]}
    if any(value < 0 or value >= expert_count for value in completed):
        raise RuntimeError("expert journal contains out-of-range expert")
    return completed


def split_contiguous_batches(
    missing: list[int],
    max_experts_per_batch: int,
    layer: int,
) -> list[ExpertBatch]:
    if max_experts_per_batch <= 0:
        raise RuntimeError("max_experts_per_batch must be positive")
    if not missing:
        return []

    ordered = sorted(missing)
    if len(set(ordered)) != len(ordered) or ordered[0] < 0:
        raise RuntimeError("missing expert list is invalid")

    runs: list[list[int]] = []
    current = [ordered[0]]
    for expert in ordered[1:]:
        if expert == current[-1] + 1:
            current.append(expert)
        else:
            runs.append(current)
            current = [expert]
    runs.append(current)

    batches: list[ExpertBatch] = []
    for run in runs:
        for start in range(0, len(run), max_experts_per_batch):
            batches.append(
                ExpertBatch(
                    layer=layer,
                    experts=tuple(run[start : start + max_experts_per_batch]),
                )
            )
    return batches


def build_execution_plan(
    manifest: dict,
    max_experts_per_batch: int,
    max_layers: int | None = None,
) -> dict:
    validate_manifest(manifest)
    geometry = manifest["expert_conversion"]["geometry"]
    expert_count = int(geometry["expert_count"])
    layer_count = int(geometry["layer_count"])
    output_dir = pathlib.Path(manifest["target"]["output_dir"])

    layers = []
    total_missing = 0
    total_complete = 0
    total_batches = 0

    for layer in range(layer_count):
        if max_layers is not None and len(layers) >= max_layers:
            break
        _, journal = layer_paths(output_dir, layer)
        complete = completed_experts(journal, layer, expert_count)
        missing = [v for v in range(expert_count) if v not in complete]
        batches = split_contiguous_batches(
            missing,
            max_experts_per_batch,
            layer,
        )
        layers.append({
            "layer": layer,
            "completed_experts": len(complete),
            "missing_experts": missing,
            "batch_count": len(batches),
            "batches": [list(item.experts) for item in batches],
        })
        total_missing += len(missing)
        total_complete += len(complete)
        total_batches += len(batches)

    return {
        "layer_count_total": layer_count,
        "layers_considered": len(layers),
        "expert_count_per_layer": expert_count,
        "completed_experts": total_complete,
        "missing_experts": total_missing,
        "batch_count": total_batches,
        "source_bytes_per_expert": expert_source_bytes(geometry),
        "layers": layers,
    }


def existing_probe(path: pathlib.Path) -> pathlib.Path:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def required_batch_free_bytes(
    manifest: dict,
    layer_file: pathlib.Path,
    batch: ExpertBatch,
) -> int:
    geometry = manifest["expert_conversion"]["geometry"]
    layer_reserve = 0 if layer_file.exists() else int(geometry["layer_bytes"])
    source_reserve = expert_source_bytes(geometry) * len(batch.experts)
    minimum = int(manifest["disk"].get("minimum_free_reserve_bytes", 0))
    if minimum < 0:
        raise RuntimeError("minimum free reserve cannot be negative")
    return layer_reserve + source_reserve + minimum


def ensure_disk_budget(path: pathlib.Path, required: int) -> int:
    free = shutil.disk_usage(existing_probe(path)).free
    if free < required:
        raise RuntimeError(
            f"insufficient disk for expert batch: free={free} required={required}"
        )
    return free


def batch_dir_name(batch: ExpertBatch) -> str:
    return (
        f"layer-{batch.layer:02d}-experts-"
        f"{batch.first:03d}-{batch.end_exclusive - 1:03d}"
    )


def batch_inputs_ready(batch_dir: pathlib.Path, batch: ExpertBatch) -> bool:
    summary_path = batch_dir / "expert-inputs" / "expert-range.json"
    if not summary_path.exists():
        return False
    try:
        summary = load_json(summary_path)
    except Exception:
        return False
    if int(summary.get("layer_index", -1)) != batch.layer:
        return False
    if [int(v) for v in summary.get("experts", [])] != list(batch.experts):
        return False
    for expert in batch.experts:
        path = (
            batch_dir
            / "expert-inputs"
            / f"expert_{expert:03d}"
            / "single-expert.json"
        )
        if not path.exists():
            return False
    return True


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def converter_matches_manifest(converter: pathlib.Path, manifest: dict) -> None:
    expected = pathlib.Path(manifest["expert_conversion"]["converter"]).name
    if converter.name != expected:
        raise RuntimeError(
            f"converter disagrees with production manifest: "
            f"expected={expected} actual={converter.name}"
        )


def execute_controller(args, runner=run_checked) -> dict:
    manifest = load_json(pathlib.Path(args.manifest))
    validate_manifest(manifest)

    metadata_dir = pathlib.Path(manifest["source"]["metadata_dir"])
    output_dir = pathlib.Path(manifest["target"]["output_dir"])
    work_dir = pathlib.Path(manifest["target"]["expert_work_dir"])
    converter = pathlib.Path(args.converter)
    converter_matches_manifest(converter, manifest)

    geometry = manifest["expert_conversion"]["geometry"]
    expert_count = int(geometry["expert_count"])
    layer_count = int(geometry["layer_count"])
    hidden = int(geometry["hidden_size"])
    intermediate = int(geometry["moe_intermediate_size"])

    plan = build_execution_plan(
        manifest,
        args.max_experts_per_batch,
        args.max_layers,
    )
    if args.dry_run:
        return {
            "completed": (
                plan["missing_experts"] == 0
                and plan["layers_considered"] == layer_count
            ),
            "stop_reason": "dry_run",
            "plan": plan,
        }

    scripts_dir = pathlib.Path(__file__).resolve().parent
    batches_run = 0
    batches_reused = 0
    source_experts_planned = 0
    layers_touched: set[int] = set()

    work_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "packed_experts").mkdir(parents=True, exist_ok=True)

    stop_reason = "complete"
    for layer_entry in plan["layers"]:
        layer = int(layer_entry["layer"])
        for values in layer_entry["batches"]:
            if args.max_batches is not None and batches_run >= args.max_batches:
                stop_reason = "batch_limit"
                break

            batch = ExpertBatch(
                layer=layer,
                experts=tuple(int(v) for v in values),
            )
            layer_file, journal_file = layer_paths(output_dir, layer)
            required = required_batch_free_bytes(manifest, layer_file, batch)
            ensure_disk_budget(output_dir, required)

            batch_dir = work_dir / batch_dir_name(batch)
            expert_root = batch_dir / "expert-inputs"
            range_manifest = batch_dir / "range-manifest.json"
            scratch = batch_dir / "scratch"

            if batch_inputs_ready(batch_dir, batch):
                batches_reused += 1
            else:
                if batch_dir.exists():
                    shutil.rmtree(batch_dir)
                batch_dir.mkdir(parents=True, exist_ok=True)

                runner([
                    sys.executable,
                    str(scripts_dir / "fetch_qwen_shard_headers.py"),
                    "--metadata-dir",
                    str(metadata_dir),
                    "--output",
                    str(range_manifest),
                    "--layer",
                    str(layer),
                    "--experts",
                    batch.csv,
                ])
                runner([
                    sys.executable,
                    str(scripts_dir / "fetch_qwen_expert_range.py"),
                    "--range-manifest",
                    str(range_manifest),
                    "--output-dir",
                    str(expert_root),
                    "--layer",
                    str(layer),
                    "--experts",
                    batch.csv,
                    "--hidden-size",
                    str(hidden),
                    "--intermediate-size",
                    str(intermediate),
                ])

            runner([
                str(converter),
                str(metadata_dir),
                str(expert_root),
                str(layer_file),
                str(journal_file),
                str(layer),
                str(batch.first),
                str(batch.end_exclusive),
                str(scratch),
            ])

            now_complete = completed_experts(
                journal_file,
                layer,
                expert_count,
            )
            missing_commit = [
                v for v in batch.experts if v not in now_complete
            ]
            if missing_commit:
                raise RuntimeError(
                    "converter returned success without journaling experts: "
                    + ",".join(str(v) for v in missing_commit)
                )

            batches_run += 1
            source_experts_planned += len(batch.experts)
            layers_touched.add(layer)

            if not args.keep_batches:
                shutil.rmtree(batch_dir)

        if stop_reason == "batch_limit":
            break

    final_plan = build_execution_plan(
        manifest,
        args.max_experts_per_batch,
        None,
    )
    completed = final_plan["missing_experts"] == 0
    if completed:
        stop_reason = "complete"
    elif stop_reason == "complete":
        stop_reason = "incomplete_scope"

    return {
        "completed": completed,
        "stop_reason": stop_reason,
        "batches_run": batches_run,
        "batches_reused": batches_reused,
        "layers_touched": sorted(layers_touched),
        "experts_processed": source_experts_planned,
        "remaining_missing_experts": final_plan["missing_experts"],
        "final_plan": final_plan,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--converter", required=True)
    parser.add_argument("--max-experts-per-batch", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-batches", action="store_true")
    args = parser.parse_args()

    if args.max_experts_per_batch <= 0:
        raise RuntimeError("max_experts_per_batch must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise RuntimeError("max_batches must be positive")
    if args.max_layers is not None and args.max_layers <= 0:
        raise RuntimeError("max_layers must be positive")

    result = execute_controller(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
