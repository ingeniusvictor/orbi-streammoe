#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass

from plan_qwen_missing_dense_rows import build_plan, read_journal

HEADER_RESERVE_BYTES = 16 * 1024 * 1024
DEFAULT_GROUP_SIZE = 64


@dataclass(frozen=True)
class DiskEstimate:
    output_payload_bytes: int
    output_reserve_bytes: int
    batch_source_bytes: int
    minimum_free_reserve_bytes: int

    @property
    def required_free_bytes(self) -> int:
        return (
            self.output_reserve_bytes
            + self.batch_source_bytes
            + self.minimum_free_reserve_bytes
        )


def load_json(path: pathlib.Path) -> dict:
    root = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return root


def product(values) -> int:
    result = 1
    for value in values:
        value = int(value)
        if value <= 0:
            raise RuntimeError("tensor dimensions must be positive")
        result *= value
    return result


def estimate_output_payload_bytes(
    inventory: dict,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> int:
    if group_size <= 0:
        raise RuntimeError("group_size must be positive")
    tensors = inventory.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise RuntimeError("dense stream inventory tensors must be non-empty")

    total = 0
    for item in tensors:
        shape = item.get("source_shape")
        action = item.get("action")
        if not isinstance(shape, list) or not shape:
            raise RuntimeError("dense inventory tensor missing source_shape")

        if action == "copy_bf16_to_f32":
            total += product(shape) * 4
            continue

        if action != "affine_quantize" or len(shape) != 2:
            raise RuntimeError(
                f"unsupported dense output geometry: {item.get('source_tensor')}"
            )
        rows = int(shape[0])
        cols = int(shape[1])
        if cols % 8 != 0 or cols % group_size != 0:
            raise RuntimeError(
                f"affine tensor columns are incompatible with Q4/group size: "
                f"{item.get('source_tensor')}"
            )
        packed_weight = rows * (cols // 8) * 4
        aux = rows * (cols // group_size) * 4
        total += packed_weight + aux + aux

    return total


def limit_plan_by_source_budget(plan: dict, budget_bytes: int | None) -> dict:
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("dense row plan tasks must be an array")
    if budget_bytes is None:
        return plan
    if budget_bytes <= 0:
        raise RuntimeError("max_batch_source_bytes must be positive")

    selected = []
    used = 0
    for task in tasks:
        size = int(task["source_byte_size"])
        if size <= 0:
            raise RuntimeError("dense row task source_byte_size must be positive")
        if not selected and size > budget_bytes:
            raise RuntimeError(
                "max_batch_source_bytes is smaller than one planned row chunk"
            )
        if used + size > budget_bytes:
            break
        selected.append(task)
        used += size

    limited = dict(plan)
    limited["tasks"] = selected
    limited["task_count"] = len(selected)
    limited["batch_source_bytes"] = used
    return limited


def plan_batch_key(plan: dict) -> str:
    tasks = plan.get("tasks")
    canonical = json.dumps(tasks, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def progress_summary(plan: dict) -> dict:
    progress = plan.get("tensor_progress")
    if not isinstance(progress, list):
        raise RuntimeError("dense row plan tensor_progress must be an array")
    total_rows = sum(int(item["total_rows"]) for item in progress)
    completed_rows = sum(int(item["completed_rows"]) for item in progress)
    missing_rows = sum(int(item["missing_rows"]) for item in progress)
    percent = (
        100.0
        if total_rows == 0
        else (100.0 * completed_rows / total_rows)
    )
    return {
        "tensor_count": len(progress),
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "missing_rows": missing_rows,
        "completed_percent": percent,
    }


def disk_estimate(
    inventory: dict,
    output_path: pathlib.Path,
    batch_source_bytes: int,
    minimum_free_reserve_bytes: int,
) -> DiskEstimate:
    payload = estimate_output_payload_bytes(inventory)
    output_reserve = 0
    if not output_path.exists():
        output_reserve = payload + HEADER_RESERVE_BYTES
    return DiskEstimate(
        output_payload_bytes=payload,
        output_reserve_bytes=output_reserve,
        batch_source_bytes=batch_source_bytes,
        minimum_free_reserve_bytes=minimum_free_reserve_bytes,
    )


def ensure_disk_budget(
    path: pathlib.Path,
    estimate: DiskEstimate,
) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < estimate.required_free_bytes:
        raise RuntimeError(
            "insufficient free disk for dense conversion batch: "
            f"free={free} required={estimate.required_free_bytes}"
        )
    return free


def write_json_exact(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing batch plan disagrees with controller: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def execute_controller(args) -> dict:
    inventory_path = pathlib.Path(args.inventory)
    journal_path = pathlib.Path(args.journal)
    output_path = pathlib.Path(args.output)
    work_dir = pathlib.Path(args.work_dir)
    metadata_dir = pathlib.Path(args.metadata_dir)
    converter = pathlib.Path(args.converter)
    inventory = load_json(inventory_path)

    batch_index = 0
    converted_batches = 0
    reused_batches = 0
    last_summary = None

    while True:
        journal, journal_source = read_journal(journal_path)
        plan = build_plan(
            inventory,
            journal,
            args.chunk_rows,
            args.max_chunks,
        )
        plan["journal_present"] = journal is not None
        plan["journal_source"] = (
            str(journal_source) if journal_source is not None else None
        )
        plan = limit_plan_by_source_budget(
            plan,
            args.max_batch_source_bytes,
        )
        summary = progress_summary(plan)
        last_summary = summary

        if plan["task_count"] == 0:
            return {
                "completed": True,
                "stop_reason": "complete",
                "batches_converted": converted_batches,
                "batches_reused": reused_batches,
                "progress": summary,
            }

        batch_bytes = sum(int(task["source_byte_size"]) for task in plan["tasks"])
        estimate = disk_estimate(
            inventory,
            output_path,
            batch_bytes,
            args.min_free_disk_bytes,
        )

        report = {
            "batch_index": batch_index,
            "task_count": plan["task_count"],
            "batch_source_bytes": batch_bytes,
            "progress": summary,
            "disk": {
                "output_payload_bytes": estimate.output_payload_bytes,
                "output_reserve_bytes": estimate.output_reserve_bytes,
                "minimum_free_reserve_bytes": estimate.minimum_free_reserve_bytes,
                "required_free_bytes": estimate.required_free_bytes,
            },
        }

        if args.dry_run:
            report["completed"] = False
            report["stop_reason"] = "dry_run"
            return report

        work_dir.mkdir(parents=True, exist_ok=True)
        report["free_disk_bytes"] = ensure_disk_budget(work_dir, estimate)

        key = plan_batch_key(plan)
        batch_dir = work_dir / f"batch-{key}"
        plan_path = batch_dir / "row-plan.json"
        manifest_path = batch_dir / "dense-stream.json"
        write_json_exact(plan_path, plan)

        if manifest_path.exists():
            reused_batches += 1
        else:
            if batch_dir.exists():
                for child in batch_dir.iterdir():
                    if child == plan_path:
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            run_checked([
                sys.executable,
                str(pathlib.Path(__file__).with_name("fetch_qwen_dense_row_plan.py")),
                "--plan",
                str(plan_path),
                "--output-dir",
                str(batch_dir),
            ])

        run_checked([
            str(converter),
            str(metadata_dir),
            str(manifest_path),
            str(batch_dir),
            str(output_path),
            str(journal_path),
        ])
        converted_batches += 1

        if not args.keep_batches:
            shutil.rmtree(batch_dir)

        batch_index += 1
        if args.max_batches is not None and batch_index >= args.max_batches:
            journal, journal_source = read_journal(journal_path)
            remaining = build_plan(
                inventory,
                journal,
                args.chunk_rows,
                args.max_chunks,
            )
            return {
                "completed": remaining["task_count"] == 0,
                "stop_reason": "batch_limit",
                "batches_converted": converted_batches,
                "batches_reused": reused_batches,
                "progress": progress_summary(remaining),
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--converter", required=True)
    parser.add_argument("--chunk-rows", type=int, required=True)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--max-batch-source-bytes", type=int)
    parser.add_argument("--min-free-disk-bytes", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-batches", action="store_true")
    args = parser.parse_args()

    if args.chunk_rows <= 0:
        raise RuntimeError("chunk_rows must be positive")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise RuntimeError("max_chunks must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise RuntimeError("max_batches must be positive")
    if args.min_free_disk_bytes < 0:
        raise RuntimeError("min_free_disk_bytes cannot be negative")

    result = execute_controller(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
