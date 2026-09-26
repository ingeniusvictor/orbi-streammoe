#!/usr/bin/env python3
import argparse
import json
import pathlib

from fetch_qwen_conversion_slice import fnv1a64
from fetch_qwen_shard_headers import fetch_exact_range, shard_url


def safe_name(tensor: str, first_row: int, row_count: int) -> str:
    stem = tensor.replace(".", "_")
    return f"{stem}.rows{first_row}-{first_row + row_count - 1}.bf16.bin"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    plan = json.loads(pathlib.Path(args.plan).read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise RuntimeError("unsupported dense row plan schema")

    inventory = plan.get("inventory")
    tasks = plan.get("tasks")
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("dense row plan inventory must be non-empty")
    if not isinstance(tasks, list):
        raise RuntimeError("dense row plan tasks must be an array")

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    chunks_by_tensor = {item["source_tensor"]: [] for item in inventory}
    total = 0

    for task in tasks:
        tensor = task["source_tensor"]
        if tensor not in chunks_by_tensor:
            raise RuntimeError(f"task references unknown tensor: {tensor}")

        begin = int(task["source_absolute_begin"])
        end = int(task["source_absolute_end"])
        expected = int(task["source_byte_size"])
        data, remote_size = fetch_exact_range(
            shard_url(task["source_shard"]),
            begin,
            end,
        )
        if remote_size != int(task["shard_file_size"]):
            raise RuntimeError(
                f"remote shard size changed for {task['source_shard']}"
            )
        if len(data) != expected:
            raise RuntimeError(f"short dense row fetch: {tensor}")

        filename = safe_name(
            tensor,
            int(task["first_row"]),
            int(task["row_count"]),
        )
        (output / filename).write_bytes(data)
        total += len(data)
        chunks_by_tensor[tensor].append(
            {
                "first_row": int(task["first_row"]),
                "row_count": int(task["row_count"]),
                "source_file": filename,
                "source_byte_size": len(data),
                "source_fnv1a64": f"{fnv1a64(data):016x}",
            }
        )
        print(
            f"dense rows: tensor={tensor} "
            f"rows={task['first_row']}+{task['row_count']} "
            f"bytes={len(data)}"
        )

    tensors = []
    for item in inventory:
        name = item["source_tensor"]
        tensors.append(
            {
                "source_tensor": name,
                "source_shape": item["source_shape"],
                "chunks": chunks_by_tensor[name],
            }
        )

    manifest = {
        "schema_version": 1,
        "model": plan["model"],
        "snapshot": plan["snapshot"],
        "total_fetched_bytes": total,
        "task_count": len(tasks),
        "tensors": tensors,
    }
    manifest_path = output / "dense-stream.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"dense row batch: tasks={len(tasks)} fetched={total} "
        f"manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
