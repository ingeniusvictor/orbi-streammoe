#!/usr/bin/env python3
import argparse
import json
import pathlib


def resolve_journal(path: pathlib.Path) -> pathlib.Path | None:
    if path.exists():
        return path
    backup = pathlib.Path(str(path) + ".bak")
    if backup.exists():
        return backup
    return None


def read_journal(path: pathlib.Path) -> tuple[dict | None, pathlib.Path | None]:
    source = resolve_journal(path)
    if source is None:
        return None, None
    root = json.loads(source.read_text(encoding="utf-8"))
    if root.get("schema_version") != 1:
        raise RuntimeError("unsupported streamed safetensors journal schema")
    tensors = root.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise RuntimeError("journal tensors must be a non-empty array")
    by_name = {}
    for tensor in tensors:
        name = tensor.get("name")
        rows = tensor.get("completed_rows")
        shape = tensor.get("shape")
        if (
            not isinstance(name, str)
            or not name
            or name in by_name
            or not isinstance(rows, int)
            or rows < 0
            or not isinstance(shape, list)
            or not shape
        ):
            raise RuntimeError("malformed streamed safetensors journal tensor")
        if rows > int(shape[0]):
            raise RuntimeError("journal completed_rows exceeds tensor shape")
        by_name[name] = tensor
    return by_name, source


def source_completed_rows(item: dict, journal: dict | None) -> int:
    if journal is None:
        return 0

    action = item["action"]
    target = item["target_path"]
    source_rows = int(item["source_shape"][0])

    if action == "copy_bf16_to_f32":
        names = [target]
    elif action == "affine_quantize":
        names = [
            target + ".weight",
            target + ".scales",
            target + ".biases",
        ]
    else:
        raise RuntimeError(f"unsupported dense action: {action}")

    states = []
    for name in names:
        state = journal.get(name)
        if state is None:
            raise RuntimeError(
                f"journal missing preplanned dense target tensor: {name}"
            )
        shape = state.get("shape")
        if not isinstance(shape, list) or not shape or int(shape[0]) != source_rows:
            raise RuntimeError(f"journal/source row geometry drift: {name}")
        states.append(int(state["completed_rows"]))

    if len(set(states)) != 1:
        raise RuntimeError(
            f"affine target progress diverged for source {item['source_tensor']}"
        )
    return states[0]


def build_plan(
    inventory: dict,
    journal: dict | None,
    chunk_rows: int,
    max_chunks: int | None,
) -> dict:
    if inventory.get("schema_version") != 1:
        raise RuntimeError("unsupported dense stream inventory schema")
    tensors = inventory.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise RuntimeError("dense stream inventory tensors must be non-empty")
    if chunk_rows <= 0:
        raise RuntimeError("chunk_rows must be positive")
    if max_chunks is not None and max_chunks <= 0:
        raise RuntimeError("max_chunks must be positive")

    tasks = []
    tensor_progress = []

    for item in tensors:
        shape = item.get("source_shape")
        offsets = item.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not shape
            or any(int(v) <= 0 for v in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise RuntimeError("malformed dense inventory tensor")

        total_rows = int(shape[0])
        completed = source_completed_rows(item, journal)
        if completed > total_rows:
            raise RuntimeError("completed rows exceed source rows")

        row_elements = 1
        for dim in shape[1:]:
            row_elements *= int(dim)
        row_bytes = row_elements * 2
        data_begin = int(offsets[0])
        data_end = int(offsets[1])
        expected_bytes = total_rows * row_bytes
        if data_end - data_begin != expected_bytes:
            raise RuntimeError(
                f"source BF16 byte geometry drift: {item['source_tensor']}"
            )

        tensor_progress.append(
            {
                "source_tensor": item["source_tensor"],
                "target_path": item["target_path"],
                "action": item["action"],
                "total_rows": total_rows,
                "completed_rows": completed,
                "missing_rows": total_rows - completed,
            }
        )

        row = completed
        while row < total_rows:
            if max_chunks is not None and len(tasks) >= max_chunks:
                break
            count = min(chunk_rows, total_rows - row)
            relative_begin = data_begin + row * row_bytes
            byte_count = count * row_bytes
            absolute_begin = 8 + int(item["header_size"]) + relative_begin
            absolute_end = absolute_begin + byte_count - 1
            tasks.append(
                {
                    "source_tensor": item["source_tensor"],
                    "target_path": item["target_path"],
                    "action": item["action"],
                    "source_shape": shape,
                    "source_shard": item["source_shard"],
                    "shard_file_size": int(item["shard_file_size"]),
                    "first_row": row,
                    "row_count": count,
                    "row_elements": row_elements,
                    "source_byte_size": byte_count,
                    "source_absolute_begin": absolute_begin,
                    "source_absolute_end": absolute_end,
                }
            )
            row += count
        if max_chunks is not None and len(tasks) >= max_chunks:
            break

    return {
        "schema_version": 1,
        "model": inventory["model"],
        "snapshot": inventory["snapshot"],
        "chunk_rows": chunk_rows,
        "dense_tensor_count": len(tensors),
        "tensor_progress": tensor_progress,
        "task_count": len(tasks),
        "tasks": tasks,
        "inventory": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--chunk-rows", type=int, required=True)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    inventory = json.loads(
        pathlib.Path(args.inventory).read_text(encoding="utf-8")
    )
    journal, journal_source = read_journal(pathlib.Path(args.journal))
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

    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
