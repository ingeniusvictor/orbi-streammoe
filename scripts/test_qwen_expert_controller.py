#!/usr/bin/env python3
import json
import pathlib
import tempfile
from types import SimpleNamespace

from run_qwen_expert_conversion import (
    ExpertBatch,
    batch_inputs_ready,
    build_execution_plan,
    completed_experts,
    execute_controller,
    expert_source_bytes,
    layer_paths,
    required_batch_free_bytes,
    split_contiguous_batches,
)


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fixture(root: pathlib.Path) -> tuple[pathlib.Path, dict]:
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    write_json(
        metadata / "config.json",
        {
            "model_type": "qwen3_next",
            "hidden_size": 8,
            "moe_intermediate_size": 8,
            "num_experts": 5,
            "num_hidden_layers": 2,
        },
    )
    output = root / "output"
    work = root / "work"
    manifest = {
        "schema_version": 1,
        "stage": "production-execution-preflight",
        "execution_id": "0123456789abcdefghij",
        "source": {"metadata_dir": str(metadata)},
        "target": {
            "output_dir": str(output),
            "expert_work_dir": str(work),
        },
        "expert_conversion": {
            "layer_first": 0,
            "layer_end_exclusive": 2,
            "expert_first": 0,
            "expert_end_exclusive": 5,
            "converter": "orbi_streammoe_qpack_convert_range",
            "geometry": {
                "hidden_size": 8,
                "moe_intermediate_size": 8,
                "expert_count": 5,
                "layer_count": 2,
                "expert_stride": 480,
                "layer_bytes": 2400,
            },
        },
        "disk": {"minimum_free_reserve_bytes": 0},
        "authorization": {
            "authorized": True,
            "immutable_preflight": True,
            "full_source_scope": True,
        },
    }
    manifest_path = root / "execution.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def write_journal(path: pathlib.Path, layer: int, completed: list[int]) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "layer_index": layer,
            "expert_count": 5,
            "expert_stride": 480,
            "completed": [
                {"expert": value, "fnv1a64": f"{value + 1:016x}"}
                for value in completed
            ],
        },
    )


def fake_batch_inputs(batch_dir: pathlib.Path, batch: ExpertBatch) -> None:
    root = batch_dir / "expert-inputs"
    write_json(
        root / "expert-range.json",
        {
            "schema_version": 1,
            "layer_index": batch.layer,
            "experts": list(batch.experts),
        },
    )
    for expert in batch.experts:
        write_json(
            root / f"expert_{expert:03d}" / "single-expert.json",
            {"schema_version": 1},
        )


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="orbi-streammoe-osm41c-"
    ) as temp:
        root = pathlib.Path(temp)
        manifest_path, manifest = fixture(root)
        output = pathlib.Path(manifest["target"]["output_dir"])

        _, journal0 = layer_paths(output, 0)
        write_journal(journal0, 0, [0, 2])

        if completed_experts(journal0, 0, 5) != {0, 2}:
            raise RuntimeError("completed expert read mismatch")

        batches = split_contiguous_batches([1, 3, 4], 2, 0)
        if [list(v.experts) for v in batches] != [[1], [3, 4]]:
            raise RuntimeError("contiguous batch split mismatch")

        plan = build_execution_plan(manifest, 2)
        if plan["missing_experts"] != 8:
            raise RuntimeError("missing expert plan mismatch")
        if plan["completed_experts"] != 2:
            raise RuntimeError("completed expert plan mismatch")
        if plan["batch_count"] != 5:
            raise RuntimeError("batch count mismatch")
        if plan["source_bytes_per_expert"] != 384:
            raise RuntimeError("source bytes per expert mismatch")
        if expert_source_bytes(
            manifest["expert_conversion"]["geometry"]
        ) != 384:
            raise RuntimeError("source geometry accounting mismatch")

        layer_file0, _ = layer_paths(output, 0)
        required = required_batch_free_bytes(
            manifest,
            layer_file0,
            ExpertBatch(0, (3, 4)),
        )
        if required != 2400 + 2 * 384:
            raise RuntimeError("first-layer disk reserve mismatch")
        layer_file0.parent.mkdir(parents=True, exist_ok=True)
        layer_file0.write_bytes(b"x")
        required_existing = required_batch_free_bytes(
            manifest,
            layer_file0,
            ExpertBatch(0, (3, 4)),
        )
        if required_existing != 2 * 384:
            raise RuntimeError("existing-layer disk reserve mismatch")
        layer_file0.unlink()

        ready_dir = root / "ready-batch"
        ready_batch = ExpertBatch(0, (3, 4))
        fake_batch_inputs(ready_dir, ready_batch)
        if not batch_inputs_ready(ready_dir, ready_batch):
            raise RuntimeError("ready batch was not recognized")

        args = SimpleNamespace(
            manifest=str(manifest_path),
            converter=str(root / "orbi_streammoe_qpack_convert_range"),
            max_experts_per_batch=2,
            max_batches=None,
            max_layers=None,
            dry_run=True,
            keep_batches=False,
        )
        dry = execute_controller(args)
        if dry["stop_reason"] != "dry_run":
            raise RuntimeError("dry-run stop reason mismatch")
        if dry["plan"]["missing_experts"] != 8:
            raise RuntimeError("dry-run plan mismatch")

        calls = []
        args.dry_run = False
        args.max_batches = 1

        def fake_runner(command: list[str]) -> None:
            calls.append(command)
            joined = " ".join(command)
            if "fetch_qwen_expert_range.py" in joined:
                layer = int(command[command.index("--layer") + 1])
                experts = tuple(
                    int(v)
                    for v in command[
                        command.index("--experts") + 1
                    ].split(",")
                )
                output_dir = pathlib.Path(
                    command[command.index("--output-dir") + 1]
                )
                fake_batch_inputs(
                    output_dir.parent,
                    ExpertBatch(layer, experts),
                )
            elif command[0].endswith(
                "orbi_streammoe_qpack_convert_range"
            ):
                layer = int(command[5])
                first = int(command[6])
                end = int(command[7])
                journal = pathlib.Path(command[4])
                before = completed_experts(journal, layer, 5)
                write_journal(
                    journal,
                    layer,
                    sorted(before | set(range(first, end))),
                )

        result = execute_controller(args, runner=fake_runner)
        if result["stop_reason"] != "batch_limit":
            raise RuntimeError("batch limit stop mismatch")
        if result["batches_run"] != 1:
            raise RuntimeError("bounded batch count mismatch")
        if result["experts_processed"] != 1:
            raise RuntimeError("bounded expert accounting mismatch")
        if result["remaining_missing_experts"] != 7:
            raise RuntimeError("post-batch remaining count mismatch")
        if len(calls) != 3:
            raise RuntimeError(
                "expected header, fetch and converter commands"
            )

        if completed_experts(journal0, 0, 5) != {0, 1, 2}:
            raise RuntimeError(
                "fake converter journal advancement mismatch"
            )

        print(
            "OSM-41C production expert controller: PASS\n"
            "  full_layer_plan=PASS\n"
            "  journal_first_planning=PASS\n"
            "  contiguous_missing_batches=PASS\n"
            "  bounded_source_accounting=PASS\n"
            "  disk_budget=PASS\n"
            "  reusable_batch_detection=PASS\n"
            "  dry_run_no_network=PASS\n"
            "  bounded_execution=PASS\n"
            "  post_converter_journal_gate=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
