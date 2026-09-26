#!/usr/bin/env python3
import argparse
import json
import pathlib
import tempfile
from types import SimpleNamespace
from unittest import mock

import run_qwen_dense_conversion as controller


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fixture_inventory():
    return {
        "schema_version": 1,
        "model": "fixture",
        "snapshot": "fixture",
        "tensors": [
            {
                "source_tensor": "model.norm.weight",
                "target_path": "model.norm",
                "action": "copy_bf16_to_f32",
                "source_shape": [8],
                "source_shard": "a.safetensors",
                "shard_file_size": 1000,
                "header_size": 100,
                "data_offsets": [0, 16],
            },
            {
                "source_tensor": "model.embed_tokens.weight",
                "target_path": "model.embed_tokens",
                "action": "affine_quantize",
                "source_shape": [4, 64],
                "source_shard": "b.safetensors",
                "shard_file_size": 5000,
                "header_size": 200,
                "data_offsets": [0, 512],
            },
        ],
    }


def fixture_plan(inventory):
    return {
        "schema_version": 1,
        "model": "fixture",
        "snapshot": "fixture",
        "chunk_rows": 2,
        "tensor_progress": [
            {
                "source_tensor": "model.norm.weight",
                "target_path": "model.norm",
                "action": "copy_bf16_to_f32",
                "total_rows": 8,
                "completed_rows": 0,
                "missing_rows": 8,
            },
            {
                "source_tensor": "model.embed_tokens.weight",
                "target_path": "model.embed_tokens",
                "action": "affine_quantize",
                "total_rows": 4,
                "completed_rows": 0,
                "missing_rows": 4,
            },
        ],
        "tasks": [
            {
                "source_tensor": "model.norm.weight",
                "first_row": 0,
                "row_count": 2,
                "source_byte_size": 4,
            },
            {
                "source_tensor": "model.norm.weight",
                "first_row": 2,
                "row_count": 2,
                "source_byte_size": 4,
            },
            {
                "source_tensor": "model.embed_tokens.weight",
                "first_row": 0,
                "row_count": 2,
                "source_byte_size": 256,
            },
        ],
        "task_count": 3,
        "inventory": inventory["tensors"],
    }


def run_tests():
    inventory = fixture_inventory()
    payload = controller.estimate_output_payload_bytes(inventory)
    require(payload == 192, f"unexpected output payload estimate: {payload}")

    plan = fixture_plan(inventory)
    limited = controller.limit_plan_by_source_budget(plan, 8)
    require(limited["task_count"] == 2, "budget should keep first two tasks")
    require(limited["batch_source_bytes"] == 8, "budget byte accounting mismatch")

    too_small = False
    try:
        controller.limit_plan_by_source_budget(plan, 3)
    except RuntimeError:
        too_small = True
    require(too_small, "budget smaller than one chunk must fail")

    key1 = controller.plan_batch_key(plan)
    key2 = controller.plan_batch_key(json.loads(json.dumps(plan)))
    require(key1 == key2 and len(key1) == 16, "batch key must be deterministic")

    summary = controller.progress_summary(plan)
    require(summary["total_rows"] == 12, "progress total rows mismatch")
    require(summary["missing_rows"] == 12, "progress missing rows mismatch")

    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        output = root / "model.safetensors"
        estimate = controller.disk_estimate(inventory, output, 8, 100)
        require(
            estimate.output_reserve_bytes == payload + controller.HEADER_RESERVE_BYTES,
            "fresh output must reserve complete payload plus header",
        )
        output.write_bytes(b"x")
        existing = controller.disk_estimate(inventory, output, 8, 100)
        require(existing.output_reserve_bytes == 0, "existing output must not re-reserve")

        plan_path = root / "batch" / "row-plan.json"
        controller.write_json_exact(plan_path, plan)
        controller.write_json_exact(plan_path, plan)
        disagree = dict(plan)
        disagree["task_count"] = 99
        rejected = False
        try:
            controller.write_json_exact(plan_path, disagree)
        except RuntimeError:
            rejected = True
        require(rejected, "existing different batch plan must be rejected")

    # Dry-run must not execute network or converter subprocesses.
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        inventory_path = root / "inventory.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        args = SimpleNamespace(
            inventory=str(inventory_path),
            journal=str(root / "journal.json"),
            output=str(root / "model.safetensors"),
            work_dir=str(root / "work"),
            metadata_dir=str(root / "metadata"),
            converter=str(root / "converter"),
            chunk_rows=2,
            max_chunks=2,
            max_batch_source_bytes=8,
            min_free_disk_bytes=0,
            max_batches=None,
            dry_run=True,
            keep_batches=False,
        )

        with mock.patch.object(controller, "build_plan", return_value=plan), \
             mock.patch.object(controller, "read_journal", return_value=(None, None)), \
             mock.patch.object(controller, "run_checked") as run_checked:
            result = controller.execute_controller(args)
            require(result["stop_reason"] == "dry_run", "dry-run stop reason mismatch")
            require(result["task_count"] == 2, "dry-run budgeted task count mismatch")
            run_checked.assert_not_called()
            require(not (root / "work").exists(), "dry-run must not create work directory")

    print("OSM-40E dense production controller: PASS")
    print("  output_size_estimate=PASS")
    print("  source_budget=PASS")
    print("  deterministic_batch_key=PASS")
    print("  progress_reporting=PASS")
    print("  batch_plan_immutability=PASS")
    print("  dry_run_no_side_effects=PASS")


if __name__ == "__main__":
    run_tests()
