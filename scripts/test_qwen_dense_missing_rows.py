#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys
import tempfile


def run(script, *args):
    process = subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(process.stdout)


def journal_tensor(name, rows, total_rows):
    return {
        "name": name,
        "dtype": "F32",
        "shape": [total_rows, 1],
        "payload_offset": 0,
        "row_bytes": 4,
        "completed_rows": rows,
        "chunks": [],
    }


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    script = root / "scripts" / "plan_qwen_missing_dense_rows.py"

    inventory = {
        "schema_version": 1,
        "model": "fixture",
        "snapshot": "fixture",
        "dense_tensor_count": 2,
        "tensors": [
            {
                "source_tensor": "model.embed_tokens.weight",
                "target_path": "model.embed_tokens",
                "action": "affine_quantize",
                "source_shape": [10, 8],
                "source_shard": "a.safetensors",
                "shard_file_size": 10000,
                "header_size": 100,
                "data_offsets": [1000, 1160],
            },
            {
                "source_tensor": "model.norm.weight",
                "target_path": "model.norm.weight",
                "action": "copy_bf16_to_f32",
                "source_shape": [6],
                "source_shard": "b.safetensors",
                "shard_file_size": 5000,
                "header_size": 80,
                "data_offsets": [2000, 2012],
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        inventory_path = tmp / "inventory.json"
        journal_path = tmp / "model.progress.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

        fresh = run(
            script,
            "--inventory", str(inventory_path),
            "--journal", str(journal_path),
            "--chunk-rows", "4",
        )
        assert fresh["journal_present"] is False
        assert fresh["task_count"] == 5
        assert fresh["tasks"][0]["source_tensor"] == "model.embed_tokens.weight"
        assert fresh["tasks"][0]["first_row"] == 0
        assert fresh["tasks"][0]["row_count"] == 4
        assert fresh["tasks"][0]["source_byte_size"] == 64
        assert fresh["tasks"][0]["source_absolute_begin"] == 1108
        assert fresh["tasks"][2]["first_row"] == 8
        assert fresh["tasks"][2]["row_count"] == 2
        assert fresh["tasks"][3]["source_tensor"] == "model.norm.weight"

        limited = run(
            script,
            "--inventory", str(inventory_path),
            "--journal", str(journal_path),
            "--chunk-rows", "4",
            "--max-chunks", "1",
        )
        assert limited["task_count"] == 1

        journal = {
            "schema_version": 1,
            "data_start": 128,
            "payload_bytes": 1024,
            "tensors": [
                journal_tensor("model.embed_tokens.weight", 4, 10),
                journal_tensor("model.embed_tokens.scales", 4, 10),
                journal_tensor("model.embed_tokens.biases", 4, 10),
                journal_tensor("model.norm.weight", 6, 6),
            ],
        }
        journal_path.write_text(json.dumps(journal), encoding="utf-8")

        resumed = run(
            script,
            "--inventory", str(inventory_path),
            "--journal", str(journal_path),
            "--chunk-rows", "4",
        )
        assert resumed["journal_present"] is True
        assert resumed["task_count"] == 2
        assert [task["first_row"] for task in resumed["tasks"]] == [4, 8]
        assert resumed["tensor_progress"][0]["completed_rows"] == 4
        assert resumed["tensor_progress"][1]["missing_rows"] == 0

        backup = pathlib.Path(str(journal_path) + ".bak")
        journal_path.rename(backup)
        recovered = run(
            script,
            "--inventory", str(inventory_path),
            "--journal", str(journal_path),
            "--chunk-rows", "4",
        )
        assert recovered["journal_source"] == str(backup)
        assert recovered["task_count"] == 2

        divergent = json.loads(backup.read_text(encoding="utf-8"))
        divergent["tensors"][1]["completed_rows"] = 5
        backup.write_text(json.dumps(divergent), encoding="utf-8")
        failed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--inventory", str(inventory_path),
                "--journal", str(journal_path),
                "--chunk-rows", "4",
            ],
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        assert "progress diverged" in failed.stderr

    print("OSM-40D missing dense row planner: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
