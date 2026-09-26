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
    return process


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    script = root / "scripts" / "build_qwen_dense_stream_inventory.py"

    sources = {
        "schema_version": 1,
        "source_checkpoint": "fixture",
        "entries": [
            {
                "source_tensor": "model.embed_tokens.weight",
                "source_shard": "a.safetensors",
                "tensor_class": "global_dense",
                "action": "affine_quantize",
                "target_path": "model.embed_tokens",
                "layer_index": None,
            },
            {
                "source_tensor": "model.norm.weight",
                "source_shard": "b.safetensors",
                "tensor_class": "global_dense",
                "action": "copy_bf16_to_f32",
                "target_path": "model.norm.weight",
                "layer_index": None,
            },
        ],
    }
    ranges = {
        "schema_version": 1,
        "model": "fixture",
        "snapshot": "fixture",
        "shards": [
            {
                "filename": "a.safetensors",
                "file_size": 10000,
                "header_size": 100,
                "selected_tensor_metadata": {
                    "model.embed_tokens.weight": {
                        "dtype": "BF16",
                        "shape": [10, 8],
                        "data_offsets": [1000, 1160],
                    }
                },
            },
            {
                "filename": "b.safetensors",
                "file_size": 5000,
                "header_size": 80,
                "selected_tensor_metadata": {
                    "model.norm.weight": {
                        "dtype": "BF16",
                        "shape": [6],
                        "data_offsets": [2000, 2012],
                    }
                },
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        sources_path = tmp / "sources.json"
        ranges_path = tmp / "ranges.json"
        output = tmp / "inventory.json"
        sources_path.write_text(json.dumps(sources), encoding="utf-8")
        ranges_path.write_text(json.dumps(ranges), encoding="utf-8")

        run(
            script,
            "--dense-sources-json", str(sources_path),
            "--range-manifest", str(ranges_path),
            "--output", str(output),
        )
        inventory = json.loads(output.read_text(encoding="utf-8"))
        assert inventory["dense_tensor_count"] == 2
        assert inventory["tensors"][0]["source_tensor"] == "model.embed_tokens.weight"
        assert inventory["tensors"][0]["source_shape"] == [10, 8]
        assert inventory["tensors"][0]["header_size"] == 100
        assert inventory["tensors"][1]["action"] == "copy_bf16_to_f32"

        broken = json.loads(ranges_path.read_text(encoding="utf-8"))
        broken["shards"][0]["filename"] = "wrong.safetensors"
        ranges_path.write_text(json.dumps(broken), encoding="utf-8")
        failed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dense-sources-json", str(sources_path),
                "--range-manifest", str(ranges_path),
                "--output", str(output),
            ],
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0

    print("OSM-40D dense stream inventory builder: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
