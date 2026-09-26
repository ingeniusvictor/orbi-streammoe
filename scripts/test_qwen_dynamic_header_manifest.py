#!/usr/bin/env python3
import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 3:
        raise RuntimeError(
            "usage: test_qwen_dynamic_header_manifest.py <manifest> <expert>"
        )
    manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    expert = int(sys.argv[2])
    required = {
        f"model.layers.0.mlp.experts.{expert}.gate_proj.weight",
        f"model.layers.0.mlp.experts.{expert}.up_proj.weight",
        f"model.layers.0.mlp.experts.{expert}.down_proj.weight",
    }
    selected = set(manifest.get("selected_tensors", []))
    missing = required - selected
    if missing:
        raise RuntimeError("dynamic header manifest missing: " + ", ".join(sorted(missing)))

    metadata = {}
    for shard in manifest.get("shards", []):
        metadata.update(shard.get("selected_tensor_metadata", {}))
    for name in required:
        item = metadata.get(name)
        if not isinstance(item, dict):
            raise RuntimeError(f"dynamic tensor metadata missing: {name}")
        if item.get("dtype") != "BF16":
            raise RuntimeError(f"dynamic tensor dtype changed: {name}")
        if not item.get("data_offsets"):
            raise RuntimeError(f"dynamic tensor range missing: {name}")

    print(f"OSM-39F dynamic expert header selection: PASS expert={expert}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
