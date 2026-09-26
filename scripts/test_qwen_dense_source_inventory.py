#!/usr/bin/env python3
import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError(
            "usage: test_qwen_dense_source_inventory.py <dense-sources-json>"
        )

    root = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    if root.get("schema_version") != 1:
        raise RuntimeError("dense source inventory schema mismatch")
    entries = root.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("dense source inventory is empty")
    if root.get("dense_source_count") != len(entries):
        raise RuntimeError("dense source inventory count mismatch")

    names = [entry.get("source_tensor") for entry in entries]
    if names != sorted(names) or len(names) != len(set(names)):
        raise RuntimeError("dense source inventory must be unique and sorted")
    if any(".mlp.experts." in name for name in names):
        raise RuntimeError("dense source inventory leaked routed expert tensor")

    for entry in entries:
        if entry.get("action") not in ("copy_bf16_to_f32", "affine_quantize"):
            raise RuntimeError("unsupported dense source action")
        if entry.get("tensor_class") not in ("global_dense", "layer_dense"):
            raise RuntimeError("unsupported dense tensor class")
        if not entry.get("source_shard") or not entry.get("target_path"):
            raise RuntimeError("dense inventory missing shard/target")

    required = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    missing = required - set(names)
    if missing:
        raise RuntimeError(
            "official dense inventory missing: " + ", ".join(sorted(missing))
        )

    print(
        "OSM-40D official dense source inventory: PASS "
        f"tensors={len(entries)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
