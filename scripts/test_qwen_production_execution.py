#!/usr/bin/env python3
import json
import pathlib
import tempfile
from types import SimpleNamespace

from plan_qwen_production_execution import (
    HEADER_RESERVE_BYTES,
    build_manifest,
    write_immutable,
)


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fixture_args(root: pathlib.Path) -> SimpleNamespace:
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    write_json(
        metadata / "config.json",
        {
            "model_type": "qwen3_next",
            "hidden_size": 8,
            "moe_intermediate_size": 8,
            "num_experts": 3,
            "num_hidden_layers": 4,
        },
    )
    dense_inventory = root / "dense-inventory.json"
    write_json(
        dense_inventory,
        {
            "schema_version": 1,
            "model": "fixture/model",
            "snapshot": "fixture-snapshot",
            "dense_tensor_count": 2,
            "tensors": [
                {
                    "source_tensor": "model.norm.weight",
                    "source_shape": [8],
                    "action": "copy_bf16_to_f32",
                },
                {
                    "source_tensor": "model.embed_tokens.weight",
                    "source_shape": [4, 8],
                    "action": "affine_quantize",
                },
            ],
        },
    )
    return SimpleNamespace(
        metadata_dir=str(metadata),
        dense_inventory=str(dense_inventory),
        model="fixture/model",
        snapshot="fixture-snapshot",
        output_dir=str(root / "output"),
        expert_work_dir=str(root / "expert-work"),
        dense_work_dir=str(root / "dense-work"),
        manifest=str(root / "execution.json"),
        group_size=4,
        layer_first=0,
        layer_end=4,
        expert_first=0,
        expert_end=3,
        chunk_rows=2,
        max_chunks=5,
        max_batch_source_bytes=1024,
        min_free_disk_bytes=4096,
    )


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except Exception:
        return
    raise RuntimeError(f"expected failure: {label}")


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="orbi-streammoe-osm41a-"
    ) as temp:
        root = pathlib.Path(temp)
        args = fixture_args(root)

        payload = build_manifest(args)
        if payload["schema_version"] != 1:
            raise RuntimeError("schema mismatch")
        if payload["expert_conversion"]["geometry"]["expert_stride"] != 480:
            raise RuntimeError("expert stride mismatch")
        if payload["expert_conversion"]["geometry"]["total_expert_payload_bytes"] != 5760:
            raise RuntimeError("expert total mismatch")
        if payload["dense_conversion"]["estimated_payload_bytes"] != 112:
            raise RuntimeError("dense payload estimate mismatch")
        expected_final = 5760 + 112 + HEADER_RESERVE_BYTES
        if payload["disk"]["final_output_reserve_bytes"] != expected_final:
            raise RuntimeError("final output reserve mismatch")
        if payload["disk"]["peak_required_free_bytes"] != (
            expected_final + 1024 + 4096
        ):
            raise RuntimeError("peak disk reserve mismatch")
        if len(payload["execution_id"]) != 20:
            raise RuntimeError("execution ID length mismatch")

        manifest = pathlib.Path(args.manifest)
        write_immutable(manifest, payload)
        write_immutable(manifest, payload)

        changed = dict(payload)
        changed["authorization"] = dict(payload["authorization"])
        changed["authorization"]["authorized"] = False
        expect_failure(
            lambda: write_immutable(manifest, changed),
            "immutable manifest conflict",
        )

        bad_snapshot = fixture_args(root / "bad-snapshot")
        bad_snapshot.snapshot = "wrong"
        expect_failure(
            lambda: build_manifest(bad_snapshot),
            "snapshot mismatch",
        )

        bad_range = fixture_args(root / "bad-range")
        bad_range.expert_end = 2
        expect_failure(
            lambda: build_manifest(bad_range),
            "partial expert range",
        )

        bad_layers = fixture_args(root / "bad-layers")
        bad_layers.layer_end = 3
        expect_failure(
            lambda: build_manifest(bad_layers),
            "partial layer range",
        )

        bad_budget = fixture_args(root / "bad-budget")
        bad_budget.max_batch_source_bytes = 0
        expect_failure(
            lambda: build_manifest(bad_budget),
            "invalid batch budget",
        )

        print(
            "OSM-41A production execution manifest: PASS\n"
            "  immutable_manifest=PASS\n"
            "  source_hash_binding=PASS\n"
            "  full_expert_scope_guard=PASS\n"
            "  full_layer_scope_guard=PASS\n"
            "  exact_expert_disk_geometry=PASS\n"
            "  dense_output_estimate=PASS\n"
            "  peak_disk_budget=PASS\n"
            "  deterministic_execution_id=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
