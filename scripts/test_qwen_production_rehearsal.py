#!/usr/bin/env python3
import json
import pathlib
import tempfile

import rehearse_qwen_production as rehearsal


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="orbi-streammoe-osm42a-"
    ) as temp:
        root = pathlib.Path(temp)
        metadata = root / "metadata"
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
        write_json(metadata / "model.safetensors.index.json", {"weight_map": {}})

        dense = root / "dense-inventory.json"
        write_json(
            dense,
            {
                "schema_version": 1,
                "model": rehearsal.MODEL,
                "snapshot": rehearsal.SNAPSHOT,
                "dense_tensor_count": 2,
                "tensors": [
                    {
                        "source_tensor": "model.embed_tokens.weight",
                        "source_shape": [4, 8],
                        "action": "affine_quantize",
                    },
                    {
                        "source_tensor": "model.norm.weight",
                        "source_shape": [8],
                        "action": "copy_bf16_to_f32",
                    },
                ],
            },
        )

        expert_slice = root / "expert-slice"
        dense_slice = root / "dense-slice"
        expert_slice.mkdir()
        dense_slice.mkdir()

        expert_dir = expert_slice / "expert_000"
        expert_dir.mkdir()
        expert_payloads = {
            "gate_proj.bf16.bin": b"gate-source",
            "up_proj.bf16.bin": b"up-source",
            "down_proj.bf16.bin": b"down-source",
        }
        for name, payload in expert_payloads.items():
            (expert_dir / name).write_bytes(payload)
        write_json(
            expert_dir / "single-expert.json",
            {
                "schema_version": 1,
                "model": rehearsal.MODEL,
                "snapshot": rehearsal.SNAPSHOT,
                "layer_index": 0,
                "expert_index": 0,
                "total_fetched_bytes": sum(
                    len(payload) for payload in expert_payloads.values()
                ),
                "projections": {
                    "gate_proj": {"source_file": "gate_proj.bf16.bin"},
                    "up_proj": {"source_file": "up_proj.bf16.bin"},
                    "down_proj": {"source_file": "down_proj.bf16.bin"},
                },
            },
        )
        write_json(
            expert_slice / "expert-range.json",
            {
                "schema_version": 1,
                "layer_index": 0,
                "first_expert": 0,
                "count": 1,
                "experts": [0],
                "total_fetched_bytes": sum(
                    len(payload) for payload in expert_payloads.values()
                ),
            },
        )

        dense_payloads = {
            "model_norm_weight.bf16.bin": b"norm-source",
            "model_layers_0_mlp_shared_expert_gate_weight.bf16.bin": (
                b"shared-gate-source"
            ),
        }
        for name, payload in dense_payloads.items():
            (dense_slice / name).write_bytes(payload)
        write_json(
            dense_slice / "dense-pilot.json",
            {
                "schema_version": 1,
                "model": rehearsal.MODEL,
                "snapshot": rehearsal.SNAPSHOT,
                "tensors": [
                    {
                        "source_tensor": "model.norm.weight",
                        "source_byte_size": len(
                            dense_payloads["model_norm_weight.bf16.bin"]
                        ),
                        "source_file": "model_norm_weight.bf16.bin",
                    },
                    {
                        "source_tensor": (
                            "model.layers.0.mlp.shared_expert_gate.weight"
                        ),
                        "source_byte_size": len(
                            dense_payloads[
                                "model_layers_0_mlp_shared_expert_gate_weight.bf16.bin"
                            ]
                        ),
                        "source_file": (
                            "model_layers_0_mlp_shared_expert_gate_weight.bf16.bin"
                        ),
                    },
                ],
            },
        )

        out = root / "rehearsal"
        result = rehearsal.build_rehearsal(
            metadata,
            dense,
            expert_slice,
            dense_slice,
            out,
            root / "bin",
            pathlib.Path(__file__).resolve().parent,
            "python-fixture",
            group_size=4,
            chunk_rows=2,
            max_batch_source_bytes=4096,
            max_evidence_bytes=4096,
        )

        if result["stage"] != "official-production-rehearsal":
            raise RuntimeError("rehearsal stage mismatch")
        if result["production_authorization"]["phase_order"] != [
            "expert_conversion",
            "expert_finalization",
            "dense_conversion",
            "full_checkpoint",
        ]:
            raise RuntimeError("phase order mismatch")
        if result["dry_run"]["next_phase"] != "expert_conversion":
            raise RuntimeError("dry-run did not select expert conversion")
        if result["safety"]["operator_state_created"]:
            raise RuntimeError("dry-run created operator state")
        if result["safety"]["receipt_dir_created"]:
            raise RuntimeError("dry-run created receipts")
        if not (out / "execution.json").exists():
            raise RuntimeError("immutable execution manifest missing")
        if not (out / "operator-state.json.commands.json").exists():
            raise RuntimeError("immutable command plan missing")
        if not (out / "rehearsal-bundle.json").exists():
            raise RuntimeError("rehearsal bundle missing")

        first = (out / "rehearsal-bundle.json").read_bytes()
        rerun = rehearsal.build_rehearsal(
            metadata,
            dense,
            expert_slice,
            dense_slice,
            out,
            root / "bin",
            pathlib.Path(__file__).resolve().parent,
            "python-fixture",
            group_size=4,
            chunk_rows=2,
            max_batch_source_bytes=4096,
            max_evidence_bytes=4096,
        )
        if rerun["bundle_sha256"] != result["bundle_sha256"]:
            raise RuntimeError("rehearsal is not reproducible")
        if (out / "rehearsal-bundle.json").read_bytes() != first:
            raise RuntimeError("rehearsal bundle changed on idempotent rerun")

        expect_failure(
            lambda: rehearsal.collect_bounded_tree(expert_slice, 1),
            "bounded evidence budget",
        )

        drift = root / "dense-drift.json"
        payload = json.loads(dense.read_text(encoding="utf-8"))
        payload["snapshot"] = "wrong-snapshot"
        write_json(drift, payload)
        expect_failure(
            lambda: rehearsal.build_rehearsal(
                metadata,
                drift,
                expert_slice,
                dense_slice,
                root / "bad-rehearsal",
                root / "bin",
                pathlib.Path(__file__).resolve().parent,
                "python-fixture",
                group_size=4,
                chunk_rows=2,
                max_batch_source_bytes=4096,
                max_evidence_bytes=4096,
            ),
            "official source pin drift",
        )

        print(
            "OSM-42A official production rehearsal: PASS\n"
            "  official_source_pin=PASS\n"
            "  immutable_full_scope_manifest=PASS\n"
            "  exact_command_plan=PASS\n"
            "  read_only_launcher_preview=PASS\n"
            "  bounded_source_evidence=PASS\n"
            "  reproducible_bundle=PASS\n"
            "  source_drift_rejection=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
