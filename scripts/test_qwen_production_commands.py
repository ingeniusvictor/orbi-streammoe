#!/usr/bin/env python3
import json
import pathlib
import tempfile

from build_qwen_production_commands import (
    build_phase_commands,
    write_immutable,
)


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def manifest_fixture(root: pathlib.Path) -> pathlib.Path:
    path = root / "execution manifest.json"
    payload = {
        "schema_version": 1,
        "stage": "production-execution-preflight",
        "execution_id": "0123456789abcdefghij",
        "authorization": {
            "authorized": True,
            "immutable_preflight": True,
            "full_source_scope": True,
        },
        "source": {
            "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "snapshot": "fixture-snapshot",
            "metadata_dir": str(root / "metadata dir"),
            "dense_inventory": str(root / "dense inventory.json"),
        },
        "target": {
            "output_dir": str(root / "output dir"),
            "expert_work_dir": str(root / "expert work"),
            "dense_work_dir": str(root / "dense work"),
            "dense_output": str(root / "output dir" / "model.safetensors"),
            "dense_journal": str(root / "output dir" / "model.progress.json"),
        },
        "quantization": {
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
        },
        "expert_conversion": {
            "layer_first": 0,
            "layer_end_exclusive": 48,
            "expert_first": 0,
            "expert_end_exclusive": 512,
            "geometry": {
                "hidden_size": 2048,
                "moe_intermediate_size": 512,
                "expert_count": 512,
                "layer_count": 48,
                "expert_stride": 1966080,
                "layer_bytes": 1006632960,
            },
            "converter": "orbi_streammoe_qpack_convert_range",
            "finalizer": "orbi_streammoe_qpack_finalize_experts",
        },
        "dense_conversion": {
            "controller": "scripts/run_qwen_dense_conversion.py",
            "converter": "orbi_streammoe_streamed_dense_convert",
            "chunk_rows": 128,
            "max_chunks": 4,
            "max_batch_source_bytes": 67108864,
            "min_free_disk_bytes": 1073741824,
        },
        "full_checkpoint": {
            "finalizer": "orbi_streammoe_finalize_full_checkpoint",
            "required_package_stage": "full-checkpoint",
        },
        "disk": {
            "max_batch_source_bytes": 67108864,
            "minimum_free_reserve_bytes": 1073741824,
        },
    }
    write_json(path, payload)
    return path


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except Exception:
        return
    raise RuntimeError(f"expected failure: {label}")


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="orbi-streammoe-osm41d-"
    ) as temp:
        root = pathlib.Path(temp)
        manifest = manifest_fixture(root)
        state = root / "operator state.json"
        bin_dir = root / "bin dir"
        scripts_dir = root / "scripts dir"

        plan = build_phase_commands(
            manifest,
            state,
            bin_dir,
            scripts_dir,
            "python-fixture",
        )

        if plan["phase_order"] != [
            "expert_conversion",
            "expert_finalization",
            "dense_conversion",
            "full_checkpoint",
        ]:
            raise RuntimeError("phase order mismatch")
        if plan["shell"] is not False:
            raise RuntimeError("command plan must forbid shell execution")
        if plan["derived"]["source_bytes_per_expert"] != 6291456:
            raise RuntimeError("expert source byte derivation mismatch")
        if plan["derived"]["max_experts_per_batch"] != 8:
            raise RuntimeError("expert batch derivation mismatch")

        expert = plan["phase_commands"]["expert_conversion"]
        if expert[:2] != [
            "python-fixture",
            str(scripts_dir / "run_qwen_expert_conversion.py"),
        ]:
            raise RuntimeError("expert controller command mismatch")
        if str(bin_dir / "orbi_streammoe_qpack_convert_range") not in expert:
            raise RuntimeError("expert converter binding missing")

        expert_final = plan["phase_commands"]["expert_finalization"]
        if expert_final != [
            str(bin_dir / "orbi_streammoe_qpack_finalize_experts"),
            str(root / "metadata dir"),
            str(root / "output dir"),
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "fixture-snapshot",
        ]:
            raise RuntimeError("expert finalizer command mismatch")

        dense = plan["phase_commands"]["dense_conversion"]
        for expected in (
            str(root / "dense inventory.json"),
            str(root / "output dir" / "model.safetensors"),
            str(root / "output dir" / "model.progress.json"),
            str(bin_dir / "orbi_streammoe_streamed_dense_convert"),
            "128",
            "67108864",
            "1073741824",
            "4",
        ):
            if expected not in dense:
                raise RuntimeError(f"dense command value missing: {expected}")

        full = plan["phase_commands"]["full_checkpoint"]
        if full != [
            str(bin_dir / "orbi_streammoe_finalize_full_checkpoint"),
            str(root / "output dir"),
            str(root / "output dir" / "model.progress.json"),
        ]:
            raise RuntimeError("full checkpoint command mismatch")

        for phase, command in plan["operator_commands"].items():
            if "--phase" not in command or phase not in command:
                raise RuntimeError(f"operator phase binding missing: {phase}")
            if "--" not in command:
                raise RuntimeError("operator separator missing")
            tail = command[command.index("--") + 1 :]
            if tail != plan["phase_commands"][phase]:
                raise RuntimeError(f"operator command tail mismatch: {phase}")

        output = root / "command plan.json"
        write_immutable(output, plan)
        write_immutable(output, plan)
        changed = dict(plan)
        changed["bin_dir"] = "different"
        expect_failure(
            lambda: write_immutable(output, changed),
            "immutable command plan conflict",
        )

        too_small = json.loads(manifest.read_text(encoding="utf-8"))
        too_small["disk"]["max_batch_source_bytes"] = 1024
        too_small_path = root / "too-small.json"
        write_json(too_small_path, too_small)
        expect_failure(
            lambda: build_phase_commands(
                too_small_path,
                state,
                bin_dir,
                scripts_dir,
                "python-fixture",
            ),
            "expert batch budget",
        )

        print(
            "OSM-41D production command builder: PASS\n"
            "  manifest_to_exact_argv=PASS\n"
            "  shell_false_contract=PASS\n"
            "  expert_batch_budget_derivation=PASS\n"
            "  expert_controller_binding=PASS\n"
            "  expert_finalizer_binding=PASS\n"
            "  dense_controller_binding=PASS\n"
            "  full_checkpoint_binding=PASS\n"
            "  operator_wrapper_binding=PASS\n"
            "  paths_with_spaces_preserved=PASS\n"
            "  immutable_command_plan=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
