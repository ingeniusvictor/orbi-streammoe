#!/usr/bin/env python3
import json
import pathlib
import tempfile

from run_qwen_production_operator import (
    ensure_phase_can_run,
    load_state,
    phase_report,
    write_state,
)
from launch_qwen_production import (
    execute_all,
    execute_next,
    preview,
)


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def manifest_fixture(root: pathlib.Path) -> pathlib.Path:
    path = root / "execution.json"
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
            "model": "fixture/model",
            "snapshot": "fixture-snapshot",
            "metadata_dir": str(root / "metadata"),
            "dense_inventory": str(root / "dense-inventory.json"),
        },
        "target": {
            "output_dir": str(root / "output"),
            "expert_work_dir": str(root / "expert-work"),
            "dense_work_dir": str(root / "dense-work"),
            "dense_output": str(root / "output" / "model.safetensors"),
            "dense_journal": str(root / "output" / "model.progress.json"),
        },
        "expert_conversion": {
            "geometry": {
                "hidden_size": 8,
                "moe_intermediate_size": 8,
                "expert_count": 3,
                "layer_count": 4,
                "expert_stride": 480,
                "layer_bytes": 1440,
            },
            "converter": "orbi_streammoe_qpack_convert_range",
            "finalizer": "orbi_streammoe_qpack_finalize_experts",
        },
        "dense_conversion": {
            "controller": "scripts/run_qwen_dense_conversion.py",
            "converter": "orbi_streammoe_streamed_dense_convert",
            "chunk_rows": 2,
            "max_chunks": 3,
            "max_batch_source_bytes": 4096,
            "min_free_disk_bytes": 1024,
        },
        "full_checkpoint": {
            "finalizer": "orbi_streammoe_finalize_full_checkpoint",
        },
        "disk": {
            "max_batch_source_bytes": 4096,
            "minimum_free_reserve_bytes": 1024,
        },
    }
    write_json(path, payload)
    return path


def fake_phase_executor(
    manifest_path,
    state_path,
    phase,
    command,
    dry_run=False,
):
    if dry_run:
        raise RuntimeError("fixture executor must not receive dry_run")
    state, _ = load_state(state_path)
    ensure_phase_can_run(state, phase)
    phase_state = state["phases"][phase]
    phase_state["status"] = "completed"
    phase_state["attempts"] += 1
    phase_state["last_exit_code"] = 0
    phase_state["last_command"] = list(command)
    state["completed"] = all(
        state["phases"][item]["status"] == "completed"
        for item in state["phase_order"]
    )
    write_state(state_path, state)
    return phase_report(state)


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="orbi-streammoe-osm41e-"
    ) as temp:
        root = pathlib.Path(temp)
        manifest = manifest_fixture(root)
        state = root / "operator-state.json"
        command_plan = root / "command-plan.json"
        bin_dir = root / "bin"
        scripts_dir = root / "scripts"

        before = preview(
            manifest, state, bin_dir, scripts_dir,
            "python-fixture", command_plan,
        )
        if before["state_initialized"]:
            raise RuntimeError("preview must not initialize state")
        if before["next_phase"] != "expert_conversion":
            raise RuntimeError("preview next phase mismatch")
        if state.exists():
            raise RuntimeError("preview mutated operator state")
        if not command_plan.exists():
            raise RuntimeError("preview did not persist immutable command plan")

        first = execute_next(
            manifest, state, bin_dir, scripts_dir,
            "python-fixture", command_plan,
            phase_executor=fake_phase_executor,
        )
        if first["phase"] != "expert_conversion":
            raise RuntimeError("first next phase mismatch")
        if first["next_phase"] != "expert_finalization":
            raise RuntimeError("first resume target mismatch")
        if not state.exists():
            raise RuntimeError("next did not initialize durable state")

        second = execute_next(
            manifest, state, bin_dir, scripts_dir,
            "python-fixture", command_plan,
            phase_executor=fake_phase_executor,
        )
        if second["phase"] != "expert_finalization":
            raise RuntimeError("second next phase mismatch")
        if second["next_phase"] != "dense_conversion":
            raise RuntimeError("second resume target mismatch")

        limited = execute_all(
            manifest, state, bin_dir, scripts_dir,
            "python-fixture", command_plan,
            max_phases=1,
            phase_executor=fake_phase_executor,
        )
        if limited["phase_count"] != 1:
            raise RuntimeError("max phase checkpoint mismatch")
        if limited["completed"]:
            raise RuntimeError("bounded all run must remain incomplete")
        if limited["next_phase"] != "full_checkpoint":
            raise RuntimeError("bounded all next phase mismatch")
        if limited["stop_reason"] != "phase_limit":
            raise RuntimeError("bounded all stop reason mismatch")

        final = execute_all(
            manifest, state, bin_dir, scripts_dir,
            "python-fixture", command_plan,
            phase_executor=fake_phase_executor,
        )
        if not final["completed"]:
            raise RuntimeError("all did not complete remaining phases")
        if final["phase_count"] != 1:
            raise RuntimeError("all should execute one remaining phase")
        if final["next_phase"] is not None:
            raise RuntimeError("completed execution still has next phase")

        after = preview(
            manifest, state, bin_dir, scripts_dir,
            "python-fixture", command_plan,
        )
        if not after["state_initialized"] or not after["completed"]:
            raise RuntimeError("completed preview mismatch")
        if after["phase_command"] is not None:
            raise RuntimeError("completed preview must have no phase command")

        no_op = execute_next(
            manifest, state, bin_dir, scripts_dir,
            "python-fixture", command_plan,
            phase_executor=fake_phase_executor,
        )
        if no_op["stop_reason"] != "already_complete":
            raise RuntimeError("completed next must be idempotent no-op")

        print(
            "OSM-41E one-command production launcher: PASS\n"
            "  preview_no_state_mutation=PASS\n"
            "  immutable_command_plan_reuse=PASS\n"
            "  next_phase_execution=PASS\n"
            "  durable_resume=PASS\n"
            "  bounded_all_checkpoint=PASS\n"
            "  complete_remaining_phases=PASS\n"
            "  completed_preview=PASS\n"
            "  completed_next_idempotent=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
