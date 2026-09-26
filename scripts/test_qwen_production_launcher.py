#!/usr/bin/env python3
import json
import pathlib
import tempfile
import subprocess
import sys

from run_qwen_production_operator import state_backup_path

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
    prepare_manifest,
    default_command_plan,
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


def expect_failure(fn, label):
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")


def verify_real_execution(root: pathlib.Path) -> None:
    # The real command builder, launcher, durable runner and OS subprocess
    # are used together; only the expensive conversion controller is a fixture.
    root.mkdir()
    manifest = manifest_fixture(root)
    state = root / "operator-state.json"
    plan = root / "commands.json"
    scripts = root / "scripts with spaces"
    scripts.mkdir()
    fixture = scripts / "run_qwen_expert_conversion.py"
    fixture.write_text(
        "import json, pathlib, sys\n"
        "root = pathlib.Path(__file__).parent\n"
        "(root / 'argv.json').write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        "sys.exit(7 if (root / 'fail').exists() else 0)\n",
        encoding="utf-8",
    )
    # Shell metacharacters must remain literal argv, including on Windows.
    binary_dir = root / "bin & literal"
    args = (manifest, state, binary_dir, scripts, sys.executable, plan)
    expected = preview(*args)["phase_command"]
    marker = scripts / "fail"
    marker.touch()
    expect_failure(lambda: execute_all(*args), "real subprocess failure")
    durable, _ = load_state(state)
    phase = durable["phases"]["expert_conversion"]
    if phase["status"] != "failed" or phase["last_exit_code"] != 7:
        raise RuntimeError("subprocess failure not durably recorded")
    if durable["phases"]["expert_finalization"]["attempts"] != 0:
        raise RuntimeError("launcher advanced beyond failed conversion")
    received = json.loads((scripts / "argv.json").read_text(encoding="utf-8"))
    if received != expected[2:]:
        raise RuntimeError("OS subprocess did not receive exact generated argv")
    if phase["last_command"] != expected:
        raise RuntimeError("durable command differs from command plan")

    # Backup-only preview must inspect without restoring or deleting any file.
    backup = state_backup_path(state)
    state.replace(backup)
    before = backup.read_bytes()
    report = preview(*args)
    if report["next_phase"] != "expert_conversion":
        raise RuntimeError("backup preview chose wrong phase")
    if state.exists() or backup.read_bytes() != before:
        raise RuntimeError("backup-only preview mutated durable state")

    # The CLI preview has the same read-only contract.
    command = [sys.executable, str(pathlib.Path(__file__).with_name("launch_qwen_production.py")),
               "--manifest", str(manifest), "--state", str(state),
               "--bin-dir", str(binary_dir), "--scripts-dir", str(scripts), "preview"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if json.loads(completed.stdout)["mutated_state"] or state.exists():
        raise RuntimeError("CLI preview restored backup")
    marker.unlink()
    retried = execute_next(*args)
    if retried["next_phase"] != "expert_finalization":
        raise RuntimeError("real retry did not advance exactly one phase")
    durable, _ = load_state(state)
    if durable["phases"]["expert_conversion"]["attempts"] != 2 or backup.exists():
        raise RuntimeError("real retry/backup recovery mismatch")

    # Simulate interruption immediately after the durable running transition.
    durable["phases"]["expert_conversion"]["status"] = "running"
    write_state(state, durable)
    execute_next(*args)
    durable, _ = load_state(state)
    if durable["phases"]["expert_conversion"]["interrupted_retries"] != 1:
        raise RuntimeError("interrupted execution was not tracked")

    before = state.read_bytes()
    expect_failure(lambda: execute_next(manifest, state, root / "changed-bin", scripts,
                                       sys.executable, plan), "immutable command plan conflict")
    if state.read_bytes() != before:
        raise RuntimeError("command conflict changed durable state")
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    expect_failure(lambda: preview(manifest, state, binary_dir, scripts, sys.executable),
                   "manifest byte binding")
    if state.read_bytes() != before:
        raise RuntimeError("manifest conflict changed durable state")


def verify_preflight_and_default_plan(root: pathlib.Path) -> None:
    from test_qwen_production_execution import fixture_args
    request = vars(fixture_args(root))
    manifest = pathlib.Path(request.pop("manifest"))
    preflight = root / "preflight.json"
    state = root / "state.json"
    write_json(preflight, request)
    prepare_manifest(manifest, preflight)
    before = manifest.read_bytes()
    prepare_manifest(manifest, preflight)
    if manifest.read_bytes() != before:
        raise RuntimeError("manifest reuse changed authorization")
    request["chunk_rows"] += 1
    write_json(preflight, request)
    expect_failure(lambda: prepare_manifest(manifest, preflight), "preflight conflict")
    if manifest.read_bytes() != before:
        raise RuntimeError("conflicting preflight changed manifest")
    args = (manifest, state, root / "bin", root / "scripts", sys.executable)
    preview(*args)
    if state.exists() or default_command_plan(state).exists():
        raise RuntimeError("default preview created durable files")
    execute_next(*args, phase_executor=fake_phase_executor)
    if not default_command_plan(state).exists():
        raise RuntimeError("execution did not persist default command plan")
    before_state = state.read_bytes()
    expect_failure(lambda: execute_next(manifest, state, root / "different-bin",
                                       root / "scripts", sys.executable,
                                       phase_executor=fake_phase_executor), "default plan conflict")
    if state.read_bytes() != before_state:
        raise RuntimeError("default plan conflict mutated state")


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="orbi-streammoe-osm41e-"
    ) as temp:
        root = pathlib.Path(temp)
        verify_real_execution(root / "real subprocess")
        verify_preflight_and_default_plan(root / "preflight")
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
            "  real_subprocess_failure_resume_exact_argv=PASS\n"
            "  backup_only_preview_read_only=PASS\n"
            "  interrupted_resume_and_conflict_guards=PASS\n"
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
