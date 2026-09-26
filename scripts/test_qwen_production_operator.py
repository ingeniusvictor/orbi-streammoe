#!/usr/bin/env python3
import json
import pathlib
import sys
import tempfile

from run_qwen_production_operator import (
    PHASES,
    initialize,
    load_state,
    run_phase,
    state_backup_path,
    status,
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
        "expert_conversion": {
            "converter": "orbi_streammoe_qpack_convert_range",
            "finalizer": "orbi_streammoe_qpack_finalize_experts",
        },
        "dense_conversion": {
            "controller": "scripts/run_qwen_dense_conversion.py",
        },
        "full_checkpoint": {
            "finalizer": "orbi_streammoe_finalize_full_checkpoint",
        },
    }
    write_json(path, payload)
    return path


def command(component: str, exit_code: int = 0) -> list[str]:
    return [
        sys.executable,
        "-c",
        f"import sys; sys.exit({exit_code})",
        component,
    ]


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except Exception:
        return
    raise RuntimeError(f"expected failure: {label}")


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="orbi-streammoe-osm41b-"
    ) as temp:
        root = pathlib.Path(temp)
        manifest = manifest_fixture(root)
        state_path = root / "operator-state.json"

        initial = initialize(manifest, state_path)
        if initial["next_phase"] != "expert_conversion":
            raise RuntimeError("initial phase mismatch")
        if initial["completed"]:
            raise RuntimeError("fresh execution cannot be complete")

        same = initialize(manifest, state_path)
        if same["execution_id"] != initial["execution_id"]:
            raise RuntimeError("idempotent init changed execution")

        expect_failure(
            lambda: run_phase(
                manifest,
                state_path,
                "dense_conversion",
                command("scripts/run_qwen_dense_conversion.py"),
            ),
            "out-of-order phase",
        )

        before = state_path.read_text(encoding="utf-8")
        dry = run_phase(
            manifest,
            state_path,
            "expert_conversion",
            command("orbi_streammoe_qpack_convert_range"),
            dry_run=True,
        )
        if not dry["dry_run"]:
            raise RuntimeError("dry-run flag missing")
        if state_path.read_text(encoding="utf-8") != before:
            raise RuntimeError("dry-run mutated operator state")

        failed = False
        try:
            run_phase(
                manifest,
                state_path,
                "expert_conversion",
                command("orbi_streammoe_qpack_convert_range", 7),
            )
        except RuntimeError:
            failed = True
        if not failed:
            raise RuntimeError("failed subprocess must fail phase")

        state, _ = load_state(state_path)
        expert = state["phases"]["expert_conversion"]
        if expert["status"] != "failed" or expert["last_exit_code"] != 7:
            raise RuntimeError("failed phase state mismatch")

        run_phase(
            manifest,
            state_path,
            "expert_conversion",
            command("orbi_streammoe_qpack_convert_range"),
        )
        state, _ = load_state(state_path)
        if state["phases"]["expert_conversion"]["attempts"] != 2:
            raise RuntimeError("failed retry attempt count mismatch")

        # Simulate a process crash after durable transition to running.
        state["phases"]["expert_finalization"]["status"] = "running"
        state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_phase(
            manifest,
            state_path,
            "expert_finalization",
            command("orbi_streammoe_qpack_finalize_experts"),
        )
        state, _ = load_state(state_path)
        if state["phases"]["expert_finalization"]["interrupted_retries"] != 1:
            raise RuntimeError("interrupted running phase was not tracked")

        # Simulate primary-state loss and prove backup recovery.
        backup = state_backup_path(state_path)
        state_path.replace(backup)
        recovered = status(manifest, state_path)
        if recovered["next_phase"] != "dense_conversion":
            raise RuntimeError("backup recovery phase mismatch")
        if not state_path.exists() or backup.exists():
            raise RuntimeError("backup state was not restored canonically")

        run_phase(
            manifest,
            state_path,
            "dense_conversion",
            command("scripts/run_qwen_dense_conversion.py"),
        )
        final = run_phase(
            manifest,
            state_path,
            "full_checkpoint",
            command("orbi_streammoe_finalize_full_checkpoint"),
        )
        if not final["completed"] or final["next_phase"] is not None:
            raise RuntimeError("final execution state mismatch")

        for phase in PHASES:
            if final["phases"][phase]["status"] != "completed":
                raise RuntimeError(f"phase not completed: {phase}")

        expect_failure(
            lambda: run_phase(
                manifest,
                state_path,
                "full_checkpoint",
                command("orbi_streammoe_finalize_full_checkpoint"),
            ),
            "rerun completed execution",
        )

        changed = json.loads(manifest.read_text(encoding="utf-8"))
        changed["execution_id"] = "xxxxxxxxxxxxxxxxxxxx"
        write_json(manifest, changed)
        expect_failure(
            lambda: status(manifest, state_path),
            "manifest binding change",
        )

        print(
            "OSM-41B restart-safe operator runner: PASS\n"
            "  immutable_manifest_binding=PASS\n"
            "  ordered_phase_machine=PASS\n"
            "  dry_run_no_mutation=PASS\n"
            "  failed_phase_retry=PASS\n"
            "  interrupted_phase_retry=PASS\n"
            "  backup_state_recovery=PASS\n"
            "  subprocess_exit_gate=PASS\n"
            "  final_completion_gate=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
