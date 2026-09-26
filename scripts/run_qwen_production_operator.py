#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from copy import deepcopy

SCHEMA_VERSION = 1
PHASES = (
    "expert_conversion",
    "expert_finalization",
    "dense_conversion",
    "full_checkpoint",
)

PHASE_COMPONENT_PATHS = {
    "expert_conversion": ("expert_conversion", "converter"),
    "expert_finalization": ("expert_conversion", "finalizer"),
    "dense_conversion": ("dense_conversion", "controller"),
    "full_checkpoint": ("full_checkpoint", "finalizer"),
}


def canonical_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    root = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return root


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported production execution manifest schema")
    if manifest.get("stage") != "production-execution-preflight":
        raise RuntimeError("manifest stage is not production-execution-preflight")
    execution_id = manifest.get("execution_id")
    if not isinstance(execution_id, str) or len(execution_id) != 20:
        raise RuntimeError("manifest execution_id is invalid")
    auth = manifest.get("authorization")
    if not isinstance(auth, dict):
        raise RuntimeError("manifest authorization is missing")
    if (
        auth.get("authorized") is not True
        or auth.get("immutable_preflight") is not True
        or auth.get("full_source_scope") is not True
    ):
        raise RuntimeError("manifest is not authorized for full production execution")

    for phase, path in PHASE_COMPONENT_PATHS.items():
        cursor = manifest
        for key in path:
            cursor = cursor.get(key) if isinstance(cursor, dict) else None
        if not isinstance(cursor, str) or not cursor:
            raise RuntimeError(f"manifest component missing for phase: {phase}")


def expected_component(manifest: dict, phase: str) -> str:
    cursor = manifest
    for key in PHASE_COMPONENT_PATHS[phase]:
        cursor = cursor[key]
    return str(cursor)


def new_state(manifest_path: pathlib.Path, manifest: dict) -> dict:
    validate_manifest(manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_id": manifest["execution_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "phase_order": list(PHASES),
        "phases": {
            phase: {
                "status": "pending",
                "attempts": 0,
                "last_exit_code": None,
                "last_command": None,
                "interrupted_retries": 0,
            }
            for phase in PHASES
        },
        "completed": False,
    }


def state_backup_path(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(path) + ".bak")


def resolve_state_path(path: pathlib.Path) -> pathlib.Path:
    if path.exists():
        return path
    backup = state_backup_path(path)
    if backup.exists():
        return backup
    return path


def write_state(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = pathlib.Path(str(path) + ".tmp")
    backup = state_backup_path(path)

    temp.write_text(canonical_text(payload), encoding="utf-8")
    if backup.exists():
        backup.unlink()
    if path.exists():
        path.replace(backup)
    try:
        temp.replace(path)
    except Exception:
        if backup.exists() and not path.exists():
            backup.replace(path)
        if temp.exists():
            temp.unlink()
        raise
    if backup.exists():
        backup.unlink()


def load_state(path: pathlib.Path, *, recover: bool = True) -> tuple[dict, pathlib.Path]:
    source = resolve_state_path(path)
    if not source.exists():
        raise RuntimeError(f"operator state does not exist: {path}")
    state = load_json(source)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported operator state schema")
    if state.get("phase_order") != list(PHASES):
        raise RuntimeError("operator state phase order changed")
    phases = state.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise RuntimeError("operator state phases are malformed")
    if source != path and recover:
        write_state(path, state)
        source.unlink(missing_ok=True)
        source = path
    return state, source


def validate_state_binding(
    state: dict,
    manifest_path: pathlib.Path,
    manifest: dict,
) -> None:
    validate_manifest(manifest)
    if state.get("execution_id") != manifest.get("execution_id"):
        raise RuntimeError("operator state execution_id disagrees with manifest")
    if state.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("operator state manifest SHA-256 binding failed")


def next_incomplete_phase(state: dict) -> str | None:
    for phase in PHASES:
        if state["phases"][phase]["status"] != "completed":
            return phase
    return None


def ensure_phase_can_run(state: dict, phase: str) -> None:
    if phase not in PHASES:
        raise RuntimeError(f"unknown phase: {phase}")
    current = next_incomplete_phase(state)
    if current is None:
        raise RuntimeError("production execution is already complete")
    if phase != current:
        raise RuntimeError(
            f"phase order violation: next phase is {current}, requested {phase}"
        )
    status = state["phases"][phase]["status"]
    if status == "completed":
        raise RuntimeError(f"phase is already completed: {phase}")
    if status not in {"pending", "running", "failed"}:
        raise RuntimeError(f"unsupported phase status: {status}")


def command_mentions_component(command: list[str], component: str) -> bool:
    expected = pathlib.Path(component).name
    for item in command:
        if pathlib.Path(str(item)).name == expected:
            return True
        if str(item) == component:
            return True
    return False


def phase_report(state: dict) -> dict:
    current = next_incomplete_phase(state)
    return {
        "execution_id": state["execution_id"],
        "completed": bool(state.get("completed")),
        "next_phase": current,
        "phases": deepcopy(state["phases"]),
    }


def initialize(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
) -> dict:
    manifest = load_json(manifest_path)
    validate_manifest(manifest)

    if state_path.exists() or state_backup_path(state_path).exists():
        state, _ = load_state(state_path)
        validate_state_binding(state, manifest_path, manifest)
        return phase_report(state)

    state = new_state(manifest_path, manifest)
    write_state(state_path, state)
    return phase_report(state)


def run_phase(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    phase: str,
    command: list[str],
    dry_run: bool = False,
) -> dict:
    if not command:
        raise RuntimeError("phase command cannot be empty")

    manifest = load_json(manifest_path)
    state, _ = load_state(state_path)
    validate_state_binding(state, manifest_path, manifest)
    ensure_phase_can_run(state, phase)

    component = expected_component(manifest, phase)
    if not command_mentions_component(command, component):
        raise RuntimeError(
            f"phase command does not reference manifest component: {component}"
        )

    if dry_run:
        report = phase_report(state)
        report["dry_run"] = True
        report["phase"] = phase
        report["component"] = component
        report["command"] = command
        return report

    phase_state = state["phases"][phase]
    if phase_state["status"] == "running":
        phase_state["interrupted_retries"] += 1

    phase_state["status"] = "running"
    phase_state["attempts"] += 1
    phase_state["last_exit_code"] = None
    phase_state["last_command"] = command
    write_state(state_path, state)

    completed = subprocess.run(command, check=False)
    phase_state["last_exit_code"] = int(completed.returncode)

    if completed.returncode == 0:
        phase_state["status"] = "completed"
    else:
        phase_state["status"] = "failed"

    state["completed"] = all(
        state["phases"][item]["status"] == "completed"
        for item in PHASES
    )
    write_state(state_path, state)

    report = phase_report(state)
    report["phase"] = phase
    report["component"] = component
    report["exit_code"] = int(completed.returncode)

    if completed.returncode != 0:
        raise RuntimeError(
            f"phase command failed: phase={phase} exit_code={completed.returncode}"
        )
    return report


def status(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    *,
    recover: bool = True,
) -> dict:
    manifest = load_json(manifest_path)
    state, _ = load_state(state_path, recover=recover)
    validate_state_binding(state, manifest_path, manifest)
    return phase_report(state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("init")
    sub.add_parser("status")

    run = sub.add_parser("run")
    run.add_argument("--phase", choices=PHASES, required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    manifest_path = pathlib.Path(args.manifest)
    state_path = pathlib.Path(args.state)

    if args.action == "init":
        result = initialize(manifest_path, state_path)
    elif args.action == "status":
        result = status(manifest_path, state_path)
    else:
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        result = run_phase(
            manifest_path,
            state_path,
            args.phase,
            command,
            dry_run=args.dry_run,
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
