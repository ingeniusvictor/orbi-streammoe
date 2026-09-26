#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys
from typing import Callable

from plan_qwen_production_execution import (
    build_manifest,
    create_parser as create_preflight_parser,
    write_immutable as write_manifest,
)

from build_qwen_production_commands import (
    build_phase_commands,
    write_immutable as write_command_plan,
)
from run_qwen_production_operator import (
    PHASES,
    initialize,
    run_phase,
    state_backup_path,
    status,
)


def canonical_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def state_exists(path: pathlib.Path) -> bool:
    return path.exists() or state_backup_path(path).exists()


def prepare_manifest(manifest_path: pathlib.Path, preflight_path: pathlib.Path) -> None:
    """Create/reuse OSM-41A authorization using its existing CLI validation."""
    request = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise RuntimeError("preflight request must be a JSON object")
    parser = create_preflight_parser()
    options = {
        action.dest: action.option_strings[0]
        for action in parser._actions
        if action.dest not in {"help", "manifest"}
    }
    argv = ["--manifest", str(manifest_path)]
    for key, value in request.items():
        if key not in options or type(value) not in {str, int}:
            raise RuntimeError(f"invalid preflight field: {key}")
        argv.append(f"{options[key]}={value}")
    args = parser.parse_args(argv)
    if args.group_size <= 0 or (args.max_chunks is not None and args.max_chunks <= 0):
        raise RuntimeError("group_size and max_chunks must be positive")
    write_manifest(manifest_path, build_manifest(args))


def default_command_plan(state_path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(state_path) + ".commands.json")


def build_launcher_context(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    bin_dir: pathlib.Path,
    scripts_dir: pathlib.Path,
    python_executable: str,
    command_plan_path: pathlib.Path | None = None,
) -> tuple[dict, dict]:
    plan = build_phase_commands(
        manifest_path,
        state_path,
        bin_dir,
        scripts_dir,
        python_executable,
    )
    if command_plan_path is None and default_command_plan(state_path).exists():
        command_plan_path = default_command_plan(state_path)
    if command_plan_path is not None:
        write_command_plan(command_plan_path, plan)

    if state_exists(state_path):
        report = status(manifest_path, state_path, recover=False)
        initialized = True
    else:
        report = {
            "execution_id": plan["execution_id"],
            "completed": False,
            "next_phase": PHASES[0],
            "phases": None,
        }
        initialized = False

    return plan, {
        "execution_id": plan["execution_id"],
        "state_initialized": initialized,
        "completed": bool(report["completed"]),
        "next_phase": report["next_phase"],
        "phase_command": (
            None
            if report["next_phase"] is None
            else plan["phase_commands"][report["next_phase"]]
        ),
        "operator_command": (
            None
            if report["next_phase"] is None
            else plan["operator_commands"][report["next_phase"]]
        ),
    }


def preview(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    bin_dir: pathlib.Path,
    scripts_dir: pathlib.Path,
    python_executable: str,
    command_plan_path: pathlib.Path | None = None,
) -> dict:
    _, report = build_launcher_context(
        manifest_path, state_path, bin_dir, scripts_dir,
        python_executable, command_plan_path,
    )
    report["action"] = "preview"
    report["mutated_state"] = False
    return report


def execute_next(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    bin_dir: pathlib.Path,
    scripts_dir: pathlib.Path,
    python_executable: str,
    command_plan_path: pathlib.Path | None = None,
    phase_executor: Callable = run_phase,
) -> dict:
    command_plan_path = command_plan_path or default_command_plan(state_path)
    plan, _ = build_launcher_context(
        manifest_path, state_path, bin_dir, scripts_dir,
        python_executable, command_plan_path,
    )
    current = initialize(manifest_path, state_path)
    phase = current["next_phase"]
    if phase is None:
        return {
            "action": "next",
            "execution_id": plan["execution_id"],
            "completed": True,
            "next_phase": None,
            "stop_reason": "already_complete",
            "phase": None,
            "command": None,
        }

    command = list(plan["phase_commands"][phase])
    result = phase_executor(
        manifest_path, state_path, phase, command, dry_run=False,
    )
    return {
        "action": "next",
        "execution_id": plan["execution_id"],
        "completed": bool(result["completed"]),
        "next_phase": result["next_phase"],
        "stop_reason": "phase_completed",
        "phase": phase,
        "command": command,
        "phase_report": result,
    }


def execute_all(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    bin_dir: pathlib.Path,
    scripts_dir: pathlib.Path,
    python_executable: str,
    command_plan_path: pathlib.Path | None = None,
    max_phases: int | None = None,
    phase_executor: Callable = run_phase,
) -> dict:
    if max_phases is not None and max_phases <= 0:
        raise RuntimeError("max_phases must be positive")

    command_plan_path = command_plan_path or default_command_plan(state_path)
    plan, _ = build_launcher_context(
        manifest_path, state_path, bin_dir, scripts_dir,
        python_executable, command_plan_path,
    )
    current = initialize(manifest_path, state_path)
    phases_run = []
    last_report = current
    while last_report["next_phase"] is not None:
        if max_phases is not None and len(phases_run) >= max_phases:
            break
        phase = last_report["next_phase"]
        command = list(plan["phase_commands"][phase])
        last_report = phase_executor(
            manifest_path, state_path, phase, command, dry_run=False,
        )
        phases_run.append({"phase": phase, "command": command})

    completed = bool(last_report["completed"])
    return {
        "action": "all",
        "execution_id": plan["execution_id"],
        "completed": completed,
        "next_phase": last_report["next_phase"],
        "stop_reason": (
            "complete"
            if completed
            else "phase_limit"
            if max_phases is not None and len(phases_run) >= max_phases
            else "incomplete"
        ),
        "phases_run": phases_run,
        "phase_count": len(phases_run),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preflight", help="JSON request for immutable OSM-41A manifest creation/reuse")
    parser.add_argument("--state", required=True)
    parser.add_argument("--bin-dir", required=True)
    parser.add_argument(
        "--scripts-dir",
        default=str(pathlib.Path(__file__).resolve().parent),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--command-plan")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("preview")
    sub.add_parser("next")
    all_parser = sub.add_parser("all")
    all_parser.add_argument("--max-phases", type=int)
    args = parser.parse_args()

    manifest = pathlib.Path(args.manifest)
    state = pathlib.Path(args.state)
    bin_dir = pathlib.Path(args.bin_dir)
    scripts_dir = pathlib.Path(args.scripts_dir)
    command_plan = pathlib.Path(args.command_plan) if args.command_plan else None

    if args.preflight:
        prepare_manifest(manifest, pathlib.Path(args.preflight))

    if args.action == "preview":
        result = preview(
            manifest, state, bin_dir, scripts_dir,
            args.python_executable, command_plan,
        )
    elif args.action == "next":
        result = execute_next(
            manifest, state, bin_dir, scripts_dir,
            args.python_executable, command_plan,
        )
    else:
        result = execute_all(
            manifest, state, bin_dir, scripts_dir,
            args.python_executable, command_plan, args.max_phases,
        )

    print(canonical_text(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
