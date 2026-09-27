#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import subprocess

EXPECTED_PHASES = (
    "windows_conversion_certification",
    "runtime_load",
    "first_token",
    "text_first_token",
    "bounded_generation",
    "chat_generation",
    "performance",
    "phase_latency",
    "cache_phase",
)


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pack(pack: dict) -> None:
    if pack.get("schema_version") != 1 or pack.get("stage") != "windows-official-pilot-execution-pack":
        raise RuntimeError("unsupported OSM-46A pack")
    phases = pack.get("phases")
    if not isinstance(phases, list) or [p.get("name") for p in phases] != list(EXPECTED_PHASES):
        raise RuntimeError("OSM-46A phase order mismatch")
    claims = pack.get("claims", {})
    if claims.get("shell_string_execution_used") is not False:
        raise RuntimeError("OSM-46A forbids shell-string execution")
    if claims.get("real_windows_pilot_executed") is not False:
        raise RuntimeError("input pack already crosses execution claim boundary")
    for phase in phases:
        argv = phase.get("argv")
        if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(x, str) and x for x in argv):
            raise RuntimeError(f"invalid argv for phase {phase.get('name')}")
        if not isinstance(phase.get("output"), str) or not phase["output"]:
            raise RuntimeError(f"missing output for phase {phase.get('name')}")
        if not isinstance(phase.get("expected_stage"), str) or not phase["expected_stage"]:
            raise RuntimeError(f"missing expected stage for phase {phase.get('name')}")


def verify_inputs(pack: dict) -> None:
    for label, meta in pack.get("inputs", {}).items():
        path = pathlib.Path(meta.get("path", ""))
        if not path.is_file():
            raise RuntimeError(f"OSM-46A input missing: {label}: {path}")
        if sha256_file(path) != meta.get("sha256"):
            raise RuntimeError(f"OSM-46A input hash changed: {label}")


def output_valid(phase: dict) -> bool:
    path = pathlib.Path(phase["output"])
    if not path.is_file():
        return False
    value = load_json(path)
    return value.get("stage") == phase["expected_stage"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    pack_path = pathlib.Path(args.pack)
    pack = load_json(pack_path)
    verify_pack(pack)
    verify_inputs(pack)

    if args.verify_only:
        print("OSM-46A Windows pilot execution pack: VERIFIED")
        return 0

    state_path = pathlib.Path(args.state)
    state = {
        "schema_version": 1,
        "stage": "windows-official-pilot-run-state",
        "pack_path": str(pack_path.resolve(strict=False)),
        "pack_sha256": sha256_file(pack_path),
        "completed_phases": [],
        "completed": False,
    }
    if state_path.exists():
        existing = load_json(state_path)
        if existing.get("pack_sha256") != state["pack_sha256"]:
            raise RuntimeError("existing OSM-46A state is bound to another pack")
        state = existing

    evidence_dir = pathlib.Path(pack["evidence_dir"])
    evidence_dir.mkdir(parents=True, exist_ok=True)

    for phase in pack["phases"]:
        name = phase["name"]
        if name in state.get("completed_phases", []):
            if not output_valid(phase):
                raise RuntimeError(f"completed phase evidence is missing/invalid: {name}")
            continue

        completed = subprocess.run(
            phase["argv"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"OSM-46A phase failed: {name}: {detail}")
        if not output_valid(phase):
            raise RuntimeError(f"OSM-46A phase produced invalid evidence: {name}")

        state.setdefault("completed_phases", []).append(name)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    state["completed"] = True
    state["final_evidence"] = pack["phases"][-1]["output"]
    state["claims"] = {
        "all_planned_phases_completed": True,
        "real_windows_pilot_executed": True,
        "performance_target_met": False,
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
