#!/usr/bin/env python3
import argparse
import pathlib
import sys
from typing import Callable

from authorize_qwen_windows_conversion import (
    OFFICIAL_MODEL,
    OFFICIAL_SNAPSHOT,
    canonical_digest,
    load_json,
    resolve_bound_path,
    sha256_file,
    verify_probe,
    verify_rehearsal,
)
from launch_qwen_production import execute_next
from run_qwen_production_operator import run_phase, state_backup_path


def resolve_from(base: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    return (base.parent / path).resolve(strict=False)


def _stable_host_bindings(original: dict, current: dict) -> None:
    original_platform = original.get("platform", {})
    current_platform = current.get("platform", {})
    for key in ("system", "machine"):
        if current_platform.get(key) != original_platform.get(key):
            raise RuntimeError(f"current host platform binding changed: {key}")

    original_memory = original.get("memory", {})
    current_memory = current.get("memory", {})
    if int(current_memory.get("total_physical_bytes", 0)) != int(
        original_memory.get("total_physical_bytes", 0)
    ):
        raise RuntimeError("current host total physical RAM binding changed")

    original_paths = original.get("paths", {})
    current_paths = current.get("paths", {})
    for key in ("target_output", "expert_work", "dense_work"):
        before = original_paths.get(key, {})
        now = current_paths.get(key, {})
        if now.get("volume_id") != before.get("volume_id"):
            raise RuntimeError(f"current host volume binding changed: {key}")

    original_vulkan = original.get("vulkan", {})
    current_vulkan = current.get("vulkan", {})
    if current_vulkan.get("summary_sha256") != original_vulkan.get("summary_sha256"):
        raise RuntimeError("current Vulkan summary binding changed")
    if current_vulkan.get("adapter_markers") != original_vulkan.get("adapter_markers"):
        raise RuntimeError("current Vulkan adapter binding changed")


def verify_guarded_start(
    authorization_path: pathlib.Path,
    current_probe_path: pathlib.Path,
    state_path: pathlib.Path,
) -> dict:
    authorization = load_json(authorization_path)
    if authorization.get("schema_version") != 1:
        raise RuntimeError("unsupported conversion authorization schema")
    if authorization.get("stage") != "windows-full-conversion-authorization":
        raise RuntimeError("not an OSM-42B Windows conversion authorization")
    if authorization.get("authorized") is not True:
        raise RuntimeError("conversion authorization is not authorized")

    expected_id = authorization.get("authorization_id")
    unsigned = dict(authorization)
    unsigned.pop("authorization_id", None)
    if expected_id != canonical_digest(unsigned)[:24]:
        raise RuntimeError("conversion authorization digest mismatch")

    source = authorization.get("source", {})
    if source.get("model") != OFFICIAL_MODEL or source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("conversion authorization source pin mismatch")

    safety = authorization.get("safety", {})
    if (
        safety.get("conversion_started")
        or safety.get("operator_state_initialized")
        or safety.get("authorization_is_machine_specific") is not True
    ):
        raise RuntimeError("conversion authorization safety boundary is invalid")

    bindings = authorization.get("bindings", {})
    rehearsal_path = resolve_from(
        authorization_path, bindings.get("rehearsal_bundle", "")
    )
    original_probe_path = resolve_from(
        authorization_path, bindings.get("host_probe", "")
    )
    if not rehearsal_path.is_file() or not original_probe_path.is_file():
        raise RuntimeError("authorization-bound evidence is missing")
    if sha256_file(rehearsal_path) != bindings.get("rehearsal_bundle_sha256"):
        raise RuntimeError("authorization rehearsal bundle hash mismatch")
    if sha256_file(original_probe_path) != bindings.get("host_probe_sha256"):
        raise RuntimeError("authorization host probe hash mismatch")

    bundle, execution = verify_rehearsal(rehearsal_path)
    if execution.get("execution_id") != source.get("execution_id"):
        raise RuntimeError("authorization execution_id binding mismatch")

    production = bundle.get("production_authorization", {})
    if production.get("manifest_sha256") != bindings.get("execution_manifest_sha256"):
        raise RuntimeError("authorization execution manifest binding mismatch")
    if production.get("command_plan_sha256") != bindings.get("command_plan_sha256"):
        raise RuntimeError("authorization command plan binding mismatch")

    manifest_path = resolve_bound_path(rehearsal_path, production.get("manifest", ""))
    command_plan_path = resolve_bound_path(
        rehearsal_path, production.get("command_plan", "")
    )

    if state_path.exists() or state_backup_path(state_path).exists():
        raise RuntimeError(
            "guarded first start requires uninitialized operator state; resume with OSM-41E"
        )

    original_probe = load_json(original_probe_path)
    current_probe = load_json(current_probe_path)
    required_memory = int(
        authorization.get("checks", {}).get("required_available_memory_bytes", 0)
    )
    if required_memory <= 0:
        raise RuntimeError("authorization RAM floor is missing")
    fresh_checks = verify_probe(current_probe, execution, required_memory)
    _stable_host_bindings(original_probe, current_probe)

    return {
        "authorization_id": expected_id,
        "execution_id": source["execution_id"],
        "manifest": manifest_path,
        "command_plan": command_plan_path,
        "fresh_checks": fresh_checks,
    }


def guarded_start(
    authorization_path: pathlib.Path,
    current_probe_path: pathlib.Path,
    state_path: pathlib.Path,
    bin_dir: pathlib.Path,
    scripts_dir: pathlib.Path,
    python_executable: str,
    receipt_dir: pathlib.Path | None = None,
    phase_executor: Callable = run_phase,
) -> dict:
    verified = verify_guarded_start(
        authorization_path,
        current_probe_path,
        state_path,
    )
    result = execute_next(
        verified["manifest"],
        state_path,
        bin_dir,
        scripts_dir,
        python_executable,
        verified["command_plan"],
        phase_executor=phase_executor,
        receipt_dir=receipt_dir,
    )
    result["action"] = "guarded_start"
    result["authorization_id"] = verified["authorization_id"]
    result["fresh_host_checks"] = verified["fresh_checks"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--current-host-probe", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--bin-dir", required=True)
    parser.add_argument(
        "--scripts-dir",
        default=str(pathlib.Path(__file__).resolve().parent),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--receipt-dir")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    authorization = pathlib.Path(args.authorization)
    current_probe = pathlib.Path(args.current_host_probe)
    state = pathlib.Path(args.state)

    if args.verify_only:
        verified = verify_guarded_start(authorization, current_probe, state)
        printable = {
            "action": "verify_only",
            "authorization_id": verified["authorization_id"],
            "execution_id": verified["execution_id"],
            "manifest": str(verified["manifest"]),
            "command_plan": str(verified["command_plan"]),
            "fresh_host_checks": verified["fresh_checks"],
            "operator_state_initialized": False,
            "conversion_started": False,
        }
        import json
        print(json.dumps(printable, indent=2, sort_keys=True))
        return 0

    result = guarded_start(
        authorization,
        current_probe,
        state,
        pathlib.Path(args.bin_dir),
        pathlib.Path(args.scripts_dir),
        args.python_executable,
        pathlib.Path(args.receipt_dir) if args.receipt_dir else None,
    )
    import json
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
