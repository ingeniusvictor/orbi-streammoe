#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib

from authorize_qwen_windows_conversion import (
    OFFICIAL_MODEL,
    OFFICIAL_SNAPSHOT,
    canonical_digest,
    load_json,
    sha256_file,
)
from record_qwen_production_receipts import (
    completion_receipt_path,
    default_receipt_dir,
    sync_production_audit,
)
from run_qwen_production_operator import (
    PHASES,
    load_state,
    validate_manifest,
    validate_state_binding,
)


SCHEMA_VERSION = 1


def canonical_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = canonical_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing OSM-42D evidence conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def resolve_from(base: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    return (base.parent / path).resolve(strict=False)


def verify_authorization(
    authorization_path: pathlib.Path,
    manifest_path: pathlib.Path,
    command_plan_path: pathlib.Path,
) -> dict:
    authorization = load_json(authorization_path)
    if authorization.get("schema_version") != 1:
        raise RuntimeError("unsupported OSM-42B authorization schema")
    if authorization.get("stage") != "windows-full-conversion-authorization":
        raise RuntimeError("not an OSM-42B Windows conversion authorization")
    if authorization.get("authorized") is not True:
        raise RuntimeError("Windows conversion authorization is not authorized")

    expected_id = authorization.get("authorization_id")
    unsigned = dict(authorization)
    unsigned.pop("authorization_id", None)
    if expected_id != canonical_digest(unsigned)[:24]:
        raise RuntimeError("Windows conversion authorization digest mismatch")

    source = authorization.get("source", {})
    if source.get("model") != OFFICIAL_MODEL:
        raise RuntimeError("authorization model pin mismatch")
    if source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("authorization snapshot pin mismatch")

    bindings = authorization.get("bindings", {})
    if sha256_file(manifest_path) != bindings.get("execution_manifest_sha256"):
        raise RuntimeError("authorization manifest SHA-256 binding failed")
    if sha256_file(command_plan_path) != bindings.get("command_plan_sha256"):
        raise RuntimeError("authorization command-plan SHA-256 binding failed")

    bound_probe = resolve_from(
        authorization_path,
        bindings.get("host_probe", ""),
    )
    if not bound_probe.is_file():
        raise RuntimeError("authorization-bound Windows host probe is missing")
    if sha256_file(bound_probe) != bindings.get("host_probe_sha256"):
        raise RuntimeError("authorization-bound Windows host probe hash mismatch")

    if authorization.get("safety", {}).get("authorization_is_machine_specific") is not True:
        raise RuntimeError("authorization is not machine-specific")
    return authorization


def certify_windows_conversion(
    authorization_path: pathlib.Path,
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    command_plan_path: pathlib.Path,
    receipt_dir: pathlib.Path | None = None,
) -> dict:
    authorization = verify_authorization(
        authorization_path,
        manifest_path,
        command_plan_path,
    )

    manifest = load_json(manifest_path)
    validate_manifest(manifest)
    state, _ = load_state(state_path, recover=False)
    validate_state_binding(state, manifest_path, manifest)

    if manifest.get("source", {}).get("model") != OFFICIAL_MODEL:
        raise RuntimeError("execution manifest is not the official Qwen3-Next model")
    if manifest.get("source", {}).get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("execution manifest snapshot pin mismatch")
    if state.get("execution_id") != authorization.get("source", {}).get("execution_id"):
        raise RuntimeError("operator state execution_id disagrees with authorization")
    if state.get("completed") is not True:
        raise RuntimeError("full Windows production execution is not complete")

    for phase in PHASES:
        phase_state = state.get("phases", {}).get(phase, {})
        if phase_state.get("status") != "completed":
            raise RuntimeError(f"production phase is incomplete: {phase}")
        if phase_state.get("last_exit_code") != 0:
            raise RuntimeError(f"production phase exit code is not zero: {phase}")

    audit = sync_production_audit(
        manifest_path,
        state_path,
        command_plan_path,
        receipt_dir,
        verify_existing_completion=True,
    )
    if audit.get("completed") is not True:
        raise RuntimeError("OSM-41F audit does not report completed execution")

    actual_receipt_dir = receipt_dir or default_receipt_dir(state_path)
    completion_path = completion_receipt_path(actual_receipt_dir)
    if not completion_path.is_file():
        raise RuntimeError("OSM-41F completion receipt is missing")
    completion = load_json(completion_path)
    package = completion.get("package", {})
    package_digest = package.get("package_digest_sha256")
    if not isinstance(package_digest, str) or len(package_digest) != 64:
        raise RuntimeError("completion receipt package digest is invalid")
    if package.get("package_stage") != "full-checkpoint":
        raise RuntimeError("completion receipt is not a full-checkpoint package")

    host_probe_path = resolve_from(
        authorization_path,
        authorization["bindings"]["host_probe"],
    )
    host_probe = load_json(host_probe_path)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "windows-full-conversion-certification",
        "source": {
            "model": OFFICIAL_MODEL,
            "snapshot": OFFICIAL_SNAPSHOT,
            "execution_id": state["execution_id"],
        },
        "authorization": {
            "path": str(authorization_path),
            "authorization_id": authorization["authorization_id"],
            "sha256": sha256_file(authorization_path),
        },
        "execution_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "operator_state": {
            "path": str(state_path),
            "sha256": sha256_file(state_path),
            "completed": True,
        },
        "command_plan": {
            "path": str(command_plan_path),
            "sha256": sha256_file(command_plan_path),
        },
        "completion_receipt": {
            "path": str(completion_path),
            "sha256": sha256_file(completion_path),
            "phase_chain_tail_sha256": completion.get("phase_chain_tail_sha256"),
        },
        "package": {
            "output_dir": package.get("output_dir"),
            "package_digest_sha256": package_digest,
            "declared_file_count": package.get("declared_file_count"),
            "declared_file_bytes": package.get("declared_file_bytes"),
        },
        "authorized_windows_host": {
            "probe_path": str(host_probe_path),
            "probe_sha256": sha256_file(host_probe_path),
            "platform": host_probe.get("platform"),
            "memory": host_probe.get("memory"),
            "vulkan": host_probe.get("vulkan"),
        },
        "phase_summary": {
            phase: {
                "attempts": state["phases"][phase]["attempts"],
                "interrupted_retries": state["phases"][phase]["interrupted_retries"],
                "exit_code": state["phases"][phase]["last_exit_code"],
            }
            for phase in PHASES
        },
        "claims": {
            "full_official_checkpoint_conversion_completed": True,
            "full_checkpoint_package_deep_audited": True,
            "runtime_inference_validated": False,
            "tokens_per_second_measured": False,
            "windows_hardware_inference_certified": False,
        },
    }
    payload["certification_id"] = canonical_digest(payload)[:24]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--command-plan", required=True)
    parser.add_argument("--receipt-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = certify_windows_conversion(
        pathlib.Path(args.authorization),
        pathlib.Path(args.manifest),
        pathlib.Path(args.state),
        pathlib.Path(args.command_plan),
        pathlib.Path(args.receipt_dir) if args.receipt_dir else None,
    )
    write_immutable(pathlib.Path(args.output), payload)
    print(canonical_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
