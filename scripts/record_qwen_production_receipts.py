#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
from typing import Any

from run_qwen_production_operator import (
    PHASES,
    load_json as load_execution_json,
    load_state,
    validate_manifest,
    validate_state_binding,
)

SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 1024 * 1024


def canonical_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    root = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return root


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = canonical_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                f"existing production receipt conflicts with requested evidence: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = pathlib.Path(str(path) + ".tmp")
    temp.write_text(rendered, encoding="utf-8")
    if path.exists():
        temp.unlink(missing_ok=True)
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                f"existing production receipt conflicts with requested evidence: {path}"
            )
        return
    temp.replace(path)


def default_receipt_dir(state_path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(state_path) + ".receipts")


def phase_receipt_path(receipt_dir: pathlib.Path, phase: str) -> pathlib.Path:
    try:
        index = PHASES.index(phase)
    except ValueError as exc:
        raise RuntimeError(f"unknown production phase: {phase}") from exc
    return receipt_dir / f"{index:02d}-{phase}.json"


def completion_receipt_path(receipt_dir: pathlib.Path) -> pathlib.Path:
    return receipt_dir / "completion.json"


def _validate_plan(
    plan: dict,
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    execution_id: str,
) -> None:
    if plan.get("schema_version") != 1:
        raise RuntimeError("unsupported production command-plan schema")
    if plan.get("stage") != "production-operator-command-plan":
        raise RuntimeError("command plan stage is not production-operator-command-plan")
    if plan.get("execution_id") != execution_id:
        raise RuntimeError("command plan execution_id disagrees with operator state")
    if plan.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("command plan manifest SHA-256 binding failed")
    if plan.get("state_path") != str(state_path):
        raise RuntimeError("command plan state path disagrees with requested state")
    if plan.get("phase_order") != list(PHASES):
        raise RuntimeError("command plan phase order changed")
    if plan.get("shell") is not False:
        raise RuntimeError("command plan must preserve shell=false")

    commands = plan.get("phase_commands")
    if not isinstance(commands, dict) or set(commands) != set(PHASES):
        raise RuntimeError("command plan phase command inventory is malformed")
    for phase in PHASES:
        command = commands.get(phase)
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) for item in command)
        ):
            raise RuntimeError(f"command plan argv is malformed: {phase}")


def _load_context(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    command_plan_path: pathlib.Path,
) -> tuple[dict, dict, dict]:
    manifest = load_execution_json(manifest_path)
    validate_manifest(manifest)
    state, _ = load_state(state_path, recover=False)
    validate_state_binding(state, manifest_path, manifest)
    plan = load_json(command_plan_path)
    _validate_plan(
        plan,
        manifest_path,
        state_path,
        str(state["execution_id"]),
    )
    return manifest, state, plan


def _completed_prefix(state: dict) -> list[str]:
    completed = []
    found_incomplete = False
    for phase in PHASES:
        status = state["phases"][phase]["status"]
        if status == "completed":
            if found_incomplete:
                raise RuntimeError(
                    "operator state contains non-prefix completed phase"
                )
            completed.append(phase)
        else:
            found_incomplete = True
    return completed


def _phase_payload(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    command_plan_path: pathlib.Path,
    state: dict,
    plan: dict,
    phase: str,
    previous_receipt_sha256: str | None,
) -> dict:
    phase_state = state["phases"][phase]
    if phase_state.get("status") != "completed":
        raise RuntimeError(f"phase is not completed: {phase}")
    if phase_state.get("last_exit_code") != 0:
        raise RuntimeError(f"completed phase has nonzero exit code: {phase}")

    command = phase_state.get("last_command")
    expected = plan["phase_commands"][phase]
    if command != expected:
        raise RuntimeError(
            f"completed phase command disagrees with immutable command plan: {phase}"
        )

    attempts = phase_state.get("attempts")
    retries = phase_state.get("interrupted_retries")
    if not isinstance(attempts, int) or attempts <= 0:
        raise RuntimeError(f"completed phase attempts are invalid: {phase}")
    if not isinstance(retries, int) or retries < 0:
        raise RuntimeError(f"completed phase interrupted retries are invalid: {phase}")

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "production-phase-receipt",
        "execution_id": state["execution_id"],
        "phase": phase,
        "phase_index": PHASES.index(phase),
        "execution_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "operator_state_path": str(state_path),
        "command_plan": {
            "path": str(command_plan_path),
            "sha256": sha256_file(command_plan_path),
        },
        "command": {
            "argv": list(command),
            "sha256": canonical_digest(list(command)),
            "shell": False,
        },
        "result": {
            "status": "completed",
            "exit_code": 0,
            "attempts": attempts,
            "interrupted_retries": retries,
        },
        "previous_receipt_sha256": previous_receipt_sha256,
    }


def sync_phase_receipts(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    command_plan_path: pathlib.Path,
    receipt_dir: pathlib.Path,
) -> list[dict]:
    _, state, plan = _load_context(
        manifest_path, state_path, command_plan_path
    )
    completed = _completed_prefix(state)
    receipts = []
    previous_sha256 = None
    for phase in completed:
        path = phase_receipt_path(receipt_dir, phase)
        payload = _phase_payload(
            manifest_path,
            state_path,
            command_plan_path,
            state,
            plan,
            phase,
            previous_sha256,
        )
        write_immutable(path, payload)
        receipt_sha256 = sha256_file(path)
        receipts.append(
            {
                "phase": phase,
                "path": str(path),
                "sha256": receipt_sha256,
            }
        )
        previous_sha256 = receipt_sha256
    return receipts


def _safe_declared_file(
    output_dir: pathlib.Path,
    relative_name: str,
) -> pathlib.Path:
    if not isinstance(relative_name, str) or not relative_name:
        raise RuntimeError("full checkpoint manifest contains invalid file path")
    relative = pathlib.PurePosixPath(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"full checkpoint manifest path escapes package root: {relative_name}"
        )
    root = output_dir.resolve()
    path = (output_dir / pathlib.Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"full checkpoint manifest path escapes package root: {relative_name}"
        ) from exc
    if not path.is_file():
        raise RuntimeError(
            f"full checkpoint declared file is missing: {relative_name}"
        )
    return path


def _file_evidence(path: pathlib.Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _package_evidence(manifest: dict) -> dict:
    source = manifest.get("source")
    target = manifest.get("target")
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise RuntimeError("production execution manifest source/target is malformed")

    output_value = target.get("output_dir")
    journal_value = target.get("dense_journal")
    model = source.get("model")
    snapshot = source.get("snapshot")
    if not all(
        isinstance(item, str) and item
        for item in (output_value, journal_value, model, snapshot)
    ):
        raise RuntimeError(
            "production execution manifest is missing package evidence paths"
        )

    output_dir = pathlib.Path(output_value)
    package_manifest_path = output_dir / "manifest.json"
    if not package_manifest_path.is_file():
        raise RuntimeError("full checkpoint manifest.json is missing")

    package_manifest = load_json(package_manifest_path)
    if package_manifest.get("packageStage") != "full-checkpoint":
        raise RuntimeError("packageStage is not full-checkpoint")
    if package_manifest.get("sourceCheckpoint") != model:
        raise RuntimeError(
            "full checkpoint sourceCheckpoint disagrees with execution manifest"
        )
    if package_manifest.get("sourceSnapshot") != snapshot:
        raise RuntimeError(
            "full checkpoint sourceSnapshot disagrees with execution manifest"
        )

    declared = package_manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise RuntimeError("full checkpoint manifest.files is missing")

    required = {
        "config.json",
        "conversion-provenance.json",
        "packed_experts/layout.json",
        "model.safetensors",
        "full-checkpoint-provenance.json",
    }
    missing = required.difference(declared)
    if missing:
        raise RuntimeError(
            "full checkpoint required file declaration is missing: "
            + ", ".join(sorted(missing))
        )

    files = {}
    total_bytes = 0
    for relative_name in sorted(declared):
        declared_bytes = declared[relative_name]
        if not isinstance(declared_bytes, int) or declared_bytes < 0:
            raise RuntimeError(
                f"full checkpoint declared byte count is invalid: {relative_name}"
            )
        path = _safe_declared_file(output_dir, relative_name)
        evidence = _file_evidence(path)
        if evidence["bytes"] != declared_bytes:
            raise RuntimeError(
                f"full checkpoint declared byte count mismatch: {relative_name}"
            )
        files[relative_name] = {
            "declared_bytes": declared_bytes,
            **evidence,
        }
        total_bytes += declared_bytes

    dense_journal_path = pathlib.Path(journal_value)
    if not dense_journal_path.is_file():
        raise RuntimeError("dense conversion journal is missing")

    package_manifest_evidence = {
        "path": str(package_manifest_path),
        **_file_evidence(package_manifest_path),
    }
    dense_journal_evidence = {
        "path": str(dense_journal_path),
        **_file_evidence(dense_journal_path),
    }

    digest_payload = {
        "package_manifest": package_manifest_evidence,
        "declared_files": files,
        "dense_journal": dense_journal_evidence,
    }
    return {
        "output_dir": str(output_dir),
        "package_stage": "full-checkpoint",
        "package_manifest": package_manifest_evidence,
        "declared_file_count": len(files),
        "declared_file_bytes": total_bytes,
        "declared_files": files,
        "dense_journal": dense_journal_evidence,
        "package_digest_sha256": canonical_digest(digest_payload),
    }


def _completion_payload(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    command_plan_path: pathlib.Path,
    manifest: dict,
    state: dict,
    receipts: list[dict],
) -> dict:
    if state.get("completed") is not True:
        raise RuntimeError("operator execution is not complete")
    if len(receipts) != len(PHASES):
        raise RuntimeError("completion receipt requires every phase receipt")

    expected_phases = list(PHASES)
    if [item["phase"] for item in receipts] != expected_phases:
        raise RuntimeError("phase receipt order is incomplete")

    package = _package_evidence(manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "production-completion-receipt",
        "execution_id": state["execution_id"],
        "execution_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "operator_state": {
            "path": str(state_path),
            "sha256": sha256_file(state_path),
        },
        "command_plan": {
            "path": str(command_plan_path),
            "sha256": sha256_file(command_plan_path),
        },
        "phase_receipts": receipts,
        "phase_chain_tail_sha256": receipts[-1]["sha256"],
        "package": package,
    }


def _validate_existing_completion(
    payload: dict,
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    command_plan_path: pathlib.Path,
    state: dict,
    receipts: list[dict],
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported production completion receipt schema")
    if payload.get("stage") != "production-completion-receipt":
        raise RuntimeError("receipt stage is not production-completion-receipt")
    if payload.get("execution_id") != state.get("execution_id"):
        raise RuntimeError("completion receipt execution_id disagrees with state")

    manifest_evidence = payload.get("execution_manifest")
    state_evidence = payload.get("operator_state")
    plan_evidence = payload.get("command_plan")
    if not all(isinstance(item, dict) for item in (
        manifest_evidence, state_evidence, plan_evidence
    )):
        raise RuntimeError("completion receipt bindings are malformed")

    expected_manifest_sha = sha256_file(manifest_path)
    expected_state_sha = sha256_file(state_path)
    expected_plan_sha = sha256_file(command_plan_path)
    if (
        manifest_evidence.get("path") != str(manifest_path)
        or manifest_evidence.get("sha256") != expected_manifest_sha
    ):
        raise RuntimeError("completion receipt manifest binding failed")
    if (
        state_evidence.get("path") != str(state_path)
        or state_evidence.get("sha256") != expected_state_sha
    ):
        raise RuntimeError("completion receipt state binding failed")
    if (
        plan_evidence.get("path") != str(command_plan_path)
        or plan_evidence.get("sha256") != expected_plan_sha
    ):
        raise RuntimeError("completion receipt command-plan binding failed")

    if payload.get("phase_receipts") != receipts:
        raise RuntimeError("completion receipt phase evidence disagrees")
    expected_tail = receipts[-1]["sha256"] if receipts else None
    if payload.get("phase_chain_tail_sha256") != expected_tail:
        raise RuntimeError("completion receipt phase chain tail disagrees")

    package = payload.get("package")
    if (
        not isinstance(package, dict)
        or not isinstance(package.get("package_digest_sha256"), str)
        or len(package["package_digest_sha256"]) != 64
    ):
        raise RuntimeError("completion receipt package digest is malformed")

    package_manifest = package.get("package_manifest")
    declared_files = package.get("declared_files")
    dense_journal = package.get("dense_journal")
    if (
        not isinstance(package_manifest, dict)
        or not isinstance(declared_files, dict)
        or not isinstance(dense_journal, dict)
    ):
        raise RuntimeError("completion receipt package evidence is malformed")
    expected_package_digest = canonical_digest(
        {
            "package_manifest": package_manifest,
            "declared_files": declared_files,
            "dense_journal": dense_journal,
        }
    )
    if package["package_digest_sha256"] != expected_package_digest:
        raise RuntimeError("completion receipt package digest binding failed")
    if package.get("declared_file_count") != len(declared_files):
        raise RuntimeError("completion receipt declared file count disagrees")
    declared_bytes = 0
    for relative_name, evidence in declared_files.items():
        if not isinstance(relative_name, str) or not isinstance(evidence, dict):
            raise RuntimeError("completion receipt declared file evidence is malformed")
        value = evidence.get("declared_bytes")
        if type(value) is not int or value < 0:
            raise RuntimeError("completion receipt declared byte evidence is malformed")
        declared_bytes += value
    if package.get("declared_file_bytes") != declared_bytes:
        raise RuntimeError("completion receipt declared byte total disagrees")


def sync_production_audit(
    manifest_path: pathlib.Path,
    state_path: pathlib.Path,
    command_plan_path: pathlib.Path,
    receipt_dir: pathlib.Path | None = None,
    *,
    verify_existing_completion: bool = False,
) -> dict:
    receipt_dir = receipt_dir or default_receipt_dir(state_path)
    manifest, state, _ = _load_context(
        manifest_path, state_path, command_plan_path
    )
    receipts = sync_phase_receipts(
        manifest_path,
        state_path,
        command_plan_path,
        receipt_dir,
    )

    completion_path = completion_receipt_path(receipt_dir)
    completion_sha256 = None
    package_digest = None

    if state.get("completed") is True:
        if len(receipts) != len(PHASES):
            raise RuntimeError("completed state is missing phase receipts")

        if completion_path.exists() and not verify_existing_completion:
            completion = load_json(completion_path)
            _validate_existing_completion(
                completion,
                manifest_path,
                state_path,
                command_plan_path,
                state,
                receipts,
            )
        else:
            completion = _completion_payload(
                manifest_path,
                state_path,
                command_plan_path,
                manifest,
                state,
                receipts,
            )
            write_immutable(completion_path, completion)

        completion_sha256 = sha256_file(completion_path)
        package = completion.get("package")
        if isinstance(package, dict):
            package_digest = package.get("package_digest_sha256")

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "production-audit-status",
        "execution_id": state["execution_id"],
        "completed": bool(state.get("completed")),
        "receipt_dir": str(receipt_dir),
        "phase_receipts": receipts,
        "completion_receipt": (
            None
            if completion_sha256 is None
            else {
                "path": str(completion_path),
                "sha256": completion_sha256,
            }
        ),
        "package_digest_sha256": package_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--command-plan", required=True)
    parser.add_argument("--receipt-dir")
    parser.add_argument(
        "--verify-package",
        action="store_true",
        help="rehash the completed package even when completion receipt exists",
    )
    args = parser.parse_args()

    result = sync_production_audit(
        pathlib.Path(args.manifest),
        pathlib.Path(args.state),
        pathlib.Path(args.command_plan),
        pathlib.Path(args.receipt_dir) if args.receipt_dir else None,
        verify_existing_completion=args.verify_package,
    )
    print(canonical_text(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
