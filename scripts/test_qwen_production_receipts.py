#!/usr/bin/env python3
import hashlib
import json
import pathlib
import tempfile

from launch_qwen_production import (
    audit_execution,
    default_command_plan,
    execute_all,
    execute_next,
)
from record_qwen_production_receipts import (
    canonical_digest,
    completion_receipt_path,
    default_receipt_dir,
    phase_receipt_path,
)
from run_qwen_production_operator import (
    ensure_phase_can_run,
    load_state,
    phase_report,
    write_state,
)


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_bytes(path: pathlib.Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def manifest_fixture(root: pathlib.Path) -> pathlib.Path:
    path = root / "execution.json"
    payload = {
        "schema_version": 1,
        "stage": "production-execution-preflight",
        "execution_id": "receiptfixture000001",
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


def write_fake_full_package(manifest_path: pathlib.Path) -> None:
    execution = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = pathlib.Path(execution["target"]["output_dir"])
    dense_journal = pathlib.Path(execution["target"]["dense_journal"])

    file_payloads = {
        "config.json": b'{"model_type":"qwen3_next"}\n',
        "conversion-provenance.json": b'{"stage":"expert-shell"}\n',
        "packed_experts/layout.json": b'{"layerCount":1}\n',
        "packed_experts/layer_00.bin": b"expert-payload-fixture",
        "model.safetensors": b"dense-payload-fixture",
        "full-checkpoint-provenance.json": b'{"stage":"full-checkpoint"}\n',
    }
    for relative, payload in file_payloads.items():
        write_bytes(output / pathlib.Path(relative), payload)
    write_json(
        dense_journal,
        {
            "schema_version": 1,
            "complete": True,
            "fixture": "OSM-41F",
        },
    )

    files = {
        relative: len(payload)
        for relative, payload in file_payloads.items()
    }
    write_json(
        output / "manifest.json",
        {
            "magic": "QPACK",
            "version": 1,
            "modelName": "qwen3_next",
            "sourceCheckpoint": execution["source"]["model"],
            "sourceSnapshot": execution["source"]["snapshot"],
            "quantBits": 4,
            "quantGroupSize": 64,
            "packageStage": "full-checkpoint",
            "files": files,
        },
    )


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
    if phase == "full_checkpoint":
        write_fake_full_package(manifest_path)

    phase_state = state["phases"][phase]
    if phase_state["status"] == "running":
        phase_state["interrupted_retries"] += 1
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


def verify_complete_receipt_chain(root: pathlib.Path) -> None:
    manifest = manifest_fixture(root)
    state = root / "operator-state.json"
    bin_dir = root / "bin"
    scripts_dir = root / "scripts"

    result = execute_all(
        manifest,
        state,
        bin_dir,
        scripts_dir,
        "python-fixture",
        phase_executor=fake_phase_executor,
    )
    if not result["completed"]:
        raise RuntimeError("fixture execution did not complete")

    plan = default_command_plan(state)
    receipts = default_receipt_dir(state)
    previous = None
    phase_evidence = []
    for index, phase in enumerate((
        "expert_conversion",
        "expert_finalization",
        "dense_conversion",
        "full_checkpoint",
    )):
        path = phase_receipt_path(receipts, phase)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["phase_index"] != index:
            raise RuntimeError("phase receipt index mismatch")
        if payload["previous_receipt_sha256"] != previous:
            raise RuntimeError("phase receipt chain mismatch")
        if payload["command"]["sha256"] != canonical_digest(
            payload["command"]["argv"]
        ):
            raise RuntimeError("phase receipt command hash mismatch")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        phase_evidence.append(
            {"phase": phase, "path": str(path), "sha256": digest}
        )
        previous = digest

    completion_path = completion_receipt_path(receipts)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion["phase_receipts"] != phase_evidence:
        raise RuntimeError("completion phase receipt inventory mismatch")
    if completion["phase_chain_tail_sha256"] != previous:
        raise RuntimeError("completion phase chain tail mismatch")
    package = completion["package"]
    if package["package_stage"] != "full-checkpoint":
        raise RuntimeError("completion package stage mismatch")
    if len(package["declared_files"]) != 6:
        raise RuntimeError("completion declared file count mismatch")
    if package["declared_file_count"] != 6:
        raise RuntimeError("completion declared file count field mismatch")
    if len(package["package_digest_sha256"]) != 64:
        raise RuntimeError("completion package digest missing")

    before = completion_path.read_bytes()
    no_op = execute_next(
        manifest,
        state,
        bin_dir,
        scripts_dir,
        "python-fixture",
        phase_executor=fake_phase_executor,
    )
    if no_op["stop_reason"] != "already_complete":
        raise RuntimeError("completed execution did not remain idempotent")
    if completion_path.read_bytes() != before:
        raise RuntimeError("idempotent resume changed completion receipt")

    verified = audit_execution(manifest, state)
    if not verified["package_reverified"]:
        raise RuntimeError("explicit audit did not deep-verify package")

    dense = pathlib.Path(
        json.loads(manifest.read_text(encoding="utf-8"))["target"]["dense_output"]
    )
    original = dense.read_bytes()
    dense.write_bytes(bytes([original[0] ^ 0x01]) + original[1:])
    expect_failure(
        lambda: audit_execution(manifest, state),
        "same-size package mutation",
    )
    dense.write_bytes(original)
    audit_execution(manifest, state)

    first_receipt = phase_receipt_path(receipts, "expert_conversion")
    original_receipt = first_receipt.read_bytes()
    tampered = json.loads(original_receipt)
    tampered["result"]["attempts"] += 1
    write_json(first_receipt, tampered)
    expect_failure(
        lambda: execute_next(
            manifest,
            state,
            bin_dir,
            scripts_dir,
            "python-fixture",
            phase_executor=fake_phase_executor,
        ),
        "phase receipt mutation",
    )
    first_receipt.write_bytes(original_receipt)
    audit_execution(manifest, state)

    if not plan.exists():
        raise RuntimeError("immutable command plan is missing")


def verify_missing_receipt_reconciliation(root: pathlib.Path) -> None:
    manifest = manifest_fixture(root)
    state = root / "operator-state.json"
    args = (
        manifest,
        state,
        root / "bin",
        root / "scripts",
        "python-fixture",
    )
    first = execute_next(*args, phase_executor=fake_phase_executor)
    if first["next_phase"] != "expert_finalization":
        raise RuntimeError("first fixture phase did not advance")

    receipt = phase_receipt_path(
        default_receipt_dir(state),
        "expert_conversion",
    )
    receipt.unlink()
    if receipt.exists():
        raise RuntimeError("fixture receipt removal failed")

    second = execute_next(*args, phase_executor=fake_phase_executor)
    if second["phase"] != "expert_finalization":
        raise RuntimeError("resume did not choose second phase")
    if not receipt.exists():
        raise RuntimeError("resume did not reconcile missing completed receipt")


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="orbi-streammoe-osm41f-"
    ) as temp:
        root = pathlib.Path(temp)
        verify_complete_receipt_chain(root / "complete")
        verify_missing_receipt_reconciliation(root / "reconcile")
        print(
            "OSM-41F production execution receipts: PASS\n"
            "  immutable_phase_receipts=PASS\n"
            "  chained_receipt_hashes=PASS\n"
            "  exact_argv_hash_binding=PASS\n"
            "  final_package_stream_hashes=PASS\n"
            "  completed_resume_no_rehash=PASS\n"
            "  explicit_deep_audit=PASS\n"
            "  same_size_tamper_detection=PASS\n"
            "  receipt_tamper_detection=PASS\n"
            "  missing_receipt_reconciliation=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
