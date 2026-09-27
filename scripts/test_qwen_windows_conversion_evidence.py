#!/usr/bin/env python3
import json
import pathlib
import tempfile

from authorize_qwen_windows_conversion import build_authorization, canonical_digest
from build_qwen_production_commands import build_phase_commands
from certify_qwen_windows_conversion import certify_windows_conversion, write_immutable
from launch_qwen_production import execute_all
from test_qwen_production_receipts import fake_phase_executor, manifest_fixture


OFFICIAL_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")


def fake_probe(manifest: dict, root: pathlib.Path) -> dict:
    paths = {}
    for key, target_key in (
        ("target_output", "output_dir"),
        ("expert_work", "expert_work_dir"),
        ("dense_work", "dense_work_dir"),
    ):
        paths[key] = {
            "path": manifest["target"][target_key],
            "exists": False,
            "is_directory": None,
            "nearest_existing_ancestor": str(root),
            "volume_id": "C:",
            "free_bytes": 1024 * 1024 * 1024,
            "total_bytes": 2 * 1024 * 1024 * 1024,
            "writable_ancestor": True,
        }
    return {
        "schema_version": 1,
        "stage": "windows-full-conversion-host-probe",
        "platform": {
            "system": "Windows",
            "release": "fixture",
            "version": "fixture",
            "machine": "AMD64",
            "python": "fixture",
            "windows_required": True,
        },
        "memory": {
            "total_physical_bytes": 1024 * 1024 * 1024,
            "available_physical_bytes": 768 * 1024 * 1024,
            "memory_load_percent": 25,
        },
        "paths": paths,
        "vulkan": {
            "available": True,
            "source": "fixture-vulkaninfo",
            "summary_sha256": "d" * 64,
            "summary_bytes": 128,
            "adapter_markers": ["GPU0: fixture", "deviceName = fixture"],
        },
    }


def make_fixture(root: pathlib.Path):
    manifest_path = manifest_fixture(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["model"] = OFFICIAL_MODEL
    manifest["source"]["snapshot"] = OFFICIAL_SNAPSHOT
    manifest["disk"]["peak_required_free_bytes"] = 4096
    write_json(manifest_path, manifest)

    state = root / "operator-state.json"
    command_plan = root / "operator-state.json.commands.json"
    plan = build_phase_commands(
        manifest_path,
        state,
        root / "bin",
        root / "scripts",
        "python-fixture",
    )
    write_json(command_plan, plan)

    bundle = {
        "schema_version": 1,
        "stage": "official-production-rehearsal",
        "source": {"model": OFFICIAL_MODEL, "snapshot": OFFICIAL_SNAPSHOT},
        "production_authorization": {
            "execution_id": manifest["execution_id"],
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "command_plan": str(command_plan),
            "command_plan_sha256": sha256_file(command_plan),
            "authorized": True,
            "full_source_scope": True,
            "phase_order": plan["phase_order"],
        },
        "dry_run": {
            "action": "preview",
            "state_initialized": False,
            "mutated_state": False,
            "next_phase": "expert_conversion",
        },
        "bounded_official_source_evidence": {
            "expert_slice": {"total_bytes": 10},
            "dense_slice": {"total_bytes": 10},
        },
        "safety": {
            "operator_state_created": False,
            "receipt_dir_created": False,
            "full_checkpoint_conversion_executed": False,
            "bounded_source_evidence_only": True,
        },
    }
    bundle["bundle_sha256"] = canonical_digest(bundle)
    bundle_path = root / "rehearsal-bundle.json"
    write_json(bundle_path, bundle)

    host_probe = root / "authorized-host-probe.json"
    write_json(host_probe, fake_probe(manifest, root))
    authorization = build_authorization(bundle_path, host_probe, 4096)
    authorization_path = root / "authorization.json"
    write_json(authorization_path, authorization)
    return manifest_path, state, command_plan, authorization_path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-streammoe-osm42d-") as temp:
        root = pathlib.Path(temp)
        manifest, state, command_plan, authorization = make_fixture(root)

        expect_failure(
            lambda: certify_windows_conversion(
                authorization, manifest, state, command_plan
            ),
            "incomplete execution",
        )

        completed = execute_all(
            manifest,
            state,
            root / "bin",
            root / "scripts",
            "python-fixture",
            command_plan,
            phase_executor=fake_phase_executor,
        )
        if not completed["completed"]:
            raise RuntimeError("fixture production execution did not complete")

        evidence = certify_windows_conversion(
            authorization,
            manifest,
            state,
            command_plan,
        )
        if not evidence["claims"]["full_official_checkpoint_conversion_completed"]:
            raise RuntimeError("full conversion claim missing")
        if not evidence["claims"]["full_checkpoint_package_deep_audited"]:
            raise RuntimeError("deep audit claim missing")
        if evidence["claims"]["runtime_inference_validated"]:
            raise RuntimeError("OSM-42D must not claim runtime inference")
        if len(evidence["package"]["package_digest_sha256"]) != 64:
            raise RuntimeError("package digest missing")
        if set(evidence["phase_summary"]) != {
            "expert_conversion",
            "expert_finalization",
            "dense_conversion",
            "full_checkpoint",
        }:
            raise RuntimeError("phase summary incomplete")

        output = root / "windows-conversion-certification.json"
        write_immutable(output, evidence)
        write_immutable(output, evidence)

        changed = json.loads(output.read_text(encoding="utf-8"))
        changed["claims"]["runtime_inference_validated"] = True
        expect_failure(
            lambda: write_immutable(output, changed),
            "immutable certification conflict",
        )

        manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
        expect_failure(
            lambda: certify_windows_conversion(
                authorization, manifest, state, command_plan
            ),
            "manifest tamper",
        )

        print(
            "OSM-42D Windows full-conversion evidence: PASS\n"
            "  incomplete_execution_rejected=PASS\n"
            "  authorization_binding=PASS\n"
            "  four_phase_completion_required=PASS\n"
            "  deep_package_audit_required=PASS\n"
            "  completion_receipt_binding=PASS\n"
            "  immutable_certification=PASS\n"
            "  inference_claim_boundary=PASS\n"
            "  tamper_rejection=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
