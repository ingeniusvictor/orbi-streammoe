#!/usr/bin/env python3
import copy
import json
import pathlib
import tempfile

from authorize_qwen_windows_conversion import build_authorization, canonical_digest
from build_qwen_production_commands import build_phase_commands
from guard_qwen_windows_start import guarded_start, verify_guarded_start
from test_qwen_production_launcher import fake_phase_executor, manifest_fixture


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
            "summary_sha256": "b" * 64,
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
    command_path = root / "operator-state.json.commands.json"
    plan = build_phase_commands(
        manifest_path,
        state,
        root / "bin",
        root / "scripts",
        "python-fixture",
    )
    write_json(command_path, plan)

    bundle = {
        "schema_version": 1,
        "stage": "official-production-rehearsal",
        "source": {"model": OFFICIAL_MODEL, "snapshot": OFFICIAL_SNAPSHOT},
        "production_authorization": {
            "execution_id": manifest["execution_id"],
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "command_plan": str(command_path),
            "command_plan_sha256": sha256_file(command_path),
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

    original_probe = fake_probe(manifest, root)
    original_probe_path = root / "authorized-host-probe.json"
    write_json(original_probe_path, original_probe)
    authorization = build_authorization(bundle_path, original_probe_path, 4096)
    authorization_path = root / "conversion-authorization.json"
    write_json(authorization_path, authorization)

    current_probe = copy.deepcopy(original_probe)
    current_probe["memory"]["available_physical_bytes"] -= 1024
    for item in current_probe["paths"].values():
        item["free_bytes"] -= 1024
    current_probe_path = root / "current-host-probe.json"
    write_json(current_probe_path, current_probe)
    return manifest_path, state, command_path, authorization_path, current_probe_path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-streammoe-osm42c-") as temp:
        root = pathlib.Path(temp)
        manifest, state, command_plan, authorization, current_probe = make_fixture(root)

        verified = verify_guarded_start(authorization, current_probe, state)
        if verified["manifest"] != manifest or verified["command_plan"] != command_plan:
            raise RuntimeError("guarded start did not recover exact bound production plan")
        if state.exists():
            raise RuntimeError("verification initialized operator state")

        started = guarded_start(
            authorization,
            current_probe,
            state,
            root / "bin",
            root / "scripts",
            "python-fixture",
            phase_executor=fake_phase_executor,
        )
        if started["phase"] != "expert_conversion":
            raise RuntimeError("guarded start did not execute first authorized phase")
        if started["next_phase"] != "expert_finalization":
            raise RuntimeError("guarded start did not stop at first checkpoint")
        if not state.exists():
            raise RuntimeError("guarded start did not initialize durable state")

        expect_failure(
            lambda: verify_guarded_start(authorization, current_probe, state),
            "second first-start attempt",
        )

        root2 = root / "tamper"
        _, state2, _, authorization2, current_probe2 = make_fixture(root2)
        changed_probe = json.loads(current_probe2.read_text(encoding="utf-8"))
        changed_probe["vulkan"]["summary_sha256"] = "c" * 64
        write_json(current_probe2, changed_probe)
        expect_failure(
            lambda: verify_guarded_start(authorization2, current_probe2, state2),
            "Vulkan binding change",
        )

        root3 = root / "low-memory"
        _, state3, _, authorization3, current_probe3 = make_fixture(root3)
        low = json.loads(current_probe3.read_text(encoding="utf-8"))
        low["memory"]["available_physical_bytes"] = 1
        write_json(current_probe3, low)
        expect_failure(
            lambda: verify_guarded_start(authorization3, current_probe3, state3),
            "fresh RAM gate",
        )

        root4 = root / "manifest-tamper"
        manifest4, state4, _, authorization4, current_probe4 = make_fixture(root4)
        manifest4.write_text(manifest4.read_text(encoding="utf-8") + " ", encoding="utf-8")
        expect_failure(
            lambda: verify_guarded_start(authorization4, current_probe4, state4),
            "manifest hash tamper",
        )

        print(
            "OSM-42C guarded Windows start: PASS\n"
            "  authorization_digest_binding=PASS\n"
            "  rehearsal_manifest_command_binding=PASS\n"
            "  fresh_ram_disk_gate=PASS\n"
            "  stable_machine_vulkan_binding=PASS\n"
            "  verify_only_no_state_mutation=PASS\n"
            "  first_phase_checkpoint_only=PASS\n"
            "  durable_state_initialization=PASS\n"
            "  repeated_first_start_rejected=PASS\n"
            "  tamper_rejection=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
