#!/usr/bin/env python3
import copy
import hashlib
import json
import pathlib
import tempfile

from authorize_qwen_windows_conversion import (
    build_authorization,
    canonical_digest,
    write_immutable,
)
from probe_qwen_windows_host import build_probe


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")


def make_rehearsal(root: pathlib.Path) -> tuple[pathlib.Path, dict]:
    output = root / "production-output"
    expert = root / "expert-work"
    dense = root / "dense-work"

    manifest = {
        "schema_version": 1,
        "stage": "production-execution-preflight",
        "execution_id": "0123456789abcdefghij",
        "source": {
            "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "snapshot": "f5e99a3698d364cf77584543481b778afee26177",
        },
        "target": {
            "output_dir": str(output),
            "expert_work_dir": str(expert),
            "dense_work_dir": str(dense),
            "dense_output": str(output / "model.safetensors"),
            "dense_journal": str(output / "model.progress.json"),
        },
        "disk": {
            "max_batch_source_bytes": 1024,
            "peak_required_free_bytes": 4096,
        },
        "authorization": {
            "authorized": True,
            "full_source_scope": True,
            "immutable_preflight": True,
        },
    }
    manifest_path = root / "execution.json"
    write_json(manifest_path, manifest)

    command_plan = {
        "schema_version": 1,
        "execution_id": manifest["execution_id"],
        "phase_order": [
            "expert_conversion",
            "expert_finalization",
            "dense_conversion",
            "full_checkpoint",
        ],
    }
    command_path = root / "operator-state.json.commands.json"
    write_json(command_path, command_plan)

    bundle = {
        "schema_version": 1,
        "stage": "official-production-rehearsal",
        "source": {
            "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "snapshot": "f5e99a3698d364cf77584543481b778afee26177",
        },
        "production_authorization": {
            "execution_id": manifest["execution_id"],
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "command_plan": str(command_path),
            "command_plan_sha256": sha256_file(command_path),
            "authorized": True,
            "full_source_scope": True,
            "phase_order": command_plan["phase_order"],
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
    return bundle_path, manifest


def fake_windows_probe(manifest: dict, root: pathlib.Path) -> dict:
    paths = {}
    for key, target_key in (
        ("target_output", "output_dir"),
        ("expert_work", "expert_work_dir"),
        ("dense_work", "dense_work_dir"),
    ):
        path = pathlib.Path(manifest["target"][target_key])
        paths[key] = {
            "path": str(path),
            "exists": False,
            "is_directory": None,
            "nearest_existing_ancestor": str(root),
            "volume_id": "C:",
            "free_bytes": 1024 * 1024,
            "total_bytes": 2 * 1024 * 1024,
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
            "total_physical_bytes": 1024 * 1024,
            "available_physical_bytes": 512 * 1024,
            "memory_load_percent": 50,
        },
        "paths": paths,
        "vulkan": {
            "available": True,
            "source": "fixture-vulkaninfo",
            "summary_sha256": "a" * 64,
            "summary_bytes": 128,
            "adapter_markers": ["GPU0: fixture", "deviceName = fixture"],
        },
    }


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="orbi-streammoe-osm42b-"
    ) as temp:
        root = pathlib.Path(temp)
        bundle_path, manifest = make_rehearsal(root)
        probe = fake_windows_probe(manifest, root)
        probe_path = root / "host-probe.json"
        write_json(probe_path, probe)

        authorization = build_authorization(
            bundle_path,
            probe_path,
            min_available_memory_bytes=4096,
        )
        if not authorization["authorized"]:
            raise RuntimeError("authorization pack was not authorized")
        if authorization["source"]["execution_id"] != manifest["execution_id"]:
            raise RuntimeError("execution binding mismatch")
        if authorization["checks"]["required_peak_free_disk_bytes"] != 4096:
            raise RuntimeError("disk requirement mismatch")
        if authorization["checks"]["required_available_memory_bytes"] != 4096:
            raise RuntimeError("RAM policy mismatch")
        if not authorization["safety"]["authorization_is_machine_specific"]:
            raise RuntimeError("machine-specific safety marker missing")
        if authorization["safety"]["conversion_started"]:
            raise RuntimeError("preflight incorrectly claims conversion started")

        output = root / "authorization.json"
        write_immutable(output, authorization)
        write_immutable(output, authorization)
        before = output.read_bytes()
        changed = copy.deepcopy(authorization)
        changed["authorized"] = False
        expect_failure(
            lambda: write_immutable(output, changed),
            "immutable authorization conflict",
        )
        if output.read_bytes() != before:
            raise RuntimeError("immutable authorization changed after conflict")

        low_memory = copy.deepcopy(probe)
        low_memory["memory"]["available_physical_bytes"] = 100
        low_memory_path = root / "low-memory.json"
        write_json(low_memory_path, low_memory)
        expect_failure(
            lambda: build_authorization(
                bundle_path, low_memory_path, 4096
            ),
            "insufficient RAM",
        )

        low_disk = copy.deepcopy(probe)
        low_disk["paths"]["target_output"]["free_bytes"] = 100
        low_disk_path = root / "low-disk.json"
        write_json(low_disk_path, low_disk)
        expect_failure(
            lambda: build_authorization(
                bundle_path, low_disk_path, 4096
            ),
            "insufficient disk",
        )

        no_vulkan = copy.deepcopy(probe)
        no_vulkan["vulkan"]["available"] = False
        no_vulkan_path = root / "no-vulkan.json"
        write_json(no_vulkan_path, no_vulkan)
        expect_failure(
            lambda: build_authorization(
                bundle_path, no_vulkan_path, 4096
            ),
            "missing Vulkan",
        )

        wrong_os = copy.deepcopy(probe)
        wrong_os["platform"]["system"] = "Linux"
        wrong_os_path = root / "wrong-os.json"
        write_json(wrong_os_path, wrong_os)
        expect_failure(
            lambda: build_authorization(
                bundle_path, wrong_os_path, 4096
            ),
            "non-Windows authorization",
        )

        command_path = pathlib.Path(
            json.loads(bundle_path.read_text(encoding="utf-8"))[
                "production_authorization"
            ]["command_plan"]
        )
        command_before = command_path.read_bytes()
        command_path.write_bytes(command_before + b" ")
        expect_failure(
            lambda: build_authorization(bundle_path, probe_path, 4096),
            "command-plan tamper",
        )
        command_path.write_bytes(command_before)

        vulkan_summary = root / "vulkan-summary.txt"
        vulkan_summary.write_text(
            "Vulkan Instance Version: 1.3\n"
            "GPU0:\n"
            "    deviceName = Fixture Vulkan Adapter\n",
            encoding="utf-8",
        )
        portable_probe = build_probe(
            root / "probe-output",
            root / "probe-expert",
            root / "probe-dense",
            vulkan_summary,
            allow_non_windows=True,
            allow_missing_vulkan=False,
        )
        if not portable_probe["vulkan"]["available"]:
            raise RuntimeError("host probe did not bind Vulkan summary")
        if not portable_probe["vulkan"]["adapter_markers"]:
            raise RuntimeError("host probe did not preserve adapter markers")
        if portable_probe["memory"]["available_physical_bytes"] <= 0:
            raise RuntimeError("host probe did not record available memory")
        if not all(
            item["writable_ancestor"]
            for item in portable_probe["paths"].values()
        ):
            raise RuntimeError("host probe path writability failed")

        print(
            "OSM-42B Windows hardware preflight: PASS\n"
            "  rehearsal_digest_binding=PASS\n"
            "  execution_command_binding=PASS\n"
            "  machine_specific_windows_guard=PASS\n"
            "  available_ram_gate=PASS\n"
            "  peak_free_disk_gate=PASS\n"
            "  writable_resume_paths=PASS\n"
            "  vulkan_adapter_evidence=PASS\n"
            "  immutable_authorization_pack=PASS\n"
            "  tamper_rejection=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
