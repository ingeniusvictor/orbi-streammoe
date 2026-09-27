#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pathlib
import platform


SCHEMA_VERSION = 1
OFFICIAL_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"


def canonical_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def canonical_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = canonical_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing authorization pack conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def resolve_bound_path(bundle_path: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    return (bundle_path.parent / path).resolve(strict=False)


def verify_rehearsal(bundle_path: pathlib.Path) -> tuple[dict, dict]:
    bundle = load_json(bundle_path)
    if bundle.get("schema_version") != 1:
        raise RuntimeError("unsupported rehearsal bundle schema")
    if bundle.get("stage") != "official-production-rehearsal":
        raise RuntimeError("not an OSM-42A rehearsal bundle")

    source = bundle.get("source", {})
    if source.get("model") != OFFICIAL_MODEL:
        raise RuntimeError("rehearsal model is not the pinned official model")
    if source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("rehearsal snapshot is not the pinned official snapshot")

    safety = bundle.get("safety", {})
    if (
        safety.get("operator_state_created")
        or safety.get("receipt_dir_created")
        or safety.get("full_checkpoint_conversion_executed")
        or not safety.get("bounded_source_evidence_only")
    ):
        raise RuntimeError("rehearsal safety boundary is invalid")

    expected_bundle_digest = bundle.get("bundle_sha256")
    unsigned = dict(bundle)
    unsigned.pop("bundle_sha256", None)
    if expected_bundle_digest != canonical_digest(unsigned):
        raise RuntimeError("rehearsal bundle digest mismatch")

    authorization = bundle.get("production_authorization", {})
    manifest_path = resolve_bound_path(
        bundle_path, authorization.get("manifest", "")
    )
    command_plan_path = resolve_bound_path(
        bundle_path, authorization.get("command_plan", "")
    )
    if not manifest_path.is_file() or not command_plan_path.is_file():
        raise RuntimeError("rehearsal bound manifest/command plan is missing")
    if sha256_file(manifest_path) != authorization.get("manifest_sha256"):
        raise RuntimeError("rehearsal execution manifest hash mismatch")
    if sha256_file(command_plan_path) != authorization.get("command_plan_sha256"):
        raise RuntimeError("rehearsal command plan hash mismatch")

    execution = load_json(manifest_path)
    if not execution.get("authorization", {}).get("authorized"):
        raise RuntimeError("OSM-41A execution manifest is not authorized")
    if not execution.get("authorization", {}).get("full_source_scope"):
        raise RuntimeError("OSM-41A execution manifest is not full-source")
    return bundle, execution


def verify_probe(probe: dict, execution: dict, min_available_memory_bytes: int) -> dict:
    if probe.get("schema_version") != 1:
        raise RuntimeError("unsupported host probe schema")
    if probe.get("stage") != "windows-full-conversion-host-probe":
        raise RuntimeError("not an OSM-42B host probe")
    if probe.get("platform", {}).get("system") != "Windows":
        raise RuntimeError("authorization requires Windows host evidence")

    memory = probe.get("memory", {})
    available_memory = int(memory.get("available_physical_bytes", 0))
    batch_bound = int(execution["disk"]["max_batch_source_bytes"])
    required_memory = max(int(min_available_memory_bytes), batch_bound)
    if required_memory <= 0:
        raise RuntimeError("required available memory must be positive")
    if available_memory < required_memory:
        raise RuntimeError(
            f"insufficient available RAM: {available_memory} < {required_memory}"
        )

    expected_paths = {
        "target_output": pathlib.Path(execution["target"]["output_dir"]),
        "expert_work": pathlib.Path(execution["target"]["expert_work_dir"]),
        "dense_work": pathlib.Path(execution["target"]["dense_work_dir"]),
    }
    probes = probe.get("paths", {})
    required_disk = int(execution["disk"]["peak_required_free_bytes"])
    if required_disk <= 0:
        raise RuntimeError("execution manifest disk requirement must be positive")

    volumes = {}
    path_checks = {}
    for key, expected in expected_paths.items():
        item = probes.get(key)
        if not isinstance(item, dict):
            raise RuntimeError(f"host probe missing path: {key}")
        observed = pathlib.Path(item.get("path", ""))
        if os.path.normcase(str(observed)) != os.path.normcase(str(expected)):
            raise RuntimeError(f"host probe path mismatch for {key}")
        if item.get("exists") and item.get("is_directory") is not True:
            raise RuntimeError(f"production path is not a directory: {key}")
        if not item.get("writable_ancestor"):
            raise RuntimeError(f"production path is not writable: {key}")
        free = int(item.get("free_bytes", 0))
        volume = item.get("volume_id")
        if not isinstance(volume, str) or not volume:
            raise RuntimeError(f"host probe missing volume ID: {key}")
        volumes[volume] = min(free, volumes.get(volume, free))
        path_checks[key] = {
            "path": str(expected),
            "volume_id": volume,
            "free_bytes": free,
            "exists": bool(item.get("exists")),
        }

    insufficient = {
        volume: free
        for volume, free in volumes.items()
        if free < required_disk
    }
    if insufficient:
        raise RuntimeError(
            f"insufficient free disk for peak production requirement: {insufficient}"
        )

    vulkan = probe.get("vulkan", {})
    if not vulkan.get("available"):
        raise RuntimeError("Vulkan adapter evidence is required")
    digest = vulkan.get("summary_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("Vulkan summary SHA-256 is missing")
    markers = vulkan.get("adapter_markers")
    if not isinstance(markers, list) or not markers:
        raise RuntimeError("Vulkan adapter markers are missing")

    return {
        "required_available_memory_bytes": required_memory,
        "available_physical_memory_bytes": available_memory,
        "required_peak_free_disk_bytes": required_disk,
        "volumes": volumes,
        "paths": path_checks,
        "vulkan_summary_sha256": digest,
        "vulkan_adapter_markers": markers,
    }


def build_authorization(
    rehearsal_bundle_path: pathlib.Path,
    host_probe_path: pathlib.Path,
    min_available_memory_bytes: int,
) -> dict:
    bundle, execution = verify_rehearsal(rehearsal_bundle_path)
    probe = load_json(host_probe_path)
    checks = verify_probe(probe, execution, min_available_memory_bytes)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "windows-full-conversion-authorization",
        "authorized": True,
        "source": {
            "model": OFFICIAL_MODEL,
            "snapshot": OFFICIAL_SNAPSHOT,
            "execution_id": execution["execution_id"],
        },
        "bindings": {
            "rehearsal_bundle": str(rehearsal_bundle_path),
            "rehearsal_bundle_sha256": sha256_file(rehearsal_bundle_path),
            "host_probe": str(host_probe_path),
            "host_probe_sha256": sha256_file(host_probe_path),
            "execution_manifest_sha256": bundle["production_authorization"][
                "manifest_sha256"
            ],
            "command_plan_sha256": bundle["production_authorization"][
                "command_plan_sha256"
            ],
        },
        "host": {
            "platform": probe["platform"],
            "memory": probe["memory"],
            "paths": probe["paths"],
            "vulkan": probe["vulkan"],
        },
        "checks": checks,
        "safety": {
            "conversion_started": False,
            "operator_state_initialized": False,
            "authorization_is_machine_specific": True,
            "full_inference_claimed": False,
        },
    }
    payload["authorization_id"] = canonical_digest(payload)[:24]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rehearsal-bundle", required=True)
    parser.add_argument("--host-probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-available-memory-bytes", required=True, type=int)
    args = parser.parse_args()

    if args.min_available_memory_bytes <= 0:
        raise RuntimeError("--min-available-memory-bytes must be positive")

    payload = build_authorization(
        pathlib.Path(args.rehearsal_bundle),
        pathlib.Path(args.host_probe),
        args.min_available_memory_bytes,
    )
    output = pathlib.Path(args.output)
    write_immutable(output, payload)
    print(canonical_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
