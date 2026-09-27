#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

OFFICIAL_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_conversion_certification(
    certification_path: pathlib.Path,
    checkpoint_dir: pathlib.Path,
) -> dict:
    cert = load_json(certification_path)
    if cert.get("schema_version") != 1:
        raise RuntimeError("unsupported OSM-42D certification schema")
    if cert.get("stage") != "windows-full-conversion-certification":
        raise RuntimeError("not an OSM-42D Windows conversion certification")

    source = cert.get("source", {})
    if source.get("model") != OFFICIAL_MODEL:
        raise RuntimeError("OSM-42D certification model pin mismatch")
    if source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("OSM-42D certification snapshot pin mismatch")

    claims = cert.get("claims", {})
    if claims.get("full_official_checkpoint_conversion_completed") is not True:
        raise RuntimeError("full official checkpoint conversion is not certified")
    if claims.get("full_checkpoint_package_deep_audited") is not True:
        raise RuntimeError("full checkpoint package was not deep-audited")
    if claims.get("runtime_inference_validated") is not False:
        raise RuntimeError("OSM-42D claim boundary changed unexpectedly")

    package = cert.get("package", {})
    output_dir = package.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise RuntimeError("OSM-42D certification is missing package output_dir")

    expected = pathlib.Path(output_dir).resolve(strict=False)
    actual = checkpoint_dir.resolve(strict=False)
    if expected != actual:
        raise RuntimeError(
            "requested checkpoint directory disagrees with OSM-42D certification"
        )

    digest = package.get("package_digest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("OSM-42D package digest is malformed")
    return cert


def run_runtime_probe(
    executable: pathlib.Path,
    checkpoint_dir: pathlib.Path,
) -> dict:
    argv = (
        [sys.executable, str(executable), str(checkpoint_dir)]
        if executable.suffix.lower() == ".py"
        else [str(executable), str(checkpoint_dir)]
    )
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"runtime checkpoint load probe failed with exit code "
            f"{completed.returncode}: {detail}"
        )

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("runtime checkpoint load probe returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("runtime checkpoint load probe returned non-object JSON")
    if result.get("stage") != "official-full-checkpoint-runtime-load":
        raise RuntimeError("runtime checkpoint load probe stage mismatch")
    if result.get("loaded") is not True:
        raise RuntimeError("runtime checkpoint load probe did not load checkpoint")
    if result.get("layer_count") != 48:
        raise RuntimeError("official runtime load must expose exactly 48 decoder layers")

    claims = result.get("claims", {})
    required = (
        "checkpoint_opened",
        "dense_inventory_bound",
        "decoder_stack_created",
        "vulkan_context_created",
    )
    for key in required:
        if claims.get(key) is not True:
            raise RuntimeError(f"runtime load claim missing: {key}")
    if claims.get("inference_executed") is not False:
        raise RuntimeError("OSM-43A must not claim inference execution")
    if claims.get("token_generated") is not False:
        raise RuntimeError("OSM-43A must not claim token generation")
    return result


def build_evidence(
    certification_path: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    executable: pathlib.Path,
) -> dict:
    cert = verify_conversion_certification(certification_path, checkpoint_dir)
    result = run_runtime_probe(executable, checkpoint_dir)
    return {
        "schema_version": 1,
        "stage": "official-full-checkpoint-runtime-load-certification",
        "source": cert["source"],
        "conversion_certification": {
            "path": str(certification_path),
            "sha256": sha256_file(certification_path),
            "certification_id": cert.get("certification_id"),
        },
        "package": {
            "checkpoint_dir": str(checkpoint_dir),
            "package_digest_sha256": cert["package"]["package_digest_sha256"],
        },
        "runtime_load": result,
        "claims": {
            "official_full_checkpoint_loaded": True,
            "vulkan_runtime_initialized": True,
            "decoder_stack_created": True,
            "runtime_inference_validated": False,
            "first_token_generated": False,
        },
    }


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing OSM-43A evidence conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversion-certification", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--probe-executable", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = build_evidence(
        pathlib.Path(args.conversion_certification),
        pathlib.Path(args.checkpoint_dir),
        pathlib.Path(args.probe_executable),
    )
    write_immutable(pathlib.Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
