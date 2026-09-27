#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

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


def verify_runtime_load_certification(
    certification_path: pathlib.Path,
    checkpoint_dir: pathlib.Path,
) -> dict:
    cert = load_json(certification_path)
    if cert.get("schema_version") != 1:
        raise RuntimeError("unsupported OSM-43A certification schema")
    if cert.get("stage") != "official-full-checkpoint-runtime-load-certification":
        raise RuntimeError("not an OSM-43A runtime-load certification")

    source = cert.get("source", {})
    if source.get("model") != OFFICIAL_MODEL:
        raise RuntimeError("OSM-43A model pin mismatch")
    if source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("OSM-43A snapshot pin mismatch")

    claims = cert.get("claims", {})
    if claims.get("official_full_checkpoint_loaded") is not True:
        raise RuntimeError("OSM-43A did not certify official checkpoint load")
    if claims.get("decoder_stack_created") is not True:
        raise RuntimeError("OSM-43A did not certify decoder stack creation")
    if claims.get("runtime_inference_validated") is not False:
        raise RuntimeError("OSM-43A claim boundary changed unexpectedly")
    if claims.get("first_token_generated") is not False:
        raise RuntimeError("OSM-43A already claims first-token generation")

    package = cert.get("package", {})
    recorded = package.get("checkpoint_dir")
    if not isinstance(recorded, str) or not recorded:
        raise RuntimeError("OSM-43A certification is missing checkpoint_dir")
    if pathlib.Path(recorded).resolve(strict=False) != checkpoint_dir.resolve(strict=False):
        raise RuntimeError("checkpoint directory disagrees with OSM-43A certification")

    digest = package.get("package_digest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("OSM-43A package digest is malformed")
    return cert


def run_probe(
    executable: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    token_id: int,
    lm_head_chunk_rows: int,
    host_cache_bytes: int,
    gpu_cache_bytes: int,
) -> dict:
    for label, value in (
        ("token_id", token_id),
        ("lm_head_chunk_rows", lm_head_chunk_rows),
        ("host_cache_bytes", host_cache_bytes),
        ("gpu_cache_bytes", gpu_cache_bytes),
    ):
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{label} must be a non-negative integer")
    if lm_head_chunk_rows == 0 or host_cache_bytes == 0 or gpu_cache_bytes == 0:
        raise RuntimeError("chunk/cache limits must be non-zero")

    args = [
        str(checkpoint_dir),
        str(token_id),
        str(lm_head_chunk_rows),
        str(host_cache_bytes),
        str(gpu_cache_bytes),
    ]
    argv = (
        [sys.executable, str(executable), *args]
        if executable.suffix.lower() == ".py"
        else [str(executable), *args]
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
            f"first-token probe failed with exit code {completed.returncode}: {detail}"
        )

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("first-token probe returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("first-token probe returned non-object JSON")
    if result.get("stage") != "official-first-token-inference":
        raise RuntimeError("first-token probe stage mismatch")
    if result.get("executed") is not True:
        raise RuntimeError("first-token probe did not execute")
    if result.get("input_token_id") != token_id:
        raise RuntimeError("first-token probe input token mismatch")

    model = result.get("model", {})
    if model.get("layer_count") != 48:
        raise RuntimeError("first-token probe must execute a 48-layer model")

    generated = result.get("generated_token_id")
    vocab = model.get("vocab_size")
    if not isinstance(generated, int) or not isinstance(vocab, int):
        raise RuntimeError("first-token probe token metadata is malformed")
    if generated < 0 or generated >= vocab:
        raise RuntimeError("generated token is outside vocabulary")

    claims = result.get("claims", {})
    required = (
        "one_checkpoint_backed_step_executed",
        "streamed_embedding_executed",
        "decoder_48_layers_executed",
        "final_rmsnorm_executed",
        "streamed_lm_head_executed",
        "first_token_generated",
    )
    for key in required:
        if claims.get(key) is not True:
            raise RuntimeError(f"first-token claim missing: {key}")
    for key in (
        "text_prompt_tokenized",
        "multi_token_generation_validated",
        "tokens_per_second_measured",
    ):
        if claims.get(key) is not False:
            raise RuntimeError(f"OSM-43B crossed claim boundary: {key}")
    return result


def build_evidence(
    runtime_load_certification: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    executable: pathlib.Path,
    token_id: int,
    lm_head_chunk_rows: int,
    host_cache_bytes: int,
    gpu_cache_bytes: int,
) -> dict:
    load_cert = verify_runtime_load_certification(
        runtime_load_certification, checkpoint_dir
    )
    result = run_probe(
        executable,
        checkpoint_dir,
        token_id,
        lm_head_chunk_rows,
        host_cache_bytes,
        gpu_cache_bytes,
    )
    return {
        "schema_version": 1,
        "stage": "official-first-token-inference-certification",
        "source": load_cert["source"],
        "runtime_load_certification": {
            "path": str(runtime_load_certification),
            "sha256": sha256_file(runtime_load_certification),
        },
        "package": load_cert["package"],
        "inference": result,
        "claims": {
            "official_checkpoint_first_token_generated": True,
            "checkpoint_backed_48_layer_inference_executed": True,
            "bounded_cache_limits_declared": True,
            "text_prompt_tokenized": False,
            "multi_token_generation_validated": False,
            "tokens_per_second_measured": False,
        },
    }


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing OSM-43B evidence conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-load-certification", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--probe-executable", required=True)
    parser.add_argument("--token-id", required=True, type=int)
    parser.add_argument("--lm-head-chunk-rows", required=True, type=int)
    parser.add_argument("--host-cache-bytes", required=True, type=int)
    parser.add_argument("--gpu-cache-bytes", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = build_evidence(
        pathlib.Path(args.runtime_load_certification),
        pathlib.Path(args.checkpoint_dir),
        pathlib.Path(args.probe_executable),
        args.token_id,
        args.lm_head_chunk_rows,
        args.host_cache_bytes,
        args.gpu_cache_bytes,
    )
    write_immutable(pathlib.Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
