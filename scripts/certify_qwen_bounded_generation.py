#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

OFFICIAL_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"
OFFICIAL_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


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


def verify_text_first_token_certification(path: pathlib.Path, checkpoint_dir: pathlib.Path) -> dict:
    cert = load_json(path)
    if cert.get("schema_version") != 1:
        raise RuntimeError("unsupported OSM-43C certification schema")
    if cert.get("stage") != "official-text-prompt-first-token-certification":
        raise RuntimeError("not an OSM-43C certification")
    source = cert.get("source", {})
    if source.get("model") != OFFICIAL_MODEL or source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("OSM-43C official model/snapshot pin mismatch")
    package = cert.get("package", {})
    recorded = package.get("checkpoint_dir")
    if not isinstance(recorded, str) or pathlib.Path(recorded).resolve(strict=False) != checkpoint_dir.resolve(strict=False):
        raise RuntimeError("checkpoint directory disagrees with OSM-43C certification")
    claims = cert.get("claims", {})
    if claims.get("official_first_continuation_token_generated") is not True:
        raise RuntimeError("OSM-43C first continuation claim missing")
    if claims.get("multi_token_generation_validated") is not False:
        raise RuntimeError("OSM-43C claim boundary changed unexpectedly")
    return cert


def verify_tokenizer_assets(asset_dir: pathlib.Path) -> str:
    tokenizer_json = asset_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise RuntimeError("official tokenizer.json is missing")
    digest = sha256_file(tokenizer_json)
    if digest != OFFICIAL_TOKENIZER_SHA256:
        raise RuntimeError("official tokenizer.json SHA256 mismatch")
    return digest


def run_probe(
    executable: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    tokenizer_asset_dir: pathlib.Path,
    prompt_file: pathlib.Path,
    max_prompt_tokens: int,
    max_new_tokens: int,
    lm_head_chunk_rows: int,
    host_cache_bytes: int,
    gpu_cache_bytes: int,
) -> dict:
    if not 2 <= max_new_tokens <= 8:
        raise RuntimeError("OSM-43D requires 2 <= max_new_tokens <= 8")
    for label, value in (
        ("max_prompt_tokens", max_prompt_tokens),
        ("lm_head_chunk_rows", lm_head_chunk_rows),
        ("host_cache_bytes", host_cache_bytes),
        ("gpu_cache_bytes", gpu_cache_bytes),
    ):
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"{label} must be a positive integer")
    if not prompt_file.is_file() or prompt_file.stat().st_size == 0:
        raise RuntimeError("prompt file must exist and be non-empty")

    args = [
        str(checkpoint_dir), str(tokenizer_asset_dir), str(prompt_file),
        str(max_prompt_tokens), str(max_new_tokens), str(lm_head_chunk_rows),
        str(host_cache_bytes), str(gpu_cache_bytes),
    ]
    argv = [sys.executable, str(executable), *args] if executable.suffix.lower() == ".py" else [str(executable), *args]
    completed = subprocess.run(argv, check=False, capture_output=True, text=True, shell=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"bounded generation probe failed with exit code {completed.returncode}: {detail}")

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("bounded generation probe returned invalid JSON") from exc
    if not isinstance(result, dict) or result.get("stage") != "official-bounded-multitoken-generation":
        raise RuntimeError("bounded generation probe stage mismatch")
    if result.get("executed") is not True:
        raise RuntimeError("bounded generation probe did not execute")

    prompt_ids = result.get("prompt_token_ids", [])
    generated_ids = result.get("generated_token_ids", [])
    logits = result.get("generated_logits", [])
    if result.get("prompt_token_count") != len(prompt_ids) or len(prompt_ids) == 0:
        raise RuntimeError("prompt token accounting mismatch")
    if len(prompt_ids) > max_prompt_tokens:
        raise RuntimeError("prompt token count exceeds bound")
    if result.get("generated_token_count") != len(generated_ids):
        raise RuntimeError("generated token accounting mismatch")
    if not 2 <= len(generated_ids) <= max_new_tokens:
        raise RuntimeError("generated token count violates OSM-43D bound")
    if len(logits) != len(generated_ids):
        raise RuntimeError("generated token/logit cardinality mismatch")
    if result.get("model_steps") != len(prompt_ids) + len(generated_ids) - 1:
        raise RuntimeError("autoregressive model-step accounting mismatch")
    if result.get("model", {}).get("layer_count") != 48:
        raise RuntimeError("bounded generation must execute a 48-layer model")

    claims = result.get("claims", {})
    for key in (
        "official_tokenizer_used",
        "text_prompt_tokenized",
        "teacher_forced_prompt_executed",
        "checkpoint_backed_48_layer_inference_executed",
        "bounded_multitoken_generation_validated",
        "generated_token_sequence_decoded",
    ):
        if claims.get(key) is not True:
            raise RuntimeError(f"OSM-43D claim missing: {key}")
    for key in ("chat_template_validated", "tokens_per_second_measured", "ram_vram_profile_measured"):
        if claims.get(key) is not False:
            raise RuntimeError(f"OSM-43D crossed claim boundary: {key}")
    return result


def build_evidence(
    text_first_token_certification: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    tokenizer_asset_dir: pathlib.Path,
    prompt_file: pathlib.Path,
    executable: pathlib.Path,
    max_prompt_tokens: int,
    max_new_tokens: int,
    lm_head_chunk_rows: int,
    host_cache_bytes: int,
    gpu_cache_bytes: int,
) -> dict:
    parent = verify_text_first_token_certification(text_first_token_certification, checkpoint_dir)
    tokenizer_sha = verify_tokenizer_assets(tokenizer_asset_dir)
    result = run_probe(
        executable, checkpoint_dir, tokenizer_asset_dir, prompt_file,
        max_prompt_tokens, max_new_tokens, lm_head_chunk_rows,
        host_cache_bytes, gpu_cache_bytes,
    )
    return {
        "schema_version": 1,
        "stage": "official-bounded-multitoken-generation-certification",
        "source": parent["source"],
        "package": parent["package"],
        "parent_text_first_token_certification": {
            "path": str(text_first_token_certification),
            "sha256": sha256_file(text_first_token_certification),
        },
        "tokenizer": {
            "model": OFFICIAL_MODEL,
            "snapshot": OFFICIAL_SNAPSHOT,
            "tokenizer_json_sha256": tokenizer_sha,
        },
        "prompt": {
            "path": str(prompt_file),
            "sha256": sha256_file(prompt_file),
            "utf8_bytes": prompt_file.stat().st_size,
        },
        "generation": result,
        "claims": {
            "official_bounded_multitoken_generation_validated": True,
            "official_generated_sequence_decoded": True,
            "bounded_prompt_generation_and_cache_limits_declared": True,
            "chat_template_validated": False,
            "tokens_per_second_measured": False,
            "ram_vram_profile_measured": False,
        },
    }


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing OSM-43D evidence conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-first-token-certification", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-asset-dir", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--probe-executable", required=True)
    parser.add_argument("--max-prompt-tokens", required=True, type=int)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--lm-head-chunk-rows", required=True, type=int)
    parser.add_argument("--host-cache-bytes", required=True, type=int)
    parser.add_argument("--gpu-cache-bytes", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_evidence(
        pathlib.Path(args.text_first_token_certification),
        pathlib.Path(args.checkpoint_dir),
        pathlib.Path(args.tokenizer_asset_dir),
        pathlib.Path(args.prompt_file),
        pathlib.Path(args.probe_executable),
        args.max_prompt_tokens,
        args.max_new_tokens,
        args.lm_head_chunk_rows,
        args.host_cache_bytes,
        args.gpu_cache_bytes,
    )
    write_immutable(pathlib.Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
