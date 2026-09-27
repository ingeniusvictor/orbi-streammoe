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


def verify_bounded_generation_parent(path: pathlib.Path, checkpoint_dir: pathlib.Path) -> dict:
    cert = load_json(path)
    if cert.get("schema_version") != 1:
        raise RuntimeError("unsupported OSM-43D certification schema")
    if cert.get("stage") != "official-bounded-multitoken-generation-certification":
        raise RuntimeError("not an OSM-43D certification")
    source = cert.get("source", {})
    if source.get("model") != OFFICIAL_MODEL or source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("OSM-43D official model/snapshot pin mismatch")
    package = cert.get("package", {})
    recorded = package.get("checkpoint_dir")
    if not isinstance(recorded, str) or pathlib.Path(recorded).resolve(strict=False) != checkpoint_dir.resolve(strict=False):
        raise RuntimeError("checkpoint directory disagrees with OSM-43D certification")
    claims = cert.get("claims", {})
    if claims.get("official_bounded_multitoken_generation_validated") is not True:
        raise RuntimeError("OSM-43D bounded generation claim missing")
    if claims.get("chat_template_validated") is not False:
        raise RuntimeError("OSM-43D chat-template claim boundary changed")
    return cert


def verify_chat_reference(path: pathlib.Path, tokenizer_asset_dir: pathlib.Path) -> dict:
    reference = load_json(path)
    if reference.get("schema_version") != 1:
        raise RuntimeError("unsupported OSM-44A reference schema")
    if reference.get("stage") != "official-qwen-chat-template-reference":
        raise RuntimeError("not an OSM-44A chat-template reference")
    if reference.get("model") != OFFICIAL_MODEL or reference.get("revision") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("OSM-44A official model/snapshot pin mismatch")

    tokenizer_json = tokenizer_asset_dir / "tokenizer.json"
    tokenizer_config = tokenizer_asset_dir / "tokenizer_config.json"
    if not tokenizer_json.is_file() or not tokenizer_config.is_file():
        raise RuntimeError("official tokenizer assets missing")
    if sha256_file(tokenizer_json) != OFFICIAL_TOKENIZER_SHA256:
        raise RuntimeError("official tokenizer.json SHA256 mismatch")
    if reference.get("tokenizer_json_sha256") != sha256_file(tokenizer_json):
        raise RuntimeError("OSM-44A tokenizer digest mismatch")
    if reference.get("tokenizer_config_sha256") != sha256_file(tokenizer_config):
        raise RuntimeError("OSM-44A tokenizer_config digest mismatch")

    config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    chat_template = config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise RuntimeError("official tokenizer_config has no chat_template")
    template_digest = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if reference.get("chat_template_sha256") != template_digest:
        raise RuntimeError("OSM-44A chat_template digest mismatch")
    return reference


def run_probe(
    executable: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    tokenizer_asset_dir: pathlib.Path,
    messages_file: pathlib.Path,
    max_messages: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    lm_head_chunk_rows: int,
    host_cache_bytes: int,
    gpu_cache_bytes: int,
) -> dict:
    if not 2 <= max_new_tokens <= 8:
        raise RuntimeError("OSM-44C requires 2 <= max_new_tokens <= 8")
    for label, value in (
        ("max_messages", max_messages),
        ("max_prompt_tokens", max_prompt_tokens),
        ("lm_head_chunk_rows", lm_head_chunk_rows),
        ("host_cache_bytes", host_cache_bytes),
        ("gpu_cache_bytes", gpu_cache_bytes),
    ):
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"{label} must be a positive integer")
    if not messages_file.is_file() or messages_file.stat().st_size == 0:
        raise RuntimeError("messages file must exist and be non-empty")

    args = [
        str(checkpoint_dir), str(tokenizer_asset_dir), str(messages_file),
        str(max_messages), str(max_prompt_tokens), str(max_new_tokens),
        str(lm_head_chunk_rows), str(host_cache_bytes), str(gpu_cache_bytes),
    ]
    argv = [sys.executable, str(executable), *args] if executable.suffix.lower() == ".py" else [str(executable), *args]
    completed = subprocess.run(argv, check=False, capture_output=True, text=True, shell=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"chat generation probe failed with exit code {completed.returncode}: {detail}")

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("chat generation probe returned invalid JSON") from exc

    if not isinstance(result, dict) or result.get("stage") != "official-chat-bounded-generation":
        raise RuntimeError("chat generation probe stage mismatch")
    if result.get("executed") is not True:
        raise RuntimeError("chat generation probe did not execute")

    message_count = result.get("message_count")
    if not isinstance(message_count, int) or not 1 <= message_count <= max_messages:
        raise RuntimeError("message count violates declared bound")
    rendered = result.get("rendered_prompt")
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("rendered chat prompt missing")

    prompt_ids = result.get("prompt_token_ids", [])
    generated_ids = result.get("generated_token_ids", [])
    logits = result.get("generated_logits", [])
    if result.get("prompt_token_count") != len(prompt_ids) or not prompt_ids:
        raise RuntimeError("chat prompt token accounting mismatch")
    if len(prompt_ids) > max_prompt_tokens:
        raise RuntimeError("chat prompt exceeds declared token bound")
    if result.get("generated_token_count") != len(generated_ids):
        raise RuntimeError("chat generated-token accounting mismatch")
    if not 2 <= len(generated_ids) <= max_new_tokens:
        raise RuntimeError("chat generated-token count violates bound")
    if len(logits) != len(generated_ids):
        raise RuntimeError("chat token/logit cardinality mismatch")
    if result.get("model_steps") != len(prompt_ids) + len(generated_ids) - 1:
        raise RuntimeError("chat autoregressive step accounting mismatch")
    if result.get("model", {}).get("layer_count") != 48:
        raise RuntimeError("chat gate must execute a 48-layer model")

    claims = result.get("claims", {})
    for key in (
        "native_chat_renderer_used",
        "official_tokenizer_used",
        "chat_formatted_prompt_executed",
        "checkpoint_backed_48_layer_inference_executed",
        "bounded_multitoken_chat_generation_validated",
    ):
        if claims.get(key) is not True:
            raise RuntimeError(f"OSM-44C claim missing: {key}")
    for key in (
        "tool_calling_validated",
        "tokens_per_second_measured",
        "ram_vram_profile_measured",
    ):
        if claims.get(key) is not False:
            raise RuntimeError(f"OSM-44C crossed claim boundary: {key}")
    return result


def build_evidence(
    bounded_generation_certification: pathlib.Path,
    chat_template_reference: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    tokenizer_asset_dir: pathlib.Path,
    messages_file: pathlib.Path,
    executable: pathlib.Path,
    max_messages: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    lm_head_chunk_rows: int,
    host_cache_bytes: int,
    gpu_cache_bytes: int,
) -> dict:
    parent = verify_bounded_generation_parent(
        bounded_generation_certification, checkpoint_dir
    )
    reference = verify_chat_reference(chat_template_reference, tokenizer_asset_dir)
    result = run_probe(
        executable, checkpoint_dir, tokenizer_asset_dir, messages_file,
        max_messages, max_prompt_tokens, max_new_tokens, lm_head_chunk_rows,
        host_cache_bytes, gpu_cache_bytes,
    )
    return {
        "schema_version": 1,
        "stage": "official-chat-bounded-generation-certification",
        "source": parent["source"],
        "package": parent["package"],
        "parent_bounded_generation_certification": {
            "path": str(bounded_generation_certification),
            "sha256": sha256_file(bounded_generation_certification),
        },
        "chat_template_reference": {
            "path": str(chat_template_reference),
            "sha256": sha256_file(chat_template_reference),
            "tokenizer_config_sha256": reference["tokenizer_config_sha256"],
            "chat_template_sha256": reference["chat_template_sha256"],
        },
        "messages": {
            "path": str(messages_file),
            "sha256": sha256_file(messages_file),
            "utf8_bytes": messages_file.stat().st_size,
        },
        "generation": result,
        "claims": {
            "native_chat_formatted_generation_validated": True,
            "official_chat_template_reference_bound": True,
            "bounded_chat_generation_limits_declared": True,
            "tool_calling_validated": False,
            "tokens_per_second_measured": False,
            "ram_vram_profile_measured": False,
        },
    }


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing OSM-44C evidence conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounded-generation-certification", required=True)
    parser.add_argument("--chat-template-reference", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-asset-dir", required=True)
    parser.add_argument("--messages-file", required=True)
    parser.add_argument("--probe-executable", required=True)
    parser.add_argument("--max-messages", required=True, type=int)
    parser.add_argument("--max-prompt-tokens", required=True, type=int)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--lm-head-chunk-rows", required=True, type=int)
    parser.add_argument("--host-cache-bytes", required=True, type=int)
    parser.add_argument("--gpu-cache-bytes", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = build_evidence(
        pathlib.Path(args.bounded_generation_certification),
        pathlib.Path(args.chat_template_reference),
        pathlib.Path(args.checkpoint_dir),
        pathlib.Path(args.tokenizer_asset_dir),
        pathlib.Path(args.messages_file),
        pathlib.Path(args.probe_executable),
        args.max_messages,
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
