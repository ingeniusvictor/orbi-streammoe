#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import sys

OFFICIAL_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"
SCHEMA_VERSION = 1

PHASES = (
    "windows_conversion_certification",
    "runtime_load",
    "first_token",
    "text_first_token",
    "bounded_generation",
    "chat_generation",
    "performance",
    "phase_latency",
    "cache_phase",
)

EXPECTED_STAGES = {
    "windows_conversion_certification": "windows-full-conversion-certification",
    "runtime_load": "official-full-checkpoint-runtime-load-certification",
    "first_token": "official-first-token-inference-certification",
    "text_first_token": "official-text-prompt-first-token-certification",
    "bounded_generation": "official-bounded-multitoken-generation-certification",
    "chat_generation": "official-chat-bounded-generation-certification",
    "performance": "official-chat-performance-instrumentation",
    "phase_latency": "official-chat-phase-latency-certification",
    "cache_phase": "official-chat-cache-phase-attribution-certification",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def canonical_digest(payload: dict) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def require_file(path: pathlib.Path, label: str) -> pathlib.Path:
    path = path.resolve(strict=False)
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def require_dir(path: pathlib.Path, label: str) -> pathlib.Path:
    path = path.resolve(strict=False)
    if not path.is_dir():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def command(script: pathlib.Path, *args: str) -> list[str]:
    return [sys.executable, str(script.resolve(strict=False)), *map(str, args)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--authorization", required=True)
    p.add_argument("--execution-manifest", required=True)
    p.add_argument("--operator-state", required=True)
    p.add_argument("--command-plan", required=True)
    p.add_argument("--receipt-dir")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--tokenizer-asset-dir", required=True)
    p.add_argument("--chat-template-reference", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--messages-file", required=True)
    p.add_argument("--probe-runtime-load", required=True)
    p.add_argument("--probe-first-token", required=True)
    p.add_argument("--probe-text-first-token", required=True)
    p.add_argument("--probe-bounded-generation", required=True)
    p.add_argument("--probe-chat-generation", required=True)
    p.add_argument("--first-token-id", required=True, type=int)
    p.add_argument("--max-messages", required=True, type=int)
    p.add_argument("--max-prompt-tokens", required=True, type=int)
    p.add_argument("--max-new-tokens", required=True, type=int)
    p.add_argument("--lm-head-chunk-rows", required=True, type=int)
    p.add_argument("--host-cache-bytes", required=True, type=int)
    p.add_argument("--gpu-cache-bytes", required=True, type=int)
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not 2 <= args.max_new_tokens <= 8:
        raise RuntimeError("OSM-46A requires 2 <= max_new_tokens <= 8")
    for label in ("max_messages", "max_prompt_tokens", "lm_head_chunk_rows",
                  "host_cache_bytes", "gpu_cache_bytes"):
        if getattr(args, label) <= 0:
            raise RuntimeError(f"{label} must be positive")
    if args.first_token_id < 0:
        raise RuntimeError("first_token_id must be non-negative")

    root = pathlib.Path(args.repo_root).resolve(strict=False)
    scripts = {
        "42d": require_file(root / "scripts/certify_qwen_windows_conversion.py", "OSM-42D script"),
        "43a": require_file(root / "scripts/certify_qwen_runtime_load.py", "OSM-43A script"),
        "43b": require_file(root / "scripts/certify_qwen_first_token.py", "OSM-43B script"),
        "43c": require_file(root / "scripts/certify_qwen_text_first_token.py", "OSM-43C script"),
        "43d": require_file(root / "scripts/certify_qwen_bounded_generation.py", "OSM-43D script"),
        "44c": require_file(root / "scripts/certify_qwen_chat_generation.py", "OSM-44C script"),
        "45a": require_file(root / "scripts/measure_qwen_chat_performance.py", "OSM-45A script"),
        "45b": require_file(root / "scripts/certify_qwen_phase_latency.py", "OSM-45B script"),
        "45c": require_file(root / "scripts/certify_qwen_cache_phase.py", "OSM-45C script"),
    }

    authorization = require_file(pathlib.Path(args.authorization), "authorization")
    manifest = require_file(pathlib.Path(args.execution_manifest), "execution manifest")
    state = require_file(pathlib.Path(args.operator_state), "operator state")
    command_plan = require_file(pathlib.Path(args.command_plan), "command plan")
    checkpoint = require_dir(pathlib.Path(args.checkpoint_dir), "checkpoint directory")
    tokenizer = require_dir(pathlib.Path(args.tokenizer_asset_dir), "tokenizer asset directory")
    chat_ref = require_file(pathlib.Path(args.chat_template_reference), "chat-template reference")
    prompt = require_file(pathlib.Path(args.prompt_file), "prompt file")
    messages = require_file(pathlib.Path(args.messages_file), "messages file")

    probes = {
        "runtime_load": require_file(pathlib.Path(args.probe_runtime_load), "runtime-load probe"),
        "first_token": require_file(pathlib.Path(args.probe_first_token), "first-token probe"),
        "text_first_token": require_file(pathlib.Path(args.probe_text_first_token), "text-first-token probe"),
        "bounded_generation": require_file(pathlib.Path(args.probe_bounded_generation), "bounded-generation probe"),
        "chat_generation": require_file(pathlib.Path(args.probe_chat_generation), "chat-generation probe"),
    }

    evidence = pathlib.Path(args.evidence_dir).resolve(strict=False)
    outputs = {
        "windows_conversion_certification": evidence / "42d-windows-conversion.json",
        "runtime_load": evidence / "43a-runtime-load.json",
        "first_token": evidence / "43b-first-token.json",
        "text_first_token": evidence / "43c-text-first-token.json",
        "bounded_generation": evidence / "43d-bounded-generation.json",
        "chat_generation": evidence / "44c-chat-generation.json",
        "performance": evidence / "45a-performance.json",
        "phase_latency": evidence / "45b-phase-latency.json",
        "cache_phase": evidence / "45c-cache-phase.json",
    }

    common = [
        "--checkpoint-dir", str(checkpoint),
    ]
    phases = [
        {
            "name": "windows_conversion_certification",
            "output": str(outputs["windows_conversion_certification"]),
            "expected_stage": EXPECTED_STAGES["windows_conversion_certification"],
            "argv": command(
                scripts["42d"],
                "--authorization", authorization,
                "--manifest", manifest,
                "--state", state,
                "--command-plan", command_plan,
                *(["--receipt-dir", args.receipt_dir] if args.receipt_dir else []),
                "--output", outputs["windows_conversion_certification"],
            ),
        },
        {
            "name": "runtime_load",
            "output": str(outputs["runtime_load"]),
            "expected_stage": EXPECTED_STAGES["runtime_load"],
            "argv": command(
                scripts["43a"],
                "--conversion-certification", outputs["windows_conversion_certification"],
                *common,
                "--probe-executable", probes["runtime_load"],
                "--output", outputs["runtime_load"],
            ),
        },
        {
            "name": "first_token",
            "output": str(outputs["first_token"]),
            "expected_stage": EXPECTED_STAGES["first_token"],
            "argv": command(
                scripts["43b"],
                "--runtime-load-certification", outputs["runtime_load"],
                *common,
                "--probe-executable", probes["first_token"],
                "--token-id", args.first_token_id,
                "--lm-head-chunk-rows", args.lm_head_chunk_rows,
                "--host-cache-bytes", args.host_cache_bytes,
                "--gpu-cache-bytes", args.gpu_cache_bytes,
                "--output", outputs["first_token"],
            ),
        },
        {
            "name": "text_first_token",
            "output": str(outputs["text_first_token"]),
            "expected_stage": EXPECTED_STAGES["text_first_token"],
            "argv": command(
                scripts["43c"],
                "--first-token-certification", outputs["first_token"],
                *common,
                "--tokenizer-asset-dir", tokenizer,
                "--prompt-file", prompt,
                "--probe-executable", probes["text_first_token"],
                "--max-prompt-tokens", args.max_prompt_tokens,
                "--lm-head-chunk-rows", args.lm_head_chunk_rows,
                "--host-cache-bytes", args.host_cache_bytes,
                "--gpu-cache-bytes", args.gpu_cache_bytes,
                "--output", outputs["text_first_token"],
            ),
        },
        {
            "name": "bounded_generation",
            "output": str(outputs["bounded_generation"]),
            "expected_stage": EXPECTED_STAGES["bounded_generation"],
            "argv": command(
                scripts["43d"],
                "--text-first-token-certification", outputs["text_first_token"],
                *common,
                "--tokenizer-asset-dir", tokenizer,
                "--prompt-file", prompt,
                "--probe-executable", probes["bounded_generation"],
                "--max-prompt-tokens", args.max_prompt_tokens,
                "--max-new-tokens", args.max_new_tokens,
                "--lm-head-chunk-rows", args.lm_head_chunk_rows,
                "--host-cache-bytes", args.host_cache_bytes,
                "--gpu-cache-bytes", args.gpu_cache_bytes,
                "--output", outputs["bounded_generation"],
            ),
        },
        {
            "name": "chat_generation",
            "output": str(outputs["chat_generation"]),
            "expected_stage": EXPECTED_STAGES["chat_generation"],
            "argv": command(
                scripts["44c"],
                "--bounded-generation-certification", outputs["bounded_generation"],
                "--chat-template-reference", chat_ref,
                *common,
                "--tokenizer-asset-dir", tokenizer,
                "--messages-file", messages,
                "--probe-executable", probes["chat_generation"],
                "--max-messages", args.max_messages,
                "--max-prompt-tokens", args.max_prompt_tokens,
                "--max-new-tokens", args.max_new_tokens,
                "--lm-head-chunk-rows", args.lm_head_chunk_rows,
                "--host-cache-bytes", args.host_cache_bytes,
                "--gpu-cache-bytes", args.gpu_cache_bytes,
                "--output", outputs["chat_generation"],
            ),
        },
        {
            "name": "performance",
            "output": str(outputs["performance"]),
            "expected_stage": EXPECTED_STAGES["performance"],
            "argv": command(
                scripts["45a"],
                "--chat-generation-certification", outputs["chat_generation"],
                "--probe-executable", probes["chat_generation"],
                *common,
                "--tokenizer-asset-dir", tokenizer,
                "--messages-file", messages,
                "--max-messages", args.max_messages,
                "--max-prompt-tokens", args.max_prompt_tokens,
                "--max-new-tokens", args.max_new_tokens,
                "--lm-head-chunk-rows", args.lm_head_chunk_rows,
                "--host-cache-bytes", args.host_cache_bytes,
                "--gpu-cache-bytes", args.gpu_cache_bytes,
                "--output", outputs["performance"],
            ),
        },
        {
            "name": "phase_latency",
            "output": str(outputs["phase_latency"]),
            "expected_stage": EXPECTED_STAGES["phase_latency"],
            "argv": command(
                scripts["45b"],
                "--performance-evidence", outputs["performance"],
                "--output", outputs["phase_latency"],
            ),
        },
        {
            "name": "cache_phase",
            "output": str(outputs["cache_phase"]),
            "expected_stage": EXPECTED_STAGES["cache_phase"],
            "argv": command(
                scripts["45c"],
                "--performance-evidence", outputs["performance"],
                "--phase-latency-evidence", outputs["phase_latency"],
                "--output", outputs["cache_phase"],
            ),
        },
    ]

    inputs = {
        "authorization": authorization,
        "execution_manifest": manifest,
        "operator_state": state,
        "command_plan": command_plan,
        "chat_template_reference": chat_ref,
        "prompt_file": prompt,
        "messages_file": messages,
        **{f"probe_{k}": v for k, v in probes.items()},
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "windows-official-pilot-execution-pack",
        "source": {"model": OFFICIAL_MODEL, "snapshot": OFFICIAL_SNAPSHOT},
        "checkpoint_dir": str(checkpoint),
        "tokenizer_asset_dir": str(tokenizer),
        "evidence_dir": str(evidence),
        "inputs": {
            k: {"path": str(v), "sha256": sha256_file(v)}
            for k, v in inputs.items()
        },
        "limits": {
            "first_token_id": args.first_token_id,
            "max_messages": args.max_messages,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "lm_head_chunk_rows": args.lm_head_chunk_rows,
            "host_cache_bytes": args.host_cache_bytes,
            "gpu_cache_bytes": args.gpu_cache_bytes,
        },
        "phases": phases,
        "claims": {
            "real_windows_pilot_execution_planned": True,
            "shell_string_execution_used": False,
            "manual_intermediate_json_editing_required": False,
            "real_windows_pilot_executed": False,
            "performance_target_met": False,
        },
    }
    payload["pack_id"] = canonical_digest(payload)[:24]

    out = pathlib.Path(args.output)
    rendered = canonical(payload)
    if out.exists() and out.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"existing OSM-46A pack conflicts: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
