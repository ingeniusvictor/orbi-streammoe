#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib

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


def certify(parent_path: pathlib.Path) -> dict:
    parent = load_json(parent_path)
    if parent.get("schema_version") != 1:
        raise RuntimeError("unsupported OSM-45A schema")
    if parent.get("stage") != "official-chat-performance-instrumentation":
        raise RuntimeError("not an OSM-45A performance evidence file")

    source = parent.get("source", {})
    if source.get("model") != OFFICIAL_MODEL or source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("OSM-45A official model/snapshot pin mismatch")

    parent_claims = parent.get("claims", {})
    if parent_claims.get("cold_run_wall_time_measured") is not True:
        raise RuntimeError("OSM-45A cold-run wall-time claim missing")
    if parent_claims.get("representative_steady_state_performance_certified") is not False:
        raise RuntimeError("OSM-45A claim boundary changed")

    probe = parent.get("probe", {})
    if probe.get("stage") != "official-chat-bounded-generation":
        raise RuntimeError("OSM-45A embedded probe stage mismatch")
    if probe.get("executed") is not True:
        raise RuntimeError("OSM-45A embedded chat probe did not execute")

    phase = probe.get("phase_latency")
    if not isinstance(phase, dict):
        raise RuntimeError("OSM-45B phase latency evidence missing")

    prompt_steps = phase.get("prompt_step_durations_ns")
    decode_steps = phase.get("decode_step_durations_ns")
    prompt_total = phase.get("prompt_prefill_ns")
    decode_total = phase.get("decode_ns")
    prompt_count = probe.get("prompt_token_count")
    generated_count = probe.get("generated_token_count")

    if not isinstance(prompt_steps, list) or not all(isinstance(x, int) and x >= 0 for x in prompt_steps):
        raise RuntimeError("invalid OSM-45B prompt-step timings")
    if not isinstance(decode_steps, list) or not all(isinstance(x, int) and x >= 0 for x in decode_steps):
        raise RuntimeError("invalid OSM-45B decode-step timings")
    if not isinstance(prompt_count, int) or prompt_count <= 0:
        raise RuntimeError("invalid OSM-45B prompt token count")
    if not isinstance(generated_count, int) or generated_count < 2:
        raise RuntimeError("OSM-45B requires multi-token decode evidence")
    if len(prompt_steps) != prompt_count:
        raise RuntimeError("OSM-45B prefill timing cardinality mismatch")
    if len(decode_steps) != generated_count - 1:
        raise RuntimeError("OSM-45B decode timing cardinality mismatch")
    if not isinstance(prompt_total, int) or prompt_total <= 0:
        raise RuntimeError("OSM-45B prefill aggregate must be positive")
    if not isinstance(decode_total, int) or decode_total <= 0:
        raise RuntimeError("OSM-45B decode aggregate must be positive")
    if sum(prompt_steps) != prompt_total:
        raise RuntimeError("OSM-45B prefill aggregate mismatch")
    if sum(decode_steps) != decode_total:
        raise RuntimeError("OSM-45B decode aggregate mismatch")

    probe_claims = probe.get("claims", {})
    if probe_claims.get("prefill_decode_latency_measured") is not True:
        raise RuntimeError("OSM-45B probe phase-latency claim missing")

    return {
        "schema_version": 1,
        "stage": "official-chat-phase-latency-certification",
        "source": source,
        "package": parent.get("package", {}),
        "parent_performance_evidence": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
        },
        "phase_latency": {
            "prompt_token_count": prompt_count,
            "generated_token_count": generated_count,
            "prompt_prefill_ns": prompt_total,
            "decode_ns": decode_total,
            "prompt_step_durations_ns": prompt_steps,
            "decode_step_durations_ns": decode_steps,
            "average_prompt_step_ns": prompt_total / prompt_count,
            "average_decode_step_ns": decode_total / len(decode_steps),
            "first_prompt_step_ns": prompt_steps[0],
            "first_decode_step_ns": decode_steps[0],
        },
        "claims": {
            "prompt_prefill_latency_measured": True,
            "decode_latency_measured": True,
            "per_model_step_latency_recorded": True,
            "phase_accounting_validated": True,
            "representative_steady_state_performance_certified": False,
            "performance_target_met": False,
        },
    }


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing OSM-45B evidence conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = certify(pathlib.Path(args.performance_evidence))
    write_immutable(pathlib.Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
