#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib

OFFICIAL_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"
SCHEMA_VERSION = 1

EXPECTED_PHASES = (
    ("windows_conversion_certification", "windows-full-conversion-certification"),
    ("runtime_load", "official-full-checkpoint-runtime-load-certification"),
    ("first_token", "official-first-token-inference-certification"),
    ("text_first_token", "official-text-prompt-first-token-certification"),
    ("bounded_generation", "official-bounded-multitoken-generation-certification"),
    ("chat_generation", "official-chat-bounded-generation-certification"),
    ("performance", "official-chat-performance-instrumentation"),
    ("phase_latency", "official-chat-phase-latency-certification"),
    ("cache_phase", "official-chat-cache-phase-attribution-certification"),
)


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def canonical_digest(payload: dict) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def resolve_path(value: str) -> pathlib.Path:
    return pathlib.Path(value).resolve(strict=False)


def verify_pack(pack_path: pathlib.Path) -> dict:
    pack = load_json(pack_path)
    if pack.get("schema_version") != 1:
        raise RuntimeError("unsupported OSM-46A pack schema")
    if pack.get("stage") != "windows-official-pilot-execution-pack":
        raise RuntimeError("not an OSM-46A execution pack")
    source = pack.get("source", {})
    if source.get("model") != OFFICIAL_MODEL or source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("OSM-46A official source pin mismatch")
    phases = pack.get("phases")
    if not isinstance(phases, list):
        raise RuntimeError("OSM-46A phases missing")
    observed = [(x.get("name"), x.get("expected_stage")) for x in phases]
    if observed != list(EXPECTED_PHASES):
        raise RuntimeError("OSM-46A phase chain mismatch")
    claims = pack.get("claims", {})
    if claims.get("real_windows_pilot_execution_planned") is not True:
        raise RuntimeError("OSM-46A plan claim missing")
    if claims.get("real_windows_pilot_executed") is not False:
        raise RuntimeError("OSM-46A pack crossed execution claim boundary")
    return pack


def verify_state(state_path: pathlib.Path, pack_path: pathlib.Path, pack: dict) -> dict:
    state = load_json(state_path)
    if state.get("schema_version") != 1 or state.get("stage") != "windows-official-pilot-run-state":
        raise RuntimeError("not an OSM-46A pilot run state")
    if state.get("pack_sha256") != sha256_file(pack_path):
        raise RuntimeError("OSM-46A state is not bound to supplied pack")
    if state.get("completed") is not True:
        raise RuntimeError("OSM-46A pilot state is incomplete")
    expected_names = [name for name, _ in EXPECTED_PHASES]
    if state.get("completed_phases") != expected_names:
        raise RuntimeError("OSM-46A completed phase list mismatch")
    claims = state.get("claims", {})
    if claims.get("all_planned_phases_completed") is not True:
        raise RuntimeError("OSM-46A completion claim missing")
    if claims.get("real_windows_pilot_executed") is not True:
        raise RuntimeError("OSM-46A real Windows pilot execution claim missing")
    if claims.get("performance_target_met") is not False:
        raise RuntimeError("OSM-46A must not pre-claim performance target")
    expected_final = pack["phases"][-1]["output"]
    if resolve_path(state.get("final_evidence", "")) != resolve_path(expected_final):
        raise RuntimeError("OSM-46A final evidence path mismatch")
    return state


def verify_evidence(pack: dict) -> tuple[list[dict], dict]:
    sealed = []
    source_ref = None
    package_ref = None
    checkpoint_ref = resolve_path(pack["checkpoint_dir"])

    for phase, expected_stage in EXPECTED_PHASES:
        phase_spec = next(p for p in pack["phases"] if p["name"] == phase)
        path = pathlib.Path(phase_spec["output"])
        if not path.is_file():
            raise RuntimeError(f"missing pilot evidence: {phase}: {path}")
        value = load_json(path)
        if value.get("stage") != expected_stage:
            raise RuntimeError(f"wrong stage for pilot evidence: {phase}")

        source = value.get("source")
        if isinstance(source, dict):
            model = source.get("model")
            snapshot = source.get("snapshot")
            if model is not None and model != OFFICIAL_MODEL:
                raise RuntimeError(f"model mismatch in evidence: {phase}")
            if snapshot is not None and snapshot != OFFICIAL_SNAPSHOT:
                raise RuntimeError(f"snapshot mismatch in evidence: {phase}")
            if model == OFFICIAL_MODEL and snapshot == OFFICIAL_SNAPSHOT:
                normalized = {"model": model, "snapshot": snapshot}
                if source_ref is None:
                    source_ref = normalized
                elif source_ref != normalized:
                    raise RuntimeError(f"source mismatch in evidence: {phase}")

        package = value.get("package")
        if isinstance(package, dict) and package:
            checkpoint_value = package.get("checkpoint_dir") or package.get("output_dir")
            digest = package.get("package_digest_sha256")
            normalized = {}
            if checkpoint_value:
                if resolve_path(checkpoint_value) != checkpoint_ref:
                    raise RuntimeError(f"checkpoint path mismatch in evidence: {phase}")
                normalized["checkpoint_dir"] = str(checkpoint_ref)
            if digest is not None:
                if not isinstance(digest, str) or len(digest) != 64:
                    raise RuntimeError(f"invalid package digest in evidence: {phase}")
                normalized["package_digest_sha256"] = digest
            if normalized:
                if package_ref is None:
                    package_ref = normalized
                else:
                    for key, val in normalized.items():
                        if key in package_ref and package_ref[key] != val:
                            raise RuntimeError(f"package mismatch in evidence: {phase}")
                        package_ref[key] = val

        sealed.append({
            "phase": phase,
            "stage": expected_stage,
            "path": str(path.resolve(strict=False)),
            "sha256": sha256_file(path),
        })

    if source_ref != {"model": OFFICIAL_MODEL, "snapshot": OFFICIAL_SNAPSHOT}:
        raise RuntimeError("pilot evidence chain does not preserve official model/snapshot")
    if package_ref is None or "package_digest_sha256" not in package_ref:
        raise RuntimeError("pilot evidence chain missing package digest")
    return sealed, package_ref


def summarize_metrics(pack: dict) -> dict:
    by_name = {p["name"]: pathlib.Path(p["output"]) for p in pack["phases"]}
    perf = load_json(by_name["performance"])
    latency = load_json(by_name["phase_latency"])
    cache = load_json(by_name["cache_phase"])

    measurement = perf.get("measurement", {})
    phase_latency = latency.get("phase_latency", {})
    required_perf = (
        "wall_time_ns",
        "peak_sampled_process_rss_bytes",
        "generated_token_count",
        "prompt_token_count",
        "cold_end_to_end_generated_tokens_per_second",
    )
    for key in required_perf:
        if measurement.get(key) is None:
            raise RuntimeError(f"missing OSM-45A metric: {key}")

    required_latency = ("prompt_prefill_ns", "decode_ns")
    for key in required_latency:
        if phase_latency.get(key) is None:
            raise RuntimeError(f"missing OSM-45B metric: {key}")

    totals = cache.get("totals")
    prefill = cache.get("prefill")
    decode = cache.get("decode")
    if not isinstance(totals, dict) or not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise RuntimeError("missing OSM-45C cache summaries")

    return {
        "performance": {
            "wall_time_ns": measurement["wall_time_ns"],
            "peak_sampled_process_rss_bytes": measurement["peak_sampled_process_rss_bytes"],
            "generated_token_count": measurement["generated_token_count"],
            "prompt_token_count": measurement["prompt_token_count"],
            "cold_end_to_end_generated_tokens_per_second": measurement["cold_end_to_end_generated_tokens_per_second"],
            "cold_end_to_end_ms_per_generated_token": measurement.get("cold_end_to_end_ms_per_generated_token"),
            "gpu_cache_resident_bytes": measurement.get("gpu_cache_resident_bytes"),
            "gpu_cache_budget_bytes": measurement.get("gpu_cache_budget_bytes"),
            "host_cache_budget_bytes": measurement.get("host_cache_budget_bytes"),
        },
        "phase_latency": {
            "prompt_prefill_ns": phase_latency["prompt_prefill_ns"],
            "decode_ns": phase_latency["decode_ns"],
            "average_prompt_step_ns": phase_latency.get("average_prompt_step_ns"),
            "average_decode_step_ns": phase_latency.get("average_decode_step_ns"),
        },
        "cache": {
            "prefill": prefill,
            "decode": decode,
            "totals": totals,
        },
    }


def build_seal(pack_path: pathlib.Path, state_path: pathlib.Path) -> dict:
    pack = verify_pack(pack_path)
    state = verify_state(state_path, pack_path, pack)
    sealed, package = verify_evidence(pack)
    metrics = summarize_metrics(pack)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "windows-official-pilot-evidence-seal",
        "source": {"model": OFFICIAL_MODEL, "snapshot": OFFICIAL_SNAPSHOT},
        "pilot_pack": {
            "path": str(pack_path.resolve(strict=False)),
            "sha256": sha256_file(pack_path),
            "pack_id": pack.get("pack_id"),
        },
        "pilot_state": {
            "path": str(state_path.resolve(strict=False)),
            "sha256": sha256_file(state_path),
            "completed_phases": state["completed_phases"],
        },
        "checkpoint": {
            "checkpoint_dir": str(resolve_path(pack["checkpoint_dir"])),
            "package_digest_sha256": package["package_digest_sha256"],
        },
        "evidence_chain": sealed,
        "metrics": metrics,
        "claims": {
            "real_windows_pilot_executed": True,
            "all_nine_evidence_phases_sealed": True,
            "official_model_snapshot_bound": True,
            "checkpoint_package_digest_bound": True,
            "performance_observed": True,
            "phase_latency_observed": True,
            "cache_phase_activity_observed": True,
            "representative_steady_state_performance_certified": False,
            "performance_target_met": False,
        },
    }
    payload["seal_id"] = canonical_digest(payload)[:24]
    return payload


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = canonical(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing OSM-46B seal conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    payload = build_seal(pathlib.Path(args.pack), pathlib.Path(args.state))
    write_immutable(pathlib.Path(args.output), payload)
    print(canonical(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
