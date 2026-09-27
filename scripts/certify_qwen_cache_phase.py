#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib

OFFICIAL_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"
COUNTERS = (
    "host_hits",
    "host_misses",
    "gpu_hits",
    "gpu_misses",
    "gpu_loads",
    "gpu_evictions",
)


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


def certify(
    performance_path: pathlib.Path,
    latency_path: pathlib.Path,
) -> dict:
    performance = load_json(performance_path)
    latency = load_json(latency_path)

    if performance.get("schema_version") != 1 or performance.get("stage") != "official-chat-performance-instrumentation":
        raise RuntimeError("not valid OSM-45A performance evidence")
    if latency.get("schema_version") != 1 or latency.get("stage") != "official-chat-phase-latency-certification":
        raise RuntimeError("not valid OSM-45B latency evidence")

    source = performance.get("source", {})
    if source.get("model") != OFFICIAL_MODEL or source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("OSM-45A model/snapshot pin mismatch")
    if latency.get("source") != source:
        raise RuntimeError("OSM-45A/45B source mismatch")
    if latency.get("package") != performance.get("package"):
        raise RuntimeError("OSM-45A/45B package mismatch")

    parent = latency.get("parent_performance_evidence", {})
    if parent.get("sha256") != sha256_file(performance_path):
        raise RuntimeError("OSM-45B is not bound to supplied OSM-45A evidence")

    probe = performance.get("probe", {})
    phase = probe.get("phase_cache")
    if not isinstance(phase, dict):
        raise RuntimeError("OSM-45C phase_cache evidence missing")
    prefill = phase.get("prefill")
    decode = phase.get("decode")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise RuntimeError("OSM-45C prefill/decode cache evidence missing")

    for name in COUNTERS:
        for label, values in (("prefill", prefill), ("decode", decode)):
            value = values.get(name)
            if not isinstance(value, int) or value < 0:
                raise RuntimeError(f"invalid {label} cache counter: {name}")

    host_final = probe.get("host_cache", {})
    gpu_final = probe.get("gpu_cache", {})
    expected = {
        "host_hits": host_final.get("hits"),
        "host_misses": host_final.get("misses"),
        "gpu_hits": gpu_final.get("hits"),
        "gpu_misses": gpu_final.get("misses"),
        "gpu_loads": gpu_final.get("loads"),
        "gpu_evictions": gpu_final.get("evictions"),
    }
    for name, total in expected.items():
        if not isinstance(total, int) or total < 0:
            raise RuntimeError(f"final cache counter missing: {name}")
        if prefill[name] + decode[name] != total:
            raise RuntimeError(f"phase cache accounting mismatch: {name}")

    claims = probe.get("claims", {})
    if claims.get("prefill_decode_cache_activity_attributed") is not True:
        raise RuntimeError("OSM-45C probe cache attribution claim missing")

    if expected["gpu_misses"] <= 0 or expected["gpu_loads"] <= 0:
        raise RuntimeError("OSM-45C requires real routed-expert cache activity")

    def hit_rate(hits: int, misses: int):
        total = hits + misses
        return hits / total if total else None

    return {
        "schema_version": 1,
        "stage": "official-chat-cache-phase-attribution-certification",
        "source": source,
        "package": performance.get("package", {}),
        "parents": {
            "performance_evidence_sha256": sha256_file(performance_path),
            "phase_latency_evidence_sha256": sha256_file(latency_path),
        },
        "prefill": {
            **prefill,
            "host_hit_rate": hit_rate(prefill["host_hits"], prefill["host_misses"]),
            "gpu_hit_rate": hit_rate(prefill["gpu_hits"], prefill["gpu_misses"]),
        },
        "decode": {
            **decode,
            "host_hit_rate": hit_rate(decode["host_hits"], decode["host_misses"]),
            "gpu_hit_rate": hit_rate(decode["gpu_hits"], decode["gpu_misses"]),
        },
        "totals": expected,
        "claims": {
            "prefill_cache_activity_attributed": True,
            "decode_cache_activity_attributed": True,
            "phase_totals_reconcile_with_final_cache_stats": True,
            "cache_policy_optimized": False,
            "performance_target_met": False,
        },
    }


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing OSM-45C evidence conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance-evidence", required=True)
    parser.add_argument("--phase-latency-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = certify(
        pathlib.Path(args.performance_evidence),
        pathlib.Path(args.phase_latency_evidence),
    )
    write_immutable(pathlib.Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
