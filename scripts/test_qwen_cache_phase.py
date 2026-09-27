#!/usr/bin/env python3
import json
import pathlib
import tempfile

import certify_qwen_cache_phase as gate


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")


def performance_fixture() -> dict:
    return {
        "schema_version": 1,
        "stage": "official-chat-performance-instrumentation",
        "source": {
            "model": gate.OFFICIAL_MODEL,
            "snapshot": gate.OFFICIAL_SNAPSHOT,
        },
        "package": {"checkpoint_dir": "fixture", "package_digest_sha256": "b" * 64},
        "probe": {
            "stage": "official-chat-bounded-generation",
            "executed": True,
            "phase_cache": {
                "prefill": {
                    "host_hits": 2, "host_misses": 5,
                    "gpu_hits": 3, "gpu_misses": 5,
                    "gpu_loads": 5, "gpu_evictions": 1,
                },
                "decode": {
                    "host_hits": 4, "host_misses": 1,
                    "gpu_hits": 6, "gpu_misses": 1,
                    "gpu_loads": 1, "gpu_evictions": 1,
                },
            },
            "host_cache": {"hits": 6, "misses": 6},
            "gpu_cache": {
                "hits": 9, "misses": 6, "loads": 6, "evictions": 2,
            },
            "claims": {
                "prefill_decode_cache_activity_attributed": True,
            },
        },
    }


def latency_fixture(performance_hash: str) -> dict:
    return {
        "schema_version": 1,
        "stage": "official-chat-phase-latency-certification",
        "source": {
            "model": gate.OFFICIAL_MODEL,
            "snapshot": gate.OFFICIAL_SNAPSHOT,
        },
        "package": {"checkpoint_dir": "fixture", "package_digest_sha256": "b" * 64},
        "parent_performance_evidence": {
            "path": "performance.json",
            "sha256": performance_hash,
        },
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-osm45c-") as temp:
        root = pathlib.Path(temp)
        performance = root / "performance.json"
        latency = root / "latency.json"

        write_json(performance, performance_fixture())
        write_json(latency, latency_fixture(gate.sha256_file(performance)))

        result = gate.certify(performance, latency)
        if not result["claims"]["phase_totals_reconcile_with_final_cache_stats"]:
            raise RuntimeError("cache reconciliation claim missing")
        if result["totals"]["gpu_loads"] != 6:
            raise RuntimeError("GPU load total mismatch")
        if result["prefill"]["gpu_hit_rate"] != 3 / 8:
            raise RuntimeError("prefill GPU hit rate mismatch")
        if result["claims"]["cache_policy_optimized"]:
            raise RuntimeError("OSM-45C crossed optimization claim boundary")

        output = root / "cert.json"
        gate.write_immutable(output, result)
        gate.write_immutable(output, result)

        bad = performance_fixture()
        bad["probe"]["gpu_cache"]["loads"] = 7
        write_json(performance, bad)
        write_json(latency, latency_fixture(gate.sha256_file(performance)))
        expect_failure(
            lambda: gate.certify(performance, latency),
            "phase/final load mismatch",
        )

        bad = performance_fixture()
        bad["probe"]["claims"]["prefill_decode_cache_activity_attributed"] = False
        write_json(performance, bad)
        write_json(latency, latency_fixture(gate.sha256_file(performance)))
        expect_failure(
            lambda: gate.certify(performance, latency),
            "missing attribution claim",
        )

        write_json(performance, performance_fixture())
        wrong_latency = latency_fixture("0" * 64)
        write_json(latency, wrong_latency)
        expect_failure(
            lambda: gate.certify(performance, latency),
            "parent evidence hash mismatch",
        )

        print(
            "OSM-45C cache phase attribution: PASS\n"
            "  prefill_cache_attribution=PASS\n"
            "  decode_cache_attribution=PASS\n"
            "  final_counter_reconciliation=PASS\n"
            "  derived_hit_rates=PASS\n"
            "  immutable_evidence=PASS\n"
            "  optimization_claim_boundary=PASS\n"
            "  parent_binding=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
