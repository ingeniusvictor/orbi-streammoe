#!/usr/bin/env python3
import json
import pathlib
import tempfile

import certify_qwen_phase_latency as gate


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")


def fixture() -> dict:
    return {
        "schema_version": 1,
        "stage": "official-chat-performance-instrumentation",
        "source": {
            "model": gate.OFFICIAL_MODEL,
            "snapshot": gate.OFFICIAL_SNAPSHOT,
        },
        "package": {
            "checkpoint_dir": "fixture",
            "package_digest_sha256": "a" * 64,
        },
        "probe": {
            "stage": "official-chat-bounded-generation",
            "executed": True,
            "prompt_token_count": 3,
            "generated_token_count": 3,
            "phase_latency": {
                "prompt_prefill_ns": 600,
                "decode_ns": 900,
                "prompt_step_durations_ns": [100, 200, 300],
                "decode_step_durations_ns": [400, 500],
            },
            "claims": {
                "prefill_decode_latency_measured": True,
            },
        },
        "claims": {
            "cold_run_wall_time_measured": True,
            "representative_steady_state_performance_certified": False,
        },
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-osm45b-") as temp:
        root = pathlib.Path(temp)
        parent = root / "performance.json"
        write_json(parent, fixture())

        result = gate.certify(parent)
        phase = result["phase_latency"]
        if phase["average_prompt_step_ns"] != 200:
            raise RuntimeError("average prefill step mismatch")
        if phase["average_decode_step_ns"] != 450:
            raise RuntimeError("average decode step mismatch")
        if not result["claims"]["phase_accounting_validated"]:
            raise RuntimeError("phase accounting claim missing")
        if result["claims"]["representative_steady_state_performance_certified"]:
            raise RuntimeError("OSM-45B crossed steady-state claim boundary")

        output = root / "cert.json"
        gate.write_immutable(output, result)
        gate.write_immutable(output, result)

        bad = fixture()
        bad["probe"]["phase_latency"]["decode_ns"] = 901
        write_json(parent, bad)
        expect_failure(lambda: gate.certify(parent), "decode aggregate mismatch")

        bad = fixture()
        bad["probe"]["phase_latency"]["decode_step_durations_ns"] = [900]
        write_json(parent, bad)
        expect_failure(lambda: gate.certify(parent), "decode cardinality mismatch")

        bad = fixture()
        bad["claims"]["representative_steady_state_performance_certified"] = True
        write_json(parent, bad)
        expect_failure(lambda: gate.certify(parent), "parent claim boundary")

        print(
            "OSM-45B phase-level latency instrumentation: PASS\n"
            "  prefill_cardinality=PASS\n"
            "  decode_cardinality=PASS\n"
            "  aggregate_accounting=PASS\n"
            "  per_step_latency=PASS\n"
            "  immutable_evidence=PASS\n"
            "  steady_state_claim_boundary=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
