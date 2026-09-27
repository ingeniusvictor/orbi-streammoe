#!/usr/bin/env python3
import json
import pathlib
import stat
import tempfile

import measure_qwen_chat_performance as perf


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")


def make_probe(path: pathlib.Path, *, premature_perf: bool = False) -> None:
    path.write_text(f"""#!/usr/bin/env python3
import json
import time

payload = bytearray(8 * 1024 * 1024)
time.sleep(0.20)
print(json.dumps({{
  "stage": "official-chat-bounded-generation",
  "executed": True,
  "message_count": 2,
  "rendered_prompt": "<|im_start|>user\\nHola<|im_end|>\\n<|im_start|>assistant\\n",
  "prompt_token_count": 4,
  "prompt_token_ids": [1, 2, 3, 4],
  "generated_token_count": 3,
  "generated_token_ids": [10, 11, 12],
  "generated_logits": [1.0, 1.1, 1.2],
  "generated_text": "fixture",
  "model_steps": 6,
  "host_cache": {{"budget_bytes": 65536}},
  "gpu_cache": {{"resident_bytes": 4096, "budget_bytes": 65536}},
  "claims": {{
    "native_chat_renderer_used": True,
    "official_tokenizer_used": True,
    "chat_formatted_prompt_executed": True,
    "bounded_multitoken_chat_generation_validated": True,
    "tool_calling_validated": False,
    "tokens_per_second_measured": {str(premature_perf)},
    "ram_vram_profile_measured": False
  }}
}}))
""", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-osm45a-") as temp:
        root = pathlib.Path(temp)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        parent = root / "chat-cert.json"
        write_json(parent, {
            "schema_version": 1,
            "stage": "official-chat-bounded-generation-certification",
            "source": {
                "model": perf.OFFICIAL_MODEL,
                "snapshot": perf.OFFICIAL_SNAPSHOT,
            },
            "package": {
                "checkpoint_dir": str(checkpoint),
                "package_digest_sha256": "f" * 64,
            },
            "claims": {
                "native_chat_formatted_generation_validated": True,
                "tokens_per_second_measured": False,
                "ram_vram_profile_measured": False,
            },
        })

        probe = root / "probe.py"
        make_probe(probe)
        result, measurement = perf.run_measured_probe(
            probe,
            [
                str(checkpoint), "assets", "messages.json",
                "8", "64", "3", "1024", "65536", "65536",
            ],
        )
        if result["generated_token_count"] != 3:
            raise RuntimeError("unexpected generated token count")
        if measurement["wall_time_ms"] < 100:
            raise RuntimeError("wall-time instrumentation did not observe probe delay")
        if measurement["peak_sampled_process_rss_bytes"] <= 0:
            raise RuntimeError("RSS instrumentation did not sample process memory")
        if measurement["cold_end_to_end_generated_tokens_per_second"] <= 0:
            raise RuntimeError("cold output rate must be positive")
        if measurement["gpu_cache_resident_bytes"] != 4096:
            raise RuntimeError("GPU cache residency was not preserved")

        verified = perf.verify_parent(parent, checkpoint)
        payload = {
            "schema_version": 1,
            "stage": "official-chat-performance-instrumentation",
            "source": verified["source"],
            "package": verified["package"],
            "probe": result,
            "measurement": measurement,
            "claims": {
                "cold_run_wall_time_measured": True,
                "process_rss_sampled": True,
                "cache_residency_reported": True,
                "cold_end_to_end_output_rate_measured": True,
                "representative_steady_state_performance_certified": False,
                "device_total_vram_measured": False,
                "performance_target_met": False,
            },
        }
        output = root / "measurement.json"
        perf.write_immutable(output, payload)
        perf.write_immutable(output, payload)

        bad_probe = root / "bad-probe.py"
        make_probe(bad_probe, premature_perf=True)
        expect_failure(
            lambda: perf.run_measured_probe(
                bad_probe,
                [
                    str(checkpoint), "assets", "messages.json",
                    "8", "64", "3", "1024", "65536", "65536",
                ],
            ),
            "premature parent performance claim",
        )

        tampered = json.loads(parent.read_text(encoding="utf-8"))
        tampered["claims"]["native_chat_formatted_generation_validated"] = False
        write_json(parent, tampered)
        expect_failure(
            lambda: perf.verify_parent(parent, checkpoint),
            "missing OSM-44C parent claim",
        )

        print(
            "OSM-45A runtime performance instrumentation: PASS\n"
            "  cold_wall_time=PASS\n"
            "  process_rss_sampling=PASS\n"
            "  cache_residency_capture=PASS\n"
            "  cold_output_rate=PASS\n"
            "  representative_performance_claim_boundary=PASS\n"
            "  immutable_evidence=PASS\n"
            "  parent_tamper_rejection=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
