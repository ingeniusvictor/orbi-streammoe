#!/usr/bin/env python3
import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

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


def verify_parent(path: pathlib.Path, checkpoint_dir: pathlib.Path) -> dict:
    cert = load_json(path)
    if cert.get("schema_version") != 1:
        raise RuntimeError("unsupported OSM-44C certification schema")
    if cert.get("stage") != "official-chat-bounded-generation-certification":
        raise RuntimeError("not an OSM-44C certification")
    source = cert.get("source", {})
    if source.get("model") != OFFICIAL_MODEL or source.get("snapshot") != OFFICIAL_SNAPSHOT:
        raise RuntimeError("OSM-44C official model/snapshot pin mismatch")
    package = cert.get("package", {})
    recorded = package.get("checkpoint_dir")
    if not isinstance(recorded, str) or pathlib.Path(recorded).resolve(strict=False) != checkpoint_dir.resolve(strict=False):
        raise RuntimeError("checkpoint directory disagrees with OSM-44C certification")
    claims = cert.get("claims", {})
    if claims.get("native_chat_formatted_generation_validated") is not True:
        raise RuntimeError("OSM-44C chat generation claim missing")
    if claims.get("tokens_per_second_measured") is not False:
        raise RuntimeError("OSM-44C performance claim boundary changed")
    if claims.get("ram_vram_profile_measured") is not False:
        raise RuntimeError("OSM-44C memory-profile claim boundary changed")
    return cert


def _linux_memory(pid: int):
    status = pathlib.Path(f"/proc/{pid}/status")
    if not status.is_file():
        return None
    rss = None
    peak = None
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            peak = int(line.split()[1]) * 1024
    if rss is None:
        return None
    return rss, peak if peak is not None else rss


def _windows_memory(pid: int):
    from ctypes import wintypes

    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
        False,
        pid,
    )
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            ctypes.sizeof(counters),
        )
        if not ok:
            return None
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def process_memory(pid: int):
    try:
        if os.name == "nt":
            return _windows_memory(pid)
        if sys.platform.startswith("linux"):
            return _linux_memory(pid)
        return None
    except (OSError, ValueError, ctypes.Error):
        return None


def validate_probe_result(result: dict) -> None:
    if result.get("stage") != "official-chat-bounded-generation":
        raise RuntimeError("OSM-45A probe stage mismatch")
    if result.get("executed") is not True:
        raise RuntimeError("OSM-45A probe did not execute")

    generated = result.get("generated_token_count")
    generated_ids = result.get("generated_token_ids")
    prompt_count = result.get("prompt_token_count")
    if not isinstance(generated, int) or generated < 2:
        raise RuntimeError("OSM-45A requires a multi-token chat run")
    if not isinstance(generated_ids, list) or len(generated_ids) != generated:
        raise RuntimeError("OSM-45A generated-token accounting mismatch")
    if not isinstance(prompt_count, int) or prompt_count <= 0:
        raise RuntimeError("OSM-45A prompt token count missing")

    claims = result.get("claims", {})
    for key in (
        "native_chat_renderer_used",
        "official_tokenizer_used",
        "chat_formatted_prompt_executed",
        "bounded_multitoken_chat_generation_validated",
    ):
        if claims.get(key) is not True:
            raise RuntimeError(f"OSM-45A inherited claim missing: {key}")
    for key in ("tokens_per_second_measured", "ram_vram_profile_measured"):
        if claims.get(key) is not False:
            raise RuntimeError(f"OSM-45A parent probe crossed claim boundary: {key}")


def run_measured_probe(executable: pathlib.Path, args: list[str]) -> tuple[dict, dict]:
    argv = (
        [sys.executable, str(executable), *args]
        if executable.suffix.lower() == ".py"
        else [str(executable), *args]
    )

    start_ns = time.perf_counter_ns()
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )

    samples = 0
    max_rss = 0
    max_reported_peak = 0
    last_rss = 0

    while process.poll() is None:
        memory = process_memory(process.pid)
        if memory is not None:
            rss, peak = memory
            samples += 1
            last_rss = rss
            max_rss = max(max_rss, rss)
            max_reported_peak = max(max_reported_peak, peak)
        time.sleep(0.05)

    stdout, stderr = process.communicate()
    end_ns = time.perf_counter_ns()

    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip()
        raise RuntimeError(
            f"measured chat probe failed with exit code {process.returncode}: {detail}"
        )

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("measured chat probe returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("measured chat probe returned non-object JSON")
    validate_probe_result(result)

    if samples == 0 or max(max_rss, max_reported_peak) <= 0:
        raise RuntimeError(
            "OSM-45A requires supported process RSS sampling on this host"
        )

    wall_ns = end_ns - start_ns
    if wall_ns <= 0:
        raise RuntimeError("invalid measured wall time")

    generated = result["generated_token_count"]
    wall_seconds = wall_ns / 1_000_000_000.0
    measurement = {
        "wall_time_ns": wall_ns,
        "wall_time_ms": wall_ns / 1_000_000.0,
        "rss_sample_count": samples,
        "last_sampled_process_rss_bytes": last_rss,
        "peak_sampled_process_rss_bytes": max(max_rss, max_reported_peak),
        "generated_token_count": generated,
        "prompt_token_count": result["prompt_token_count"],
        "cold_end_to_end_generated_tokens_per_second": generated / wall_seconds,
        "cold_end_to_end_ms_per_generated_token": (wall_ns / 1_000_000.0) / generated,
        "host_cache_budget_bytes": result.get("host_cache", {}).get("budget_bytes"),
        "gpu_cache_resident_bytes": result.get("gpu_cache", {}).get("resident_bytes"),
        "gpu_cache_budget_bytes": result.get("gpu_cache", {}).get("budget_bytes"),
    }
    return result, measurement


def write_immutable(path: pathlib.Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing OSM-45A evidence conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-generation-certification", required=True)
    parser.add_argument("--probe-executable", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-asset-dir", required=True)
    parser.add_argument("--messages-file", required=True)
    parser.add_argument("--max-messages", required=True, type=int)
    parser.add_argument("--max-prompt-tokens", required=True, type=int)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--lm-head-chunk-rows", required=True, type=int)
    parser.add_argument("--host-cache-bytes", required=True, type=int)
    parser.add_argument("--gpu-cache-bytes", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    parent_path = pathlib.Path(args.chat_generation_certification)
    parent = verify_parent(parent_path, checkpoint_dir)

    probe_args = [
        str(checkpoint_dir),
        args.tokenizer_asset_dir,
        args.messages_file,
        str(args.max_messages),
        str(args.max_prompt_tokens),
        str(args.max_new_tokens),
        str(args.lm_head_chunk_rows),
        str(args.host_cache_bytes),
        str(args.gpu_cache_bytes),
    ]
    result, measurement = run_measured_probe(
        pathlib.Path(args.probe_executable),
        probe_args,
    )

    payload = {
        "schema_version": 1,
        "stage": "official-chat-performance-instrumentation",
        "source": parent["source"],
        "package": parent["package"],
        "parent_chat_generation_certification": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
        },
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

    write_immutable(pathlib.Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
