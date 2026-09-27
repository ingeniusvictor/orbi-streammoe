#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys


SCHEMA_VERSION = 1


def canonical_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def windows_memory_status() -> dict:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("GlobalMemoryStatusEx failed")
    return {
        "total_physical_bytes": int(status.ullTotalPhys),
        "available_physical_bytes": int(status.ullAvailPhys),
        "memory_load_percent": int(status.dwMemoryLoad),
    }


def portable_memory_status() -> dict:
    if sys.platform == "win32":
        return windows_memory_status()
    if hasattr(os, "sysconf"):
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return {
            "total_physical_bytes": page_size * total_pages,
            "available_physical_bytes": page_size * available_pages,
            "memory_load_percent": None,
        }
    raise RuntimeError("physical memory probe is unsupported on this platform")


def nearest_existing(path: pathlib.Path) -> pathlib.Path:
    current = path.resolve(strict=False)
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise RuntimeError(f"no existing ancestor for path: {path}")
        current = parent
    return current


def volume_id(path: pathlib.Path) -> str:
    anchor = nearest_existing(path)
    if sys.platform == "win32":
        drive = pathlib.PureWindowsPath(str(anchor)).drive
        if not drive:
            raise RuntimeError(f"could not determine Windows drive: {path}")
        return drive.upper()
    return f"dev:{anchor.stat().st_dev}"


def path_probe(path: pathlib.Path) -> dict:
    anchor = nearest_existing(path)
    usage = shutil.disk_usage(anchor)
    exists = path.exists()
    is_dir = path.is_dir() if exists else None
    writable_anchor = os.access(anchor, os.W_OK)
    return {
        "path": str(path),
        "exists": exists,
        "is_directory": is_dir,
        "nearest_existing_ancestor": str(anchor),
        "volume_id": volume_id(path),
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "writable_ancestor": bool(writable_anchor),
    }


def read_vulkan_summary(path: pathlib.Path | None, allow_missing: bool) -> dict:
    if path is not None:
        data = path.read_bytes()
        source = str(path)
        return vulkan_payload(data, source)

    executable = shutil.which("vulkaninfo")
    if executable is None:
        if allow_missing:
            return {
                "available": False,
                "source": None,
                "summary_sha256": None,
                "adapter_markers": [],
            }
        raise RuntimeError("vulkaninfo is required for Windows authorization evidence")

    completed = subprocess.run(
        [executable, "--summary"],
        check=False,
        capture_output=True,
    )
    data = completed.stdout + completed.stderr
    if completed.returncode != 0:
        if allow_missing:
            return {
                "available": False,
                "source": executable,
                "summary_sha256": sha256_bytes(data),
                "adapter_markers": [],
            }
        raise RuntimeError(
            f"vulkaninfo --summary failed with exit code {completed.returncode}"
        )
    return vulkan_payload(data, executable)


def vulkan_payload(data: bytes, source: str) -> dict:
    text = data.decode("utf-8", errors="replace")
    markers = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if (
            "devicename" in lowered
            or lowered.startswith("gpu")
            or "device name" in lowered
        ):
            markers.append(stripped)
    return {
        "available": True,
        "source": source,
        "summary_sha256": sha256_bytes(data),
        "summary_bytes": len(data),
        "adapter_markers": markers[:32],
    }


def build_probe(
    target_output: pathlib.Path,
    expert_work: pathlib.Path,
    dense_work: pathlib.Path,
    vulkan_summary: pathlib.Path | None,
    allow_non_windows: bool,
    allow_missing_vulkan: bool,
) -> dict:
    is_windows = platform.system() == "Windows"
    if not is_windows and not allow_non_windows:
        raise RuntimeError("OSM-42B host probe must run on Windows")

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "windows-full-conversion-host-probe",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "windows_required": not allow_non_windows,
        },
        "memory": portable_memory_status(),
        "paths": {
            "target_output": path_probe(target_output),
            "expert_work": path_probe(expert_work),
            "dense_work": path_probe(dense_work),
        },
        "vulkan": read_vulkan_summary(
            vulkan_summary,
            allow_missing=allow_missing_vulkan,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-output", required=True)
    parser.add_argument("--expert-work", required=True)
    parser.add_argument("--dense-work", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vulkan-summary-file")
    parser.add_argument("--allow-non-windows", action="store_true")
    parser.add_argument("--allow-missing-vulkan", action="store_true")
    args = parser.parse_args()

    payload = build_probe(
        pathlib.Path(args.target_output),
        pathlib.Path(args.expert_work),
        pathlib.Path(args.dense_work),
        pathlib.Path(args.vulkan_summary_file)
        if args.vulkan_summary_file
        else None,
        args.allow_non_windows,
        args.allow_missing_vulkan,
    )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_text(payload), encoding="utf-8")
    print(canonical_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
