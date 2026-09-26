#!/usr/bin/env python3
import argparse
import json
import pathlib
import struct
import urllib.request

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"
MAX_HEADER_BYTES = 64 * 1024 * 1024

SELECTED_TENSORS = (
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.linear_attn.in_proj_qkvz.weight",
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
    "model.layers.0.mlp.experts.0.down_proj.weight",
    "model.layers.0.mlp.experts.1.gate_proj.weight",
    "model.layers.0.mlp.experts.1.up_proj.weight",
    "model.layers.0.mlp.experts.1.down_proj.weight",
    "model.layers.0.mlp.shared_expert_gate.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "model.layers.3.mlp.experts.0.down_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
)


def parse_content_range(value: str, start: int, end: int) -> int:
    if not value.startswith("bytes ") or "/" not in value:
        raise RuntimeError(f"invalid Content-Range: {value!r}")
    span, total_text = value[6:].split("/", 1)
    actual_start, actual_end = (int(part) for part in span.split("-", 1))
    total = int(total_text)
    if actual_start != start or actual_end != end:
        raise RuntimeError(
            f"unexpected Content-Range span {actual_start}-{actual_end}; "
            f"wanted {start}-{end}"
        )
    if total <= actual_end:
        raise RuntimeError("invalid Content-Range total")
    return total


def fetch_exact_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    expected = end - start + 1
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "orbi-streammoe-osm38c/1",
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        status = response.getcode()
        if status != 206:
            raise RuntimeError(
                f"server ignored strict range request ({status}) for {url}"
            )
        total = parse_content_range(
            response.headers.get("Content-Range", ""),
            start,
            end,
        )
        data = response.read(expected + 1)
    if len(data) != expected:
        raise RuntimeError(
            f"range read length mismatch: got {len(data)}, wanted {expected}"
        )
    return data, total


def shard_url(filename: str) -> str:
    return (
        f"https://huggingface.co/{MODEL}/resolve/{SNAPSHOT}/{filename}"
        "?download=true"
    )


def fetch_header(filename: str) -> tuple[int, int, dict]:
    url = shard_url(filename)
    prefix, file_size = fetch_exact_range(url, 0, 7)
    header_size = struct.unpack("<Q", prefix)[0]
    if header_size <= 0 or header_size > MAX_HEADER_BYTES:
        raise RuntimeError(
            f"unsafe safetensors header size {header_size} for {filename}"
        )

    header_bytes, second_total = fetch_exact_range(
        url,
        8,
        8 + header_size - 1,
    )
    if second_total != file_size:
        raise RuntimeError(f"shard size changed across range reads: {filename}")

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid safetensors JSON header for {filename}: {exc}"
        ) from exc

    if not isinstance(header, dict):
        raise RuntimeError(f"safetensors header must be an object: {filename}")
    return file_size, header_size, header


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--experts", help="comma-separated routed expert IDs")
    parser.add_argument(
        "--tensor-list-json",
        help="OSM-40D dense source inventory JSON to append to selected tensors",
    )
    args = parser.parse_args()

    if (args.layer is None) != (args.experts is None):
        raise RuntimeError("--layer and --experts must be supplied together")

    selected_tensors = list(SELECTED_TENSORS)
    if args.tensor_list_json is not None:
        tensor_list = json.loads(
            pathlib.Path(args.tensor_list_json).read_text(encoding="utf-8")
        )
        if tensor_list.get("schema_version") != 1:
            raise RuntimeError("unsupported tensor-list schema")
        entries = tensor_list.get("entries")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError("tensor-list entries must be a non-empty array")
        for entry in entries:
            name = entry.get("source_tensor")
            if not isinstance(name, str) or not name:
                raise RuntimeError("tensor-list contains invalid source_tensor")
            if name not in selected_tensors:
                selected_tensors.append(name)

    if args.experts is not None:
        if args.layer < 0:
            raise RuntimeError("--layer must be non-negative")
        seen = set()
        expert_ids = []
        for item in args.experts.split(","):
            item = item.strip()
            if not item:
                continue
            expert = int(item)
            if expert < 0 or expert in seen:
                raise RuntimeError(
                    "--experts must contain unique non-negative integers"
                )
            seen.add(expert)
            expert_ids.append(expert)
        if not expert_ids:
            raise RuntimeError("--experts produced an empty expert set")
        for expert in expert_ids:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = (
                    f"model.layers.{args.layer}.mlp.experts.{expert}."
                    f"{projection}.weight"
                )
                if name not in selected_tensors:
                    selected_tensors.append(name)

    metadata_dir = pathlib.Path(args.metadata_dir)
    index_path = metadata_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]

    shard_to_selected: dict[str, list[str]] = {}
    for name in selected_tensors:
        shard = weight_map.get(name)
        if not isinstance(shard, str):
            raise RuntimeError(f"selected tensor missing from weight_map: {name}")
        shard_to_selected.setdefault(shard, []).append(name)

    shard_results = []
    total_fetched = 0

    for filename in sorted(shard_to_selected):
        file_size, header_size, header = fetch_header(filename)
        selected_metadata = {}

        tensor_count = 0
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            tensor_count += 1
            if name in shard_to_selected[filename]:
                if not isinstance(entry, dict):
                    raise RuntimeError(f"invalid tensor header entry: {name}")
                selected_metadata[name] = {
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                    "data_offsets": entry["data_offsets"],
                }

        missing = set(shard_to_selected[filename]) - set(selected_metadata)
        if missing:
            raise RuntimeError(
                f"selected tensors absent from {filename} header: "
                + ", ".join(sorted(missing))
            )

        fetched_bytes = 8 + header_size
        total_fetched += fetched_bytes
        shard_results.append(
            {
                "filename": filename,
                "file_size": file_size,
                "header_size": header_size,
                "fetched_bytes": fetched_bytes,
                "tensor_count": tensor_count,
                "selected_tensor_metadata": selected_metadata,
            }
        )
        print(
            f"{filename}: file={file_size} header={header_size} "
            f"selected={len(selected_metadata)}"
        )

    manifest = {
        "schema_version": 1,
        "model": MODEL,
        "snapshot": SNAPSHOT,
        "selected_tensors": selected_tensors,
        "shards": shard_results,
        "total_fetched_bytes": total_fetched,
    }

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"range pilot: {len(shard_results)} shard headers, "
        f"{total_fetched} bytes fetched"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
