"""Attribute GFSDP gather, expert quantization, and dW packing in an EP8 trace."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ANALYZER = Path(__file__).resolve().parents[1] / "compile-baseline/analyze.py"
CALLBACK = "GFSDP::root_post_backward_callback"
DENSE_MTP = ("mtp.layers.0", "mtp.layers.0.block.feed_forward")
EXPERT_GRIDS = {(384, 32, 1): "w13", (512, 12, 1): "w2"}


def load_analyzer():
    spec = importlib.util.spec_from_file_location("_gfsdp_trace_analyzer", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def owner(scopes, prefix):
    return next(
        (
            scope.name[len(prefix) : -1]
            for scope in scopes
            if scope.name.startswith(prefix) and scope.name.endswith(")")
        ),
        None,
    )


def summarize(path: Path, update: int, rank: int):
    analyzer = load_analyzer()
    raw = path.read_bytes()
    content = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    records = json.loads(content, parse_float=Decimal)["traceEvents"]
    events = [
        analyzer.Event(
            index,
            row["name"],
            row["cat"],
            analyzer.ns(row["ts"]),
            analyzer.ns(row["dur"]),
            row["pid"],
            row["tid"],
            row.get("args", {}),
        )
        for index, row in enumerate(records)
        if row.get("ph") == "X"
        and row.get("cat") in analyzer.GPU | analyzer.API | analyzer.CPU
    ]
    device = [event for event in events if event.cat in analyzer.GPU]
    apis = [event for event in events if event.cat in analyzer.API]
    cpu = [event for event in events if event.cat in analyzer.CPU]
    label = f"task.profile_update.{update}.microbatch.rank{rank}"
    markers = sorted(
        (
            event
            for event in cpu
            if event.cat == "user_annotation" and event.name == label
        ),
        key=lambda event: event.start,
    )
    if len(markers) != 3 or any(
        left.end > right.start for left, right in zip(markers, markers[1:])
    ):
        raise ValueError(f"{path} must contain three disjoint microbatches for {label}")
    if len({event.args.get("device", event.pid) for event in device}) != 1:
        raise ValueError(f"{path} must contain exactly one CUDA device")
    launches, coverage = analyzer.launches(
        records, device, apis, len({event.pid for event in apis}) == 1
    )
    ancestry = analyzer.ancestors(apis, cpu)
    microbatches = []
    for marker in markers:
        host = [event for event in cpu if marker.start <= event.start < marker.end]
        callbacks = sorted(
            (event for event in host if event.name == CALLBACK),
            key=lambda event: event.start,
        )
        if len(callbacks) != 3:
            raise ValueError("Expected two head callbacks followed by decoder backward")
        head_begin, head_end = callbacks[0].end, callbacks[1].start
        dense_gathers, packing, quantization, refresh = [], [], [], []
        for event, launch in launches.items():
            if not marker.start <= launch.start < marker.end:
                continue
            scopes = ancestry[launch]
            gather = owner(scopes, "GFSDP::all_gather (")
            if (
                gather in DENSE_MTP
                and "allgather" in event.name.lower()
                and event.args.get("Group size") == 8
            ):
                dense_gathers.append(
                    {
                        "group": gather,
                        "between_head_callbacks": head_begin <= launch.start < head_end,
                        "duration_ms": analyzer.ms(event.dur),
                        "input_elements": event.args.get("In msg nelems"),
                        "output_elements": event.args.get("Out msg nelems"),
                        "dtype": event.args.get("dtype"),
                    }
                )
            packed = owner(scopes, "FSDP::post_backward_accumulate (")
            if packed and packed.endswith("[efsdp]"):
                packing.append(event)
            if "_block_quantization_kernel" in event.name:
                if any(scope.name == "GFSDP::fp8_expert_refresh" for scope in scopes):
                    refresh.append(event)
                elif tuple(event.args.get("grid", ())) in EXPERT_GRIDS:
                    quantization.append(event)
        microbatches.append(
            {
                "expert_logical_unshards": dict(
                    Counter(
                        event.name
                        for event in host
                        if event.name.startswith("GFSDP::all_gather (")
                        and event.name.endswith("[efsdp])")
                    )
                ),
                "dense_mtp_nccl_gathers": dense_gathers,
                "expert_post_backward_accumulate": analyzer.metrics(packing),
                "expert_block_quantization": analyzer.metrics(quantization),
                "expert_fp8_cache_refresh": analyzer.metrics(refresh),
            }
        )
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "update": update,
        "rank": rank,
        "device_launch_attribution": dict(Counter(coverage.values())),
        "microbatches": microbatches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--update", type=int, default=21)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace, args.update, args.rank), indent=2))


if __name__ == "__main__":
    main()
