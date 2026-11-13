"""Preserve overlapping Kineto device events on separate Perfetto display lanes."""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


def prepare(source: Path, output: Path) -> None:
    with gzip.open(source, "rt") as stream:
        trace = json.load(stream)
    events = trace["traceEvents"]
    device_categories = {"kernel", "gpu_memcpy", "gpu_memset"}
    tracks = collections.defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in device_categories:
            tracks[event["pid"], event["tid"]].append(event)

    moves = {}
    moved = 0
    census = collections.defaultdict(lambda: {"count": 0, "sum_duration_us": 0.0})
    for (pid, tid), track in tracks.items():
        ends = []
        for event in sorted(track, key=lambda item: item["ts"]):
            lane = next((i for i, end in enumerate(ends) if end <= event["ts"]), len(ends))
            if lane == len(ends):
                ends.append(0)
            ends[lane] = event["ts"] + event["dur"]
            category = census[event["cat"]]
            category["count"] += 1
            category["sum_duration_us"] += event["dur"]
            if lane:
                # A display lane does not change args.stream or CUDA ordering.
                key = (pid, tid, event["ts"])
                assert key not in moves
                moves[key] = 1_000_000 + tid * 100 + lane
                event["tid"] = moves[key]
                moved += 1

    moved_flows = 0
    for event in events:
        if event.get("ph") == "f":
            key = (event.get("pid"), event.get("tid"), event.get("ts"))
            if key in moves:
                event["tid"] = moves[key]
                moved_flows += 1
    kept = [event for event in events if event.get("cat") != "gpu_user_annotation"]
    trace["traceEvents"] = kept
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=1) as compressed:
            compressed.write(json.dumps(trace, separators=(",", ":")).encode())
    receipt = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "transformation": "Remove GPU user ranges; color overlapping device events into nonoverlapping synthetic display tids and move their matching flow endpoints. Keep original timestamps, durations, names, arguments (including actual args.stream), CPU events and trace metadata. Source bytes are unchanged.",
        "device_census": dict(census),
        "moved_device_events": moved,
        "moved_flow_endpoints": moved_flows,
        "removed_gpu_ranges": len(events) - len(kept),
    }
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
