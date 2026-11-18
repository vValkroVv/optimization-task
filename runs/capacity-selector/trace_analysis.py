#!/usr/bin/env python3
"""Cross-check MCP Perfetto's capacity queries against original Kineto events."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from statistics import median


GPU = {"kernel", "gpu_memcpy", "gpu_memset"}
CUDA_API = {"cuda_driver", "cuda_runtime"}
CONDITIONAL = "torchtitan::conditional_capacity_mask"
QUANTIZE = "_rowwise_1d_quantize_kernel"
DISPATCH = "void hybrid_ep::dispatch_kernel<"
METADATA = "_routing_metadata_kernel"
SELECTOR = "_capacity_mask_kernel"
MICROBATCH = re.compile(r"task\.profile_update\.(\d+)\.microbatch\.rank(\d+)")


@dataclass(frozen=True)
class Event:
    name: str
    category: str
    start: int
    duration: int
    pid: int
    tid: int
    args: dict

    @classmethod
    def from_json(cls, event):
        return cls(
            event["name"], event.get("cat", ""), round(event["ts"] * 1000),
            round(event["dur"] * 1000), event.get("pid", -1), event.get("tid", -1),
            event.get("args", {}),
        )

    @property
    def end(self):
        return self.start + self.duration

    def contains(self, other):
        return (
            self.pid == other.pid and self.tid == other.tid
            and self.start <= other.start and self.end >= other.end
        )


def is_capacity(event):
    return (
        event.category == "kernel"
        and event.name.startswith("ncclDevKernel_AllReduce_Sum_u64")
        and event.args.get("In msg nelems") == 128
        and event.args.get("Group size") == 8
    )


def interval_union(events, start, end):
    active = 0
    covered_until = start
    for event in events:
        if event.start >= end:
            break
        if event.end > covered_until:
            clipped_end = min(event.end, end)
            active += clipped_end - max(covered_until, event.start)
            covered_until = clipped_end
    return active


def stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "sum": sum(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def launch_signature(event):
    return (
        event.name,
        tuple(event.args.get("grid", [])),
        tuple(event.args.get("block", [])),
    )


def analyze(path):
    contents = path.read_bytes()
    trace = json.loads(contents, parse_float=Decimal)
    events = sorted(
        (Event.from_json(item) for item in trace["traceEvents"]
         if item.get("ph") == "X" and "dur" in item),
        key=lambda event: event.start,
    )
    device = [event for event in events if event.category in GPU and event.duration > 0]
    if not device:
        raise ValueError(f"No GPU activity in {path}")
    origin = device[0].start
    cpu_by_correlation = defaultdict(list)
    syncs_by_external_id = defaultdict(list)
    conditionals = []
    dispatch_ops = []
    markers = []
    for event in events:
        corr = event.args.get("correlation")
        external = event.args.get("External id")
        if event.category in CUDA_API and corr is not None:
            cpu_by_correlation[corr].append(event)
        if event.name == "cudaStreamSynchronize" and external is not None:
            syncs_by_external_id[event.pid, event.tid, external].append(event)
        if event.category == "cpu_op" and event.name == CONDITIONAL:
            conditionals.append(event)
        if event.category == "cpu_op" and event.name == "hybridep::dispatch":
            dispatch_ops.append(event)
        match = MICROBATCH.fullmatch(event.name)
        if event.category == "user_annotation" and match:
            markers.append((event, int(match[1]), int(match[2])))

    def launch_for(event):
        matches = cpu_by_correlation.get(event.args.get("correlation"), [])
        return matches[0] if len(matches) == 1 else None

    copies = [
        event for event in device
        if event.category == "gpu_memcpy" and event.name.startswith("Memcpy DtoH")
        and event.args.get("bytes") == 1
    ]
    predicate_copies = []
    for copy in copies:
        runtime = launch_for(copy)
        owner = next((op for op in conditionals if runtime and op.contains(runtime)), None)
        if owner is None:
            continue
        candidates = syncs_by_external_id.get(
            (runtime.pid, runtime.tid, runtime.args.get("External id")), []
        )
        sync = next(
            (event for event in candidates
             if owner.contains(event) and event.start >= runtime.end and event.end >= copy.end),
            None,
        )
        predicate_copies.append({
            "device_start_ns": copy.start - origin,
            "duration_ns": copy.duration,
            "host_sync_ns": sync.duration if sync else None,
            "host_return_after_copy_ns": sync.end - copy.end if sync else None,
        })

    capacity = [event for event in device if is_capacity(event)]
    quantizes = [
        event for event in device if event.category == "kernel" and event.name == QUANTIZE
    ]
    dispatches = [
        event for event in device
        if event.category == "kernel" and event.name.startswith(DISPATCH)
    ]
    selectors = [
        event for event in device if event.category == "kernel" and SELECTOR in event.name
    ]
    dispatch_by_op = {}
    for dispatch in dispatches:
        launch = launch_for(dispatch)
        if launch:
            op = next((op for op in dispatch_ops if op.contains(launch)), None)
            if op:
                dispatch_by_op[op.start, op.pid, op.tid] = dispatch
    quant_starts = [event.start for event in quantizes]
    sites = []
    for index, collective in enumerate(capacity):
        start = collective.end
        group = {
            key: collective.args.get(key)
            for key in ("dtype", "Group size", "Process Group Ranks", "In msg nelems")
        }
        row = {
            "capacity_start_ns": collective.start - origin,
            "allreduce_ns": collective.duration,
            "collective_seq": collective.args.get("Seq"),
            "collective_group": group,
        }
        sites.append(row)
        quant_index = bisect_left(quant_starts, start)
        if quant_index == len(quantizes):
            row["unmatched"] = "no later input quantization"
            continue
        quant = quantizes[quant_index]
        if index + 1 < len(capacity) and quant.start >= capacity[index + 1].start:
            row["unmatched"] = "next capacity collective precedes input quantization"
            continue
        quant_launch = launch_for(quant)
        dispatch_op = next(
            (op for op in dispatch_ops if quant_launch and op.contains(quant_launch)), None
        )
        dispatch = (
            dispatch_by_op.get((dispatch_op.start, dispatch_op.pid, dispatch_op.tid))
            if dispatch_op else None
        )
        marker_index = next(
            (i for i, (mark, _, _) in enumerate(markers, 1)
             if quant_launch and mark.pid == quant_launch.pid
             and mark.start <= quant_launch.start < mark.end),
            None,
        )
        prep = [
            event for event in device
            if start <= event.start < quant.start and event.tid == quant.tid
            and event.category in {"kernel", "gpu_memcpy"}
        ]
        active = interval_union(device, start, quant.start)
        row.update(
            microbatch=marker_index,
            elapsed_ns=quant.start - start,
            active_ns=active,
            idle_ns=quant.start - start - active,
            quant_ns=quant.duration,
            quant_submit_vs_allreduce_end_ns=quant_launch.start - start if quant_launch else None,
            quant_start_after_submit_end_ns=quant.start - quant_launch.end if quant_launch else None,
            dispatch_signature=launch_signature(dispatch) if dispatch else None,
            quant_before_dispatch=quant.end <= dispatch.start if dispatch else None,
            prep=[
                {
                    "name": event.name,
                    "start_vs_allreduce_end_ns": event.start - start,
                    "duration_ns": event.duration,
                    "submit_vs_allreduce_end_ns": launch.start - start if launch else None,
                }
                for event in prep
                for launch in [launch_for(event)]
            ],
        )

    def signature_counts(family):
        return [[*signature, count] for signature, count in
                sorted(Counter(map(launch_signature, family)).items())]

    return {
        "path": str(path),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "rank_and_world_size": {
            key: trace.get("distributedInfo", {}).get(key) for key in ("rank", "world_size")
        },
        "profile_update_and_rank": sorted({(update, rank) for _, update, rank in markers}),
        "microbatches": len(markers),
        "device_envelope_ns": max(event.end for event in device) - origin,
        "capacity_collectives": len(capacity),
        "by_microbatch": sorted(Counter(row.get("microbatch") or 0 for row in sites).items()),
        "matched_windows": sum("unmatched" not in row for row in sites),
        "window_stats_ns": {
            key: stats(row.get(key) for row in sites)
            for key in ("elapsed_ns", "active_ns", "idle_ns", "allreduce_ns",
                        "quant_submit_vs_allreduce_end_ns", "quant_start_after_submit_end_ns")
        },
        "conditional_cpu_calls": len(conditionals),
        "one_byte_dtoh_total": len(copies),
        "confirmed_predicate_copies": predicate_copies,
        "confirmed_predicate_syncs": sum(
            row["host_sync_ns"] is not None for row in predicate_copies
        ),
        "predicate_sync_stats_ns": stats(row["host_sync_ns"] for row in predicate_copies),
        "device_selector_launches": len(selectors),
        "device_selector_stats_ns": stats(event.duration for event in selectors),
        "device_selector_resources": sorted({
            (event.args.get("registers per thread", -1), event.args.get("shared memory", -1))
            for event in selectors
        }),
        "quantize_launch_signatures": signature_counts(quantizes),
        "dispatch_launch_signatures": signature_counts(dispatches),
        "sites": sites,
    }


def compare(control, candidate):
    def relative_collective_sequence(trace):
        sites = trace["sites"]
        if not sites or any(site["collective_seq"] is None for site in sites):
            return None
        first_seq = sites[0]["collective_seq"]
        return [(site["collective_seq"] - first_seq, site["collective_group"]) for site in sites]

    control_collectives = relative_collective_sequence(control)
    candidate_collectives = relative_collective_sequence(candidate)
    return {
        "same_rank_and_world_size": control["rank_and_world_size"] == candidate["rank_and_world_size"],
        "same_profile_update_and_rank": (
            control["profile_update_and_rank"] == candidate["profile_update_and_rank"]
        ),
        "same_capacity_windows_per_microbatch": control["by_microbatch"] == candidate["by_microbatch"],
        "same_collective_groups_and_relative_sequence_numbers": (
            control_collectives == candidate_collectives
            if control_collectives is not None and candidate_collectives is not None else None
        ),
        "same_quantize_launch_signatures": (
            control["quantize_launch_signatures"] == candidate["quantize_launch_signatures"]
        ),
        "same_dispatch_launch_signatures": (
            control["dispatch_launch_signatures"] == candidate["dispatch_launch_signatures"]
        ),
        "sites_verified_quantize_before_dispatch": [
            sum(row.get("quant_before_dispatch") is True for row in trace["sites"])
            for trace in (control, candidate)
        ],
        "saved_ns": {
            key: control["window_stats_ns"][key]["sum"] - candidate["window_stats_ns"][key]["sum"]
            for key in ("elapsed_ns", "active_ns", "idle_ns")
            if control["matched_windows"] and candidate["matched_windows"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    control = analyze(args.control)
    result = {"control": control}
    if args.candidate:
        candidate = analyze(args.candidate)
        result.update(candidate=candidate, comparison=compare(control, candidate))
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
