#!/usr/bin/env python3
"""Inspect the parent reduce-scatter to layer-7 backward interval in a saved trace."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import sys

import analyze_diagnostic as trace
from campaign import digest, read_json, require, write_json


SYNC_API = re.compile(r"^(?:cuda|cu)(?:EventRecord|StreamWaitEvent|.*Synchronize)")
EVENT_FIELDS = {"event", "eventid", "eventhandle", "cudaevent", "cudaeventid", "cudaeventhandle"}


def link_work(events):
    runtime, endpoints, work = defaultdict(list), defaultdict(set), []
    for event in events:
        category, phase = event.get("cat"), event.get("ph")
        if category == "ac2g" and phase in ("s", "f") and "id" in event:
            endpoints[phase, event["id"]].add(trace.key(event))
        elif category in trace.HOST and phase == "X":
            runtime[event.get("args", {}).get("correlation")].append(event)
        elif category in trace.GPU and phase == "X":
            work.append(event)

    launches = {}
    for device in work:
        correlation = device.get("args", {}).get("correlation")
        matches = [host for host in runtime[correlation]
                   if trace.key(host) in endpoints["s", correlation]]
        if len(matches) == 1 and trace.key(device) in endpoints["f", correlation]:
            launches[id(device)] = matches[0]
    return work, launches


def event_handle(event):
    return {key: value for key, value in event.get("args", {}).items()
            if re.sub(r"[^a-z]", "", key.lower()) in EVENT_FIELDS and value is not None}


def describe(event, origin):
    return {
        "name": event["name"], "cat": event.get("cat"), "pid": event.get("pid"), "tid": event.get("tid"),
        "start_ms": trace.relative(event, origin), "end_ms": trace.relative(event, origin, "end"),
        "args": event.get("args", {}),
    }


def context(event, scopes):
    parents = sorted((scope for scope in scopes if trace.contains(scope, event)), key=lambda e: e["dur"])
    return [scope["name"] for scope in parents[:6]]


def summarize(index, micro, scopes, work, launches, sync):
    origin = micro["ts"]
    stop = trace.end(micro)
    cpu = [scope for scope in scopes if origin <= scope["ts"] < stop]
    pairs = [(device, launch) for device in work if (launch := launches.get(id(device))) is not None
             and origin <= launch["ts"] < stop]
    collectives = trace.semantic_collectives(pairs, cpu, (), origin, ())

    def collective(op, group):
        candidates = [device for device, row in collectives[op, group] if row["phase"] == "backward"]
        require(len(candidates) == 1, f"Microbatch {index}: expected one backward {op} for {group}")
        return candidates[0]

    layer7, reason = trace.layer7_compute(cpu, pairs)
    require(layer7, f"Microbatch {index}: {reason}")
    first_l7 = trace.first(layer7)
    l7_launch = launches[id(first_l7)]
    layer7_ag = collective("all_gather", trace.LAYER7)
    parent_rs = collective("post_backward_reduce", trace.MTP)
    norm_rs = collective("post_backward_reduce", trace.NORM)
    rs_done, l7_start = trace.end(parent_rs), first_l7["ts"]

    def describe_device(device):
        launch = launches.get(id(device))
        return describe(device, origin) | {
            "launch": None if launch is None else describe(launch, origin) | {"scopes": context(launch, cpu)},
        }

    def scoped_work(name):
        owners = [scope for scope in cpu if scope["name"] == name]
        return [device for device, launch in pairs if any(trace.contains(owner, launch) for owner in owners)]

    compute_stream = [device for device in work if trace.thread(device) == trace.thread(first_l7)]
    predecessors = [device for device in compute_stream
                    if device["ts"] < l7_start and trace.end(device) > rs_done and rs_done < l7_start]
    before_rs_end = trace.last([device for device in compute_stream if trace.end(device) <= rs_done
                                and device["ts"] < l7_start])
    gap_ms = max(0.0, (l7_start - rs_done) / 1000)
    busy_ms = trace.overlap_ms([(rs_done, l7_start)], trace.intervals(predecessors))
    parent_epilogue = [device for device in scoped_work(f"GFSDP::post_backward_reduce ({trace.MTP})")
                       if device is not parent_rs and trace.end(device) > rs_done]
    norm_prelude = [device for device in scoped_work(f"GFSDP::post_backward_reduce ({trace.NORM})")
                    if device is not norm_rs and device["ts"] < norm_rs["ts"]]
    nccl_neighbors = [device for device in work if trace.thread(device) == trace.thread(parent_rs)
                      and device["ts"] < norm_rs["ts"] and trace.end(device) > rs_done
                      and device is not parent_rs and device is not norm_rs]

    # CUDA API calls can be queued much earlier than the GPU gap they affect.
    host_start = min(launches[id(layer7_ag)]["ts"], launches[id(parent_rs)]["ts"], l7_launch["ts"])
    host_end = max(trace.end(launches[id(norm_rs)]), trace.end(l7_launch))
    host_sync = [event for event in sync if event["ts"] < host_end and trace.end(event) >= host_start]
    return {
        "microbatch": index,
        "layer7_first_kernel": describe_device(first_l7),
        "layer7_dense_gather": describe_device(layer7_ag),
        "parent_nccl_reduce_scatter": describe_device(parent_rs),
        "norm_nccl_reduce_scatter": describe_device(norm_rs),
        "parent_rs_end_to_layer7_device_ms": (l7_start - rs_done) / 1000,
        "first_layer7_host_launch_minus_parent_rs_device_end_ms": (l7_launch["ts"] - rs_done) / 1000,
        "first_layer7_launch_precedes_parent_rs_device_end": l7_launch["ts"] < rs_done,
        "norm_rs_device_start_minus_first_layer7_device_start_ms": (norm_rs["ts"] - l7_start) / 1000,
        "observed_compute_stream_busy_ms_in_gap": busy_ms,
        "gap_without_observed_compute_stream_device_work_ms": max(0.0, gap_ms - busy_ms),
        "compute_stream_last_before_parent_rs_end": None if before_rs_end is None else describe_device(before_rs_end),
        "compute_stream_predecessors_in_gap": [describe_device(device) for device in predecessors],
        "parent_scoped_rs_epilogue": [describe_device(device) for device in parent_epilogue],
        "norm_scoped_rs_prelude": [describe_device(device) for device in norm_prelude],
        "parent_nccl_stream_device_events_before_norm_rs": [describe_device(device) for device in nccl_neighbors],
        "host_sync_window_ms": [(host_start - origin) / 1000, (host_end - origin) / 1000],
        "host_sync_event_handle_count": sum(bool(event_handle(event)) for event in host_sync),
        "host_sync_apis": [describe(event, origin) | {"scopes": context(event, cpu), "event_handle": event_handle(event)}
                           for event in host_sync],
    }


def analyze_trace(path, rank, update, microbatch_count):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        content = json.load(stream)
    seen_rank = content.get("distributedInfo", {}).get("rank")
    require(seen_rank is None or seen_rank == rank, f"{path}: rank mismatch")
    events = content["traceEvents"]
    work, launches = link_work(events)
    work.sort(key=lambda e: e["ts"])
    scopes = [event | {"name": trace.canonical(event["name"])} for event in events
              if event.get("ph") == "X" and event.get("cat") in ("user_annotation", "cpu_op")]
    marker = f"task.profile_update.{update}.microbatch.rank{rank}"
    micros = sorted((scope for scope in scopes if scope["name"] == marker), key=lambda e: e["ts"])
    require(len(micros) == microbatch_count and all(trace.end(a) <= b["ts"] for a, b in zip(micros, micros[1:])),
            f"{path}: expected {microbatch_count} distinct microbatch scopes for rank {rank}")
    sync = sorted((event for event in events if event.get("ph") == "X" and event.get("cat") in trace.HOST
                   and SYNC_API.match(event["name"])), key=lambda e: e["ts"])
    return {
        "rank": rank, "trace": str(path),
        "device_events_without_unique_launch": len(work) - len(launches),
        "microbatches": [summarize(index, micro, scopes, work, launches, sync)
                        for index, micro in enumerate(micros, 1)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="verified EP8 all-four diagnostic receipt")
    parser.add_argument("--attempt-root", type=Path, help="local root of the raw attempt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = read_json(args.receipt)
    require((receipt["arm"], receipt["ep"], receipt["efsdp"], receipt["update"], receipt["microbatches_per_rank"])
            == ("all", 8, 1, 11, 3) and set(receipt["traces_by_rank"]) == {"0", "1"},
            "Expected the EP8 all-four update-11 diagnostic with three microbatches each on ranks 0 and 1")
    attempt = (args.attempt_root or Path(receipt["attempt"])).resolve(strict=True)
    ranks = []
    for rank, pin in sorted(receipt["traces_by_rank"].items()):
        path = (attempt / pin["path"]).resolve(strict=True)
        require(path.is_relative_to(attempt) and digest(path) == pin["sha256"], f"Trace path or hash differs for rank {rank}")
        ranks.append(analyze_trace(path, int(rank), receipt["update"], receipt["microbatches_per_rank"]))
    result = {
        "receipt": str(args.receipt), "attempt": str(attempt), "source_sha256": receipt["source_sha256"],
        "limits": [
            "Read model-specific CPU scopes and unique Kineto ac2g links before attributing GPU launches.",
            "The CUDA API list preserves raw arguments. Without event handles, record-to-wait edges are unobservable.",
            "A host API launch earlier than a GPU boundary excludes a newly submitted launch as the immediate gap cause. "
            "It does not identify an event dependency or establish a critical path.",
            "An interval without recorded work on the compute stream can contain device waits or work that Kineto omitted.",
        ],
        "ranks": ranks,
    }
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1) from error
