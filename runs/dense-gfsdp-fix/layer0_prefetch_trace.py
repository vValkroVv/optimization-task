#!/usr/bin/env python3
"""Measure the first FFN backward gather against its launch and consumer scopes."""

import argparse
import json
from pathlib import Path
from statistics import median

import analyze_diagnostic as trace
from campaign import digest, read_json, require, write_json
from mtp_parent_boundary import link_work


PARENT = "layers.0"
FFN = "layers.0.feed_forward"
PREVIOUS = "layers.1[fsdp]"
GROUPS = (PARENT, FFN, PREVIOUS)


def backward_scope(scopes, group):
    pre = [event for event in scopes if event["name"] == f"GFSDP::pre_backward ({group})"]
    require(len(pre) == 1, f"Expected one pre-backward for {group}")
    engine = [event for event in scopes if event["name"] == trace.COMPILED_ENGINE
              and trace.contains(event, pre[0])]
    require(len(engine) == 1, f"Expected a unique compiled autograd node for {group}")
    return engine[0]


def measure_microbatch(micro, all_scopes, work, launches):
    origin = micro["ts"]
    bounds = (origin, trace.end(micro))
    scopes = [event for event in all_scopes if bounds[0] <= event["ts"] < bounds[1]]
    pairs = [(event, launch) for event in work if (launch := launches.get(id(event))) is not None
             and bounds[0] <= launch["ts"] < bounds[1]]
    compute = trace.intervals(event for event, _ in pairs if trace.local_compute(event))
    rows = trace.semantic_collectives(pairs, scopes, (), origin, compute, groups=GROUPS)

    def pick(op, group, phase):
        matches = [(event, row) for event, row in rows[op, group] if row["phase"] == phase]
        require(len(matches) == 1, f"Expected exactly one {phase} {op} for {group}")
        return matches[0]

    engines = {group: backward_scope(scopes, group) for group in GROUPS}
    consumer = [event for event in scopes if event["name"] == trace.COMPILED
                and trace.contains(engines[FFN], event)]
    require(len(consumer) == 1, "Expected one compiled backward for the first FFN")
    first_compute = trace.first([event for event, host in pairs
                                 if trace.local_compute(event) and trace.contains(consumer[0], host)])
    require(first_compute is not None, "No linked FFN consumer kernel")
    ffn, ffn_row = pick("all_gather", FFN, "backward")
    source = [group for group, event in engines.items() if trace.contains(event, launches[id(ffn)])]
    require(len(source) == 1, "Could not identify the FFN gather's source autograd node")
    prior_rs, prior_rs_row = pick("post_backward_reduce", PREVIOUS, "backward")
    require(ffn_row["dtype"] == "BFloat16" and ffn_row["stream"] == prior_rs_row["stream"],
            "Expected the BF16 gather and previous dense reduction on the same NCCL stream")
    local_non_ffn = trace.intervals(event for event in work if event is not ffn)
    predecessor_compute = trace.intervals(
        event for event, host in pairs
        if trace.local_compute(event) and trace.contains(engines[PREVIOUS], host)
    )
    return {
        "ffn_backward_ag": ffn_row,
        "ffn_backward_ag_source": source[0],
        "previous_backward_rs": prior_rs_row,
        "ffn_ag_end_to_consumer_ms": trace.offset_ms(first_compute, ffn),
        "ffn_ag_start_minus_previous_rs_end_ms": trace.offset_ms(ffn, prior_rs),
        "ffn_ag_overlap_previous_backward_compute_ms": trace.overlap_ms(trace.intervals([ffn]), predecessor_compute),
        "ffn_ag_without_other_observed_work_ms": max(
            0.0, ffn_row["duration_ms"] - trace.overlap_ms(trace.intervals([ffn]), local_non_ffn)
        ),
        "forward_ag": {group: pick("all_gather", group, "forward")[1] for group in GROUPS},
    }


def measure(attempt, receipt_path):
    receipt = read_json(receipt_path)
    require((receipt["ep"], receipt["efsdp"], receipt["arm"], receipt["update"],
             receipt["microbatches_per_rank"]) == (8, 1, "all", 11, 3)
            and set(receipt["traces_by_rank"]) == {"0", "1"}, "Expected the matched all-four EP8 profile")
    attempt = attempt.resolve(strict=True)
    records = []
    for rank, pin in sorted(receipt["traces_by_rank"].items()):
        path = (attempt / pin["path"]).resolve(strict=True)
        require(path.is_relative_to(attempt) and digest(path) == pin["sha256"], f"Rank {rank} trace hash differs")
        raw = read_json(path)
        require(str(raw["distributedInfo"]["rank"]) == rank, "Trace rank differs from its receipt")
        events = raw["traceEvents"]
        work, launches = link_work(events)
        scopes = [event | {"name": trace.canonical(event["name"])} for event in events
                  if event.get("ph") == "X" and event.get("cat") in ("user_annotation", "cpu_op")]
        marker = f"task.profile_update.11.microbatch.rank{rank}"
        micros = sorted((event for event in scopes if event["name"] == marker), key=lambda event: event["ts"])
        require(len(micros) == 3 and all(trace.end(left) <= right["ts"] for left, right in zip(micros, micros[1:])),
                f"Incomplete or overlapping rank {rank} microbatches")
        for index, micro in enumerate(micros, 1):
            records.append({"rank": int(rank), "microbatch": index} | measure_microbatch(micro, scopes, work, launches))
    return {
        "receipt": str(receipt_path), "source_sha256": receipt["source_sha256"],
        "reference": receipt["reference"], "input_hashes": receipt["input_hashes"], "features": receipt["features"],
        "rows": records,
        "median_ffn_ag_without_other_observed_work_ms": median(
            row["ffn_ag_without_other_observed_work_ms"] for row in records
        ),
        "scope_limit": "Unique Kineto launch flows and time overlap. CUDA event DAG is unavailable.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(args.attempt, args.receipt)
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
