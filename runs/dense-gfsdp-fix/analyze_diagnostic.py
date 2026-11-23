#!/usr/bin/env python3
"""Read the native GFSDP diagnostic traces and compare the matched MTP arms."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import sys

from campaign import digest, read_json, require


MTP = "mtp.layers.0"
FFN = f"{MTP}.block.feed_forward"
LAYER7 = "layers.7[fsdp]"
HEAD = "lm_head"
NORM = "norm"
GROUPS = (MTP, FFN, LAYER7, HEAD, NORM)
GPU = {"kernel", "gpu_memcpy", "gpu_memset"}
HOST = {"cuda_runtime", "cuda_driver"}
COMPILED = "CompiledFunctionBackward"
COMPILED_ENGINE = f"autograd::engine::evaluate_function: {COMPILED}"
COLLECTIVE = re.compile(r"^GFSDP::(all_gather|post_backward_reduce) \((.*)\)$")
TRANSPORT = re.compile(r"nccl|deep_?ep|nvshmem|(?:inter|intra)node|_nvl_kernel|hierarchical_(?:dispatch|combine)", re.I)


def end(event):
    return event["ts"] + event["dur"]


def canonical(name):
    return name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


def key(event):
    return event.get("pid"), event.get("tid"), event["ts"]


def thread(event):
    return event.get("pid"), event.get("tid")


def contains(scope, event):
    return thread(scope) == thread(event) and scope["ts"] <= event["ts"] < end(scope)


def intervals(events):
    return [(event["ts"], end(event)) for event in events]


def merged(items):
    result = []
    for start, stop in sorted(items):
        if stop <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], stop))
        else:
            result.append((start, stop))
    return result


def union_ms(items):
    return sum(stop - start for start, stop in merged(items)) / 1000


def overlap_ms(a, b):
    return union_ms((max(m, p), min(n, q)) for m, n in merged(a) for p, q in merged(b))


def relative(event, origin, when="start"):
    return None if event is None else ((event["ts"] if when == "start" else end(event)) - origin) / 1000


def offset_ms(a, b, a_when="start", b_when="end"):
    if a is None or b is None:
        return None
    left = a["ts"] if a_when == "start" else end(a)
    right = b["ts"] if b_when == "start" else end(b)
    return (left - right) / 1000


def local_compute(event):
    return (event["cat"] == "kernel" and not event.get("args", {}).get("Collective name")
            and not TRANSPORT.search(event["name"]))


def read_trace(path, rank, update, microbatch_count):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    events = trace["traceEvents"]
    recorded_rank = trace.get("distributedInfo", {}).get("rank")
    require(recorded_rank is None or recorded_rank == rank, f"{path} records rank {recorded_rank}")
    scopes, runtime, work, flows = [], defaultdict(list), [], defaultdict(set)
    for event in events:
        category, phase = event.get("cat"), event.get("ph")
        if category == "ac2g" and phase in ("s", "f") and "id" in event:
            flows[phase, event["id"]].add(key(event))
        elif phase == "X" and category in HOST:
            runtime[event.get("args", {}).get("correlation")].append(event)
        elif phase == "X" and category in GPU:
            work.append(event)
        elif phase == "X" and category in ("user_annotation", "cpu_op"):
            name = canonical(event["name"])
            if (name.startswith(("task.profile_update.", "GFSDP::", "FSDP::post_backward_accumulate"))
                    or name in (COMPILED, COMPILED_ENGINE)):
                scopes.append(event | {"name": name})
    del trace, events
    marker = f"task.profile_update.{update}.microbatch.rank{rank}"
    micros = sorted((s for s in scopes if s["name"] == marker), key=lambda s: s["ts"])
    require(len(micros) == microbatch_count, f"{path}: expected {microbatch_count} CPU microbatch markers")
    require(all(end(a) <= b["ts"] for a, b in zip(micros, micros[1:])),
            "Overlapping CPU microbatch markers")
    starts = [micro["ts"] for micro in micros]
    per_micro = [[] for _ in micros]
    unjoined = unjoined_collectives = 0
    for device in work:
        correlation = device.get("args", {}).get("correlation")
        launches = [host for host in runtime.get(correlation, ())
                    if key(host) in flows.get(("s", correlation), ())]
        if len(launches) != 1 or key(device) not in flows.get(("f", correlation), ()):
            unjoined += 1
            unjoined_collectives += bool(device.get("args", {}).get("Collective name"))
            continue
        launch = launches[0]
        index = bisect_right(starts, launch["ts"]) - 1
        if index >= 0 and launch["ts"] < end(micros[index]):
            per_micro[index].append((device, launch))
    compute = merged(intervals(event for event in work if local_compute(event)))
    return {
        "rank": rank,
        "path": str(path),
        "trace_device_events": len(work),
        "trace_device_events_without_unique_runtime_flow": unjoined,
        "trace_collective_kernels_without_unique_runtime_flow": unjoined_collectives,
        "microbatches": [
            summarize_micro(i, micro, sorted(pairs, key=lambda p: p[0]["ts"]),
                            [s for s in scopes if micro["ts"] <= s["ts"] < end(micro)],
                            compute, unjoined_collectives == 0)
            for i, (micro, pairs) in enumerate(zip(micros, per_micro), 1)
        ],
    }


def detached_chunks(scopes):
    head_bwd = sorted((s for s in scopes if s["name"] == f"GFSDP::pre_backward ({HEAD})"),
                      key=lambda s: s["ts"])
    head_fwd = [s for s in scopes if s["name"] == f"GFSDP::pre_forward ({HEAD})"]
    mtp_bwd = [s for s in scopes if s["name"] in {
        f"GFSDP::pre_backward ({MTP})", f"GFSDP::pre_backward ({FFN})",
    }]
    if len(head_bwd) != 2 or not mtp_bwd:
        return []
    first, second = [s["ts"] for s in head_bwd]
    next_forwards = [s["ts"] for s in head_fwd if first < s["ts"] < second]
    decoder = min(s["ts"] for s in mtp_bwd)
    if len(next_forwards) != 1 or decoder <= second:
        return []
    return [(first, next_forwards[0]), (second, decoder)]


def semantic_collectives(pairs, scopes, chunks, origin, compute, *, groups=GROUPS):
    result = defaultdict(list)
    scope_owners = [s for s in scopes if COLLECTIVE.fullmatch(s["name"])]
    predecessors = [s for s in scopes if ("_prefetch for " in s["name"]
                                           or s["name"].startswith(("GFSDP::pre_forward", "GFSDP::pre_backward")))]
    for event, launch in pairs:
        metadata = event.get("args", {})
        collective = metadata.get("Collective name", "").lower().replace("_", "")
        if event["cat"] != "kernel" or not collective or not TRANSPORT.search(event["name"]):
            continue
        enclosing = [scope for scope in scope_owners if contains(scope, launch)]
        if len(enclosing) != 1:
            continue
        operation, group = COLLECTIVE.fullmatch(enclosing[0]["name"]).groups()
        if group not in groups:
            continue
        if (operation == "all_gather" and "allgather" not in collective
                or operation == "post_backward_reduce" and "reducescatter" not in collective):
            continue
        parents = [s["name"] for s in predecessors if contains(s, launch)]
        if any("forward_prefetch for " in name and "backward_prefetch for " not in name
               or name.startswith("GFSDP::pre_forward") for name in parents):
            phase = "forward"
        elif any("backward_prefetch for " in name or name.startswith("GFSDP::pre_backward")
                 for name in parents):
            phase = "backward"
        else:
            phase = "unclassified" if operation == "all_gather" else "backward"
        chunk = next((idx for idx, (start, stop) in enumerate(chunks, 1)
                      if start <= launch["ts"] < stop and phase == "backward"), None)
        span = intervals([event])
        record = {
            "start_ms": relative(event, origin),
            "end_ms": relative(event, origin, "end"),
            "launch_ms": relative(launch, origin),
            "duration_ms": event["dur"] / 1000,
            "stream": metadata.get("stream", event["tid"]),
            "phase": phase,
            "prefetch": any("_prefetch for " in name for name in parents),
            "detached_head_chunk": chunk,
            "input_elements": metadata.get("In msg nelems"),
            "output_elements": metadata.get("Out msg nelems"),
            "dtype": metadata.get("dtype"),
            "group_size": metadata.get("Group size"),
            "group_ranks": metadata.get("Process Group Ranks"),
            "process_group": metadata.get("Process Group Name"),
            "sequence": metadata.get("Seq"),
            "communication_id": metadata.get("Comms Id"),
            "uncovered_by_classified_local_compute_ms": max(
                0, event["dur"] / 1000 - overlap_ms(span, compute)
            ) if operation == "all_gather" else None,
            "host_context": parents,
        }
        result[operation, group].append((event, record))
    return result


def repeat_summary(gathers, chunks, group_observed, complete_coverage):
    if not chunks:
        return {"status": "unobservable", "reason": "two detached head backward windows not established"}
    if not group_observed:
        return {"status": "unobservable", "reason": "the GFSDP group has no pre-backward scope"}
    by_payload = defaultdict(lambda: defaultdict(dict))
    incomplete = not complete_coverage
    occurrences = {1: [], 2: []}
    for _, event in gathers:
        if event["phase"] != "backward" or event["dtype"] != "BFloat16" or event["detached_head_chunk"] is None:
            continue
        occurrences[event["detached_head_chunk"]].append(event)
        payload = tuple(event[k] for k in ("dtype", "input_elements", "output_elements", "group_size", "process_group"))
        ranks = event["group_ranks"]
        payload += (tuple(ranks) if isinstance(ranks, list) else ranks,)
        sequence = event["sequence"]
        if None in payload or sequence is None:
            incomplete = True
            continue
        by_payload[payload][event["detached_head_chunk"]].setdefault(sequence, []).append(event)
    repeated = []
    for payload, by_chunk in by_payload.items():
        if set(by_chunk) != {1, 2}:
            continue
        first_sequences, second_sequences = set(by_chunk[1]), set(by_chunk[2])
        if first_sequences & second_sequences:
            incomplete = True
            continue
        repeated.append({
            "payload": dict(zip(("dtype", "input_elements", "output_elements", "group_size",
                                 "process_group", "group_ranks"), payload)),
            "first_chunk_sequences": sorted(first_sequences),
            "second_chunk_sequences": sorted(second_sequences),
            "second_chunk_kernels": [event for kernels in by_chunk[2].values() for event in kernels],
        })
    return {
        "status": "observed" if repeated else ("unobservable" if incomplete else "not_observed"),
        "reason": "some BF16 gather metadata or collective launch attribution unavailable" if incomplete else None,
        "repeated_payloads": repeated,
        "gathers_by_detached_head_chunk": occurrences,
    }


def layer7_compute(scopes, pairs):
    pre = [scope for scope in scopes if scope["name"] == f"GFSDP::pre_backward ({LAYER7})"]
    if len(pre) != 1:
        return [], "unique layer-7 dense pre-backward scope not established"
    engine = [scope for scope in scopes if scope["name"] == COMPILED_ENGINE and contains(scope, pre[0])]
    if len(engine) != 1:
        return [], "layer-7 pre-backward not enclosed by a unique compiled autograd node"
    compiled = [scope for scope in scopes if scope["name"] == COMPILED
                and contains(engine[0], scope) and scope["ts"] >= end(pre[0])]
    if len(compiled) != 1:
        return [], "unique compiled backward inside the same autograd node not established"
    kernels = [event for event, launch in pairs if local_compute(event) and contains(compiled[0], launch)]
    return kernels, None if kernels else "no runtime-flow-owned layer-7 local kernels"


def first(events):
    return min(events, key=lambda e: e["ts"], default=None)


def last(events):
    return max(events, key=end, default=None)


def summarize_micro(index, micro, pairs, scopes, compute, complete_coverage):
    origin = micro["ts"]
    chunks = detached_chunks(scopes)
    collectives = semantic_collectives(pairs, scopes, chunks, origin, compute)

    def select(operation, group, phase="backward"):
        return [device for device, receipt in collectives[operation, group] if receipt["phase"] == phase]

    parent_ag = select("all_gather", MTP)
    ffn_ag = select("all_gather", FFN)
    layer7_ag = select("all_gather", LAYER7)
    head_rs = select("post_backward_reduce", HEAD)
    ffn_rs = select("post_backward_reduce", FFN)
    parent_rs = select("post_backward_reduce", MTP)
    pack_scopes = [s for s in scopes if s["name"] == f"FSDP::post_backward_accumulate ({NORM})"]
    pack = [e for e, launch in pairs if any(contains(s, launch) for s in pack_scopes)]
    layer7, compute_missing = layer7_compute(scopes, pairs)
    return {
        "microbatch": index,
        "host_duration_ms": micro["dur"] / 1000,
        "joined_device_events": len(pairs),
        "detached_head_chunk_boundaries_ms": [[(a-origin)/1000, (b-origin)/1000] for a, b in chunks] or None,
        "collectives": {f"{operation}:{group}": [record for _, record in rows]
                        for (operation, group), rows in collectives.items()},
        "mtp_parent_and_ffn_ag_before_head_rs": (
            end(last(parent_ag + ffn_ag)) <= first(head_rs)["ts"] if parent_ag and ffn_ag and head_rs else None
        ),
        "mtp_parent_last_ag_end_minus_head_rs_start_ms": offset_ms(last(parent_ag), first(head_rs), "end", "start"),
        "mtp_ffn_last_ag_end_minus_head_rs_start_ms": offset_ms(last(ffn_ag), first(head_rs), "end", "start"),
        "layer7_dense_ag_start_minus_mtp_parent_rs_start_ms": offset_ms(first(layer7_ag), first(parent_rs), "start", "start"),
        "layer7_dense_ag_start_minus_mtp_ffn_rs_end_ms": offset_ms(first(layer7_ag), last(ffn_rs)),
        "norm_pack_first_ms": relative(first(pack), origin),
        "norm_pack_start_minus_mtp_ffn_rs_end_ms": offset_ms(first(pack), last(ffn_rs)),
        "norm_pack_start_minus_mtp_parent_rs_end_ms": offset_ms(first(pack), last(parent_rs)),
        "mtp_ffn_exact_lease_release": {
            "status": "unobservable",
            "reason": "trace lacks buffer ownership and CUDA event-handle identity; RS end is a device timestamp",
        },
        "layer7_compiled_backward": {
            "status": "observed" if layer7 else "unobservable",
            "reason": compute_missing,
            "first_ms": relative(first(layer7), origin),
            "local_kernel_count": len(layer7),
            "local_compute_union_ms": union_ms(intervals(layer7)) if layer7 else None,
            "mtp_parent_rs_overlap_ms": overlap_ms(intervals(layer7), intervals(parent_rs)) if layer7 and parent_rs else None,
            "start_minus_mtp_parent_rs_end_ms": offset_ms(first(layer7), last(parent_rs)),
        },
        "same_bf16_ag_in_both_detached_head_chunks": {
            group: repeat_summary(
                collectives["all_gather", group], chunks,
                any(scope["name"] == f"GFSDP::pre_backward ({group})" for scope in scopes),
                complete_coverage,
            )
            for group in (MTP, FFN)
        },
    }


def receipt_results(path, attempt_override):
    receipt = read_json(path)
    attempt = attempt_override or Path(receipt["attempt"])
    require(receipt["microbatches_per_rank"] == 3, f"{path}: unexpected microbatch count")
    rank_rows = []
    for rank_string, record in sorted(receipt["traces_by_rank"].items()):
        trace = attempt / record["path"]
        require(digest(trace) == record["sha256"], f"Trace digest differs for {trace}")
        rank_rows.append(read_trace(trace, int(rank_string), receipt["update"], receipt["microbatches_per_rank"]))
    return {key: receipt[key] for key in ("arm", "ep", "efsdp", "update", "source_sha256", "input_hashes")} | {
        "diagnostic_receipt": str(path),
        "ranks": rank_rows,
    }


def comparisons(arms):
    comparisons = []
    for ep in sorted({arm["ep"] for arm in arms}):
        by_name = {arm["arm"]: arm for arm in arms if arm["ep"] == ep}
        if not {"mtp", "mtp-transition"} <= by_name.keys():
            continue
        control, transition = (by_name[key] for key in ("mtp", "mtp-transition"))
        require(control["update"] == transition["update"] and control["input_hashes"] == transition["input_hashes"],
                f"EP{ep}: MTP-only and transition receipts differ in input hashes or update")
        left = {(rank["rank"], micro["microbatch"]): micro for rank in control["ranks"] for micro in rank["microbatches"]}
        right = {(rank["rank"], micro["microbatch"]): micro for rank in transition["ranks"] for micro in rank["microbatches"]}
        for rank, micro in sorted(left.keys() & right.keys()):
            a, b = left[rank, micro], right[rank, micro]
            fields = ("mtp_parent_and_ffn_ag_before_head_rs", "mtp_ffn_last_ag_end_minus_head_rs_start_ms",
                      "layer7_dense_ag_start_minus_mtp_ffn_rs_end_ms", "norm_pack_start_minus_mtp_parent_rs_end_ms")
            comparisons.append({
                "ep": ep, "rank": rank, "microbatch": micro,
                **{key: [a[key], b[key]] for key in fields},
                "layer7_compute_mtp_parent_rs_overlap_ms": [
                    row["layer7_compiled_backward"]["mtp_parent_rs_overlap_ms"] for row in (a, b)
                ],
            })
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--receipt", action="append", type=Path, help="diagnostic receipt, repeat for matched arms")
    modes.add_argument("--trace", type=Path, help="analyze one raw trace without claiming verified provenance")
    parser.add_argument("--attempt-root", action="append", type=Path, help="local trace root for each receipt, in argument order")
    parser.add_argument("--rank", type=int, default=0, help="standalone trace rank")
    parser.add_argument("--update", type=int, default=11, help="standalone trace update")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.receipt:
        require(not args.attempt_root or len(args.attempt_root) == len(args.receipt),
                "Specify one local attempt root per diagnostic receipt")
        roots = args.attempt_root or [None] * len(args.receipt)
        arms = [receipt_results(path, root) for path, root in zip(args.receipt, roots, strict=True)]
        require(len(arms) == len({(row["ep"], row["arm"]) for row in arms}), "Duplicate topology and arm receipts")
        evidence = {"arms": arms, "mtp_only_then_transition": comparisons(arms)}
    else:
        require(not args.attempt_root, "--attempt-root requires --receipt")
        evidence = {"standalone_trace": read_trace(args.trace, args.rank, args.update, 3)}
    output = {
        "scope": "Three profiled microbatches per selected rank. No unprofiled throughput, end-to-end critical path or causal wait claim.",
        "attribution": "Device work joins unique Kineto ac2g endpoints and runtime correlation; GFSDP ownership uses containing CPU scopes.",
        "exposure": "AG device spans without concurrent classified local kernels are a screening signal. They do not prove blocking. Detached chunk indices use the two head pre-backward windows and the first MTP decoder backward scope.",
        **evidence,
    }
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
