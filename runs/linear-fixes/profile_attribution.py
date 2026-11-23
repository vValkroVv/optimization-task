"""Summarize one complete native Kineto update without flattening overlapping work.

Inputs are read-only. Run one rank at a time to bound memory. Device ownership
comes from its correlated CUDA API launch, while elapsed/union/sum remain distinct.
"""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re


DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
HOST_CATEGORIES = {"cuda_runtime", "cuda_driver"}
ATTENTION_ROOTS = {
    "torchtitan::gdn_chunk",
    "torchtitan::fla_vjp",
    "torchtitan::gdn_fla_backward",
    "torchtitan::gdn_short_conv_backward",
    "torchtitan::gdn_output_norm_backward",
    "torchtitan::gdn_output_norm_quantize",
}


def union(intervals):
    end = float("-inf")
    length = 0.0
    for start, stop in sorted(intervals):
        length += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return length


def family(event):
    name = event["name"].lower()
    if "ag_nvl_kernel" in name:
        return "ep_aux_allgather_nvl"
    if "rs_nvl_kernel" in name:
        return "ep_aux_reducescatter_nvl"
    if "hybrid_ep::device_sync_kernel" in name:
        return "ep_peer_sync"
    if "hybrid_ep::dispatch_kernel" in name:
        return "ep_dispatch"
    if "hybrid_ep::combine_kernel" in name:
        return "ep_combine"
    if "hybrid_ep::" in name:
        return "ep_other"
    if "nccldevkernel_allgather" in name:
        return "nccl_allgather"
    if "nccldevkernel_reducescatter" in name:
        return "nccl_reducescatter"
    if "nccldevkernel_allreduce" in name:
        return "nccl_allreduce"
    if "nccldevkernel" in name:
        return "nccl_other"
    if event["cat"] != "kernel":
        return event["cat"]
    for token, label in (
        ("_mla_rope_pack", "mla_rope_pack"),
        ("_mla_rope_unpack", "mla_rope_unpack"),
        ("_gdn_short_conv_backward_kernel", "gdn_short_conv_bwd"),
        ("causal_conv1d_bwd_kernel", "gdn_native_conv_bwd"),
        ("_qkv_bwd_pack_kernel", "gdn_native_qkv_pack_bwd"),
        ("_gdn_output_norm_quantize", "gdn_output_norm_fp8"),
        ("_gdn_output_norm_backward", "gdn_output_norm_bwd"),
        ("_gdn_output_norm_weight_grad", "gdn_output_norm_weight_grad"),
        ("layer_norm_gated_bwd_kernel", "gdn_native_norm_bwd"),
        ("gdn_gate_bwd_kernel", "gdn_gate_bwd"),
        ("gdn_gate_chunk_cumsum", "gdn_gate_cumsum"),
    ):
        if token in name:
            return label
    return "other_kernel"


def intervals(events):
    return [(event["ts"], event["ts"] + event["dur"]) for event in events]


def summarize(events):
    if not events:
        return {"count": 0, "duration_sum_us": 0.0, "busy_union_us": 0.0, "span_us": 0.0}
    earliest = min(event["ts"] for event in events)
    latest = max(event["ts"] + event["dur"] for event in events)
    return {
        "count": len(events),
        "duration_sum_us": sum(event["dur"] for event in events),
        "busy_union_us": union(intervals(events)),
        "span_us": latest - earliest,
        "first_ts_us": earliest,
        "last_end_ts_us": latest,
    }


def owner_lookup(annotations):
    by_thread = defaultdict(list)
    for event in annotations:
        by_thread[event["pid"], event["tid"]].append(event)
    starts = {}
    for thread, events in by_thread.items():
        events.sort(key=lambda event: event["ts"])
        starts[thread] = [event["ts"] for event in events]

    def find(host):
        thread = (host["pid"], host["tid"])
        events = by_thread.get(thread, ())
        index = bisect_right(starts.get(thread, ()), host["ts"]) - 1
        while index >= 0:
            event = events[index]
            if event["ts"] <= host["ts"] < event["ts"] + event["dur"]:
                return event["name"]
            index -= 1
        return None

    return find


def analyze(path, rank):
    open_trace = gzip.open if path.suffix == ".gz" else open
    with open_trace(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    update_markers = [event for event in events if event.get("ph") == "X"
                      and event.get("cat") == "user_annotation"
                      and re.fullmatch(rf"task\.profile_update\.\d+\.microbatch\.rank{rank}", event.get("name", ""))]
    update_markers.sort(key=lambda event: event["ts"])
    if len(update_markers) != 32:
        raise ValueError(f"rank {rank}: expected 32 full microbatches, found {len(update_markers)}")
    update = update_markers[0]["name"].split(".")[2]
    if any(event["name"] != update_markers[0]["name"] for event in update_markers):
        raise ValueError("mixed profiled update numbers")
    optimizer = [event for event in events if event.get("ph") == "X"
                 and event.get("cat") == "user_annotation"
                 and event.get("name") == f"task.profile_update.{update}.optimizer_step.rank{rank}"]
    if len(optimizer) != 1:
        raise ValueError(f"rank {rank}: expected one optimizer marker, found {len(optimizer)}")
    step = [event for event in events if event.get("ph") == "X"
            and event.get("cat") == "user_annotation"
            and event.get("name") == f"task.profile_update.{update}.train_step.rank{rank}"]
    if len(step) != 1:
        raise ValueError(f"rank {rank}: expected one train-step marker, found {len(step)}")
    cpu_annotations = [event for event in events if event.get("ph") == "X"
                       and event.get("cat") in {"cpu_op", "user_annotation"}]
    attention_roots = [event for event in cpu_annotations if event["name"] in ATTENTION_ROOTS]
    fsdp_roots = [event for event in cpu_annotations
                  if event["name"].startswith("GFSDP::all_gather (")
                  or event["name"].startswith("GFSDP::post_backward_reduce (")]
    attention_owner = owner_lookup(attention_roots)
    fsdp_owner = owner_lookup(fsdp_roots)
    host_by_correlation = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in HOST_CATEGORIES:
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                host_by_correlation[correlation].append(event)
    devices = [event for event in events if event.get("ph") == "X"
               and event.get("cat") in DEVICE_CATEGORIES]
    starts = [event["ts"] for event in update_markers]
    linked = defaultdict(list)
    fsdp = defaultdict(list)
    attention = defaultdict(list)
    families = defaultdict(list)
    sync_by_microbatch = defaultdict(list)
    issues = Counter()
    host_waits = defaultdict(lambda: Counter())
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in HOST_CATEGORIES:
            continue
        if event.get("name") not in {"cudaStreamSynchronize", "cudaDeviceSynchronize",
                                     "cudaEventSynchronize", "cudaStreamWaitEvent", "cudaEventQuery"}:
            continue
        index = bisect_right(starts, event["ts"]) - 1
        if index >= 0 and event["ts"] < update_markers[index]["ts"] + update_markers[index]["dur"]:
            host_waits[index + 1][event["name"]] += event["dur"]
    for device in devices:
        hosts = [host for host in host_by_correlation.get(device.get("args", {}).get("correlation"), ())
                 if host["ts"] <= device["ts"] + 1]
        if len(hosts) != 1:
            issues[f"host_links_{len(hosts)}"] += 1
            continue
        host = hosts[0]
        index = bisect_right(starts, host["ts"]) - 1
        marker = update_markers[index] if index >= 0 else None
        if marker is not None and host["ts"] < marker["ts"] + marker["dur"]:
            linked[index + 1].append(device)
            label = family(device)
            families[label].append(device)
            root = fsdp_owner(host)
            if root and label.startswith("nccl"):
                group = "expert" if "[efsdp]" in root else "dense"
                fsdp[group, root.split("::", 1)[1].split(" (")[0], label].append(device)
            attention_root = attention_owner(host)
            if attention_root:
                attention[attention_root].append(device)
            if label == "ep_peer_sync":
                sync_by_microbatch[index + 1].append(device)
        elif optimizer[0]["ts"] <= host["ts"] < optimizer[0]["ts"] + optimizer[0]["dur"]:
            linked["optimizer"].append(device)
        else:
            issues["outside_microbatch_optimizer"] += 1
    if any(len(linked[index]) == 0 for index in range(1, 33)):
        raise ValueError("at least one microbatch has no linked device work")
    family_results = {label: summarize(group) for label, group in sorted(families.items())}
    compute = [device for label, group in families.items()
               if label not in {"gpu_memcpy", "gpu_memset"}
               and not label.startswith(("ep_", "nccl_")) for device in group]
    communication = [device for label, group in families.items()
                     if label.startswith(("ep_", "nccl_")) for device in group]
    copies_and_memsets = families.get("gpu_memcpy", ()) + families.get("gpu_memset", ())
    all_microbatch_device = [device for index in range(1, 33) for device in linked[index]]
    fsdp_nccl_count = sum(len(group) for group in fsdp.values())
    nccl_count = sum(len(group) for label, group in families.items() if label.startswith("nccl_"))
    full_update_device = all_microbatch_device + linked["optimizer"]
    return {
        "rank": rank,
        "update": int(update),
        "source": str(path),
        "coverage": {
            "microbatches": len(update_markers),
            "optimizer_markers": len(optimizer),
            "device_categories": dict(Counter(event["cat"] for event in devices)),
            "link_issues": dict(issues),
            "linked_in_microbatches": len(all_microbatch_device),
            "linked_in_optimizer": len(linked["optimizer"]),
            "nccl_with_fsdp_owner": fsdp_nccl_count,
            "nccl_without_fsdp_owner": nccl_count - fsdp_nccl_count,
        },
        "microbatches": [
            {"ordinal": index, "cpu_span_us": marker["dur"], "cpu_start_ts_us": marker["ts"],
             "device": summarize(linked[index]),
             "ep_peer_sync": summarize(sync_by_microbatch[index]),
             "host_wait_api_duration_us": dict(host_waits[index]),
             "ep_peer_sync_intervals_us": sorted(intervals(sync_by_microbatch[index]))}
            for index, marker in enumerate(update_markers, 1)
        ],
        "optimizer": {"cpu_span_us": optimizer[0]["dur"], "device": summarize(linked["optimizer"])},
        "full_microbatch_device": summarize(all_microbatch_device),
        "full_update_device": summarize(full_update_device),
        "train_step_cpu_span_us": step[0]["dur"],
        "compute_union_us": union(intervals(compute)),
        "copy_memset_union_us": union(intervals(copies_and_memsets)),
        "communication_union_us": union(intervals(communication)),
        "compute_communication_overlap_us": (
            union(intervals(compute)) + union(intervals(communication))
            - union(intervals(compute + communication))
        ),
        "exposed_communication_us": (
            union(intervals(compute + communication)) - union(intervals(compute))
        ),
        "families": family_results,
        "fsdp_owned_nccl": {"/".join(key): summarize(group) for key, group in sorted(fsdp.items())},
        "attention_root_device": {label: summarize(group) for label, group in sorted(attention.items())},
        "attention_root_cpu": {label: summarize(group) for label, group in
                               sorted(((label, [root for root in attention_roots if root["name"] == label])
                                       for label in ATTENTION_ROOTS))},
        "limits": "Launch-correlated device ownership and interval unions. Sync kernels include device-side peer waits. Unowned generic kernels and unmatched launches remain unassigned. Single traced update cannot estimate unprofiled throughput or establish causal residuals.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.trace, args.rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "coverage": result["coverage"],
                      "families": {key: value["count"] for key, value in result["families"].items()}}))


if __name__ == "__main__":
    main()
