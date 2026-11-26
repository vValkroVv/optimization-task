"""Attribute one isolated head replay sample using Kineto launch correlations."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
import sys


PHASE = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE.parents[1] / "full-head"))
from summarise_grid import benchmark


GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
BUSY_CLASSES = {"projection", "dhidden", "dw", "ce_forward", "ce_backward", "ce_consuming",
                "ce_join_copy", "dw_staging", "allgather", "reduce_scatter", "rs_post_add",
                "other_compute", "other_copy", "other_memset"}
GEMMS = {"projection", "dhidden", "dw"}
SYNC_NAMES = {"cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize",
              "cuStreamSynchronize", "cuCtxSynchronize", "cuEventSynchronize"}


def read(path):
    return json.loads(path.read_text())


def end(event):
    return event["ts"] + event["dur"]


def encloses(event, nested):
    return event["tid"] == nested["tid"] and event["ts"] <= nested["ts"] <= end(event)


def union_duration(intervals):
    occupied = 0.0
    end_of_union = float("-inf")
    for start, stop in sorted(intervals):
        if stop > end_of_union:
            occupied += stop - max(start, end_of_union)
            end_of_union = stop
    return occupied


def enclosing(ops, event):
    if event is None:
        return []
    return [op for op in ops[event["tid"]] if encloses(op, event)]


def classify(event, parents, thread_ops):
    name = event["name"]
    ancestry = {op["name"] for op in parents}
    if "ncclDevKernel_AllGather" in name:
        return "allgather"
    if "ncclDevKernel_ReduceScatter" in name:
        return "reduce_scatter"
    if "ncclDevKernel" in name:
        raise ValueError(f"Unexpected NCCL activity inside measured call: {name}")
    if "SplitBackward0" in ancestry and "aten::cat" in ancestry:
        return "ce_join_copy"
    if "torchtitan::exact_folded_ce_consume" in ancestry or "_PrecomputedGradient" in ancestry:
        return "ce_consuming"
    if "CompiledFunctionBackward" in ancestry:
        return "ce_backward"
    if "CompiledFunction" in ancestry:
        return "ce_forward"
    if "FSDP::post_backward_accumulate" in ancestry:
        return "dw_staging"
    if "GFSDP::post_backward_reduce" in ancestry:
        return "rs_post_add"
    if "_StagedHeadLinearBackward" in ancestry:
        return "dhidden" if "aten::mm" in ancestry else "dw"
    if "_StagedHeadLinear" in ancestry or "aten::linear" in ancestry:
        return "projection"
    if "MmBackward0" in ancestry:
        # This transpose-of-weight graph submits the weight-gradient matmul
        # before the input-gradient matmul. Confirm their observed descriptors
        # and kernel names rather than grouping by a tile name alone.
        backward, = [p for p in parents if p["name"] == "MmBackward0"]
        active_mm, = [p for p in parents if p["name"] == "aten::mm"]
        mm_calls = sorted((p for p in thread_ops if p["name"] == "aten::mm" and encloses(backward, p)),
                          key=lambda p: p["ts"])
        if len(mm_calls) != 2:
            raise ValueError(f"Expected two matmuls under MmBackward0, got {len(mm_calls)}")
        return "dw" if active_mm is mm_calls[0] else "dhidden"
    return {"gpu_memcpy": "other_copy", "gpu_memset": "other_memset"}.get(event["cat"], "other_compute")


def profile_trace(path, timing):
    trace = read(path)
    base_ns = trace["baseTimeNanoseconds"]
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    marker, = [event for event in events if event.get("cat") == "user_annotation"
               and event["name"] == "c8_phase_measured_call"]
    start, stop = marker["ts"], end(marker)
    absolute = lambda us: base_ns + round(us * 1000)
    if abs(absolute(start) - timing["start_utc_ns"]) > 200_000:
        raise ValueError(f"Trace marker and timer UTC differ by >0.2 ms: {path}")

    operators = defaultdict(list)
    launches = {}
    waits = defaultdict(list)
    for event in events:
        if event.get("cat") in {"cpu_op", "user_annotation"}:
            operators[event["tid"]].append(event)
        elif event.get("cat") in {"cuda_runtime", "cuda_driver"}:
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation] = event
            if event["name"] in SYNC_NAMES and event["tid"] == marker["tid"] and start <= event["ts"] < stop:
                waits[event["name"]].append(event["dur"])

    gpu = sorted([event for event in events if event.get("cat") in GPU_CATEGORIES
                  and start <= event["ts"] < stop], key=lambda e: e["ts"])
    if not gpu or any(end(event) > stop + 20 for event in gpu):
        raise ValueError(f"Measured marker missing or clips timed CUDA work: {path}")
    groups = defaultdict(list)
    signatures = Counter()
    for event in gpu:
        launch = launches.get(event["args"].get("correlation"))
        parents = enclosing(operators, launch)
        phase = classify(event, parents, operators[launch["tid"]] if launch else [])
        groups[phase].append(event)
        if phase in GEMMS | {"ce_forward", "ce_backward", "ce_consuming", "ce_join_copy",
                             "dw_staging", "allgather", "reduce_scatter"}:
            signature = dict(phase=phase, name=event["name"], category=event["cat"],
                             grid=event["args"].get("grid"), block=event["args"].get("block"),
                             bytes=event["args"].get("bytes"))
            signatures[json.dumps(signature, sort_keys=True)] += 1

    for phase in GEMMS:
        if len([e for e in groups[phase] if e["cat"] == "kernel"]) < 8:
            raise ValueError(f"Missing one of eight {phase} kernel launches: {path}")
    gather, = groups["allgather"]
    scatter, = groups["reduce_scatter"]
    phase_stats = {
        phase: {"sum_ms": sum(e["dur"] for e in groups[phase]) / 1000,
                "union_ms": union_duration((e["ts"], end(e)) for e in groups[phase]) / 1000,
                "events": dict(Counter(e["cat"] for e in groups[phase]))}
        for phase in sorted(BUSY_CLASSES)
    }
    first_gpu, last_gpu = gpu[0]["ts"], max(map(end, gpu))
    busy = union_duration((e["ts"], end(e)) for e in gpu)
    pre_rs_busy = union_duration((e["ts"], min(end(e), scatter["ts"]))
                                 for e in gpu if e["ts"] < scatter["ts"])
    compute = [e for e in gpu if e not in (gather, scatter)]
    nccl_overlap = (gather["dur"] + scatter["dur"]) - (
        busy - union_duration((e["ts"], end(e)) for e in compute)
    )
    last_end = first_gpu
    gaps = []
    for event in gpu:
        if event["ts"] > last_end:
            launch = launches.get(event["args"].get("correlation"))
            gap = event["ts"] - last_end
            gaps.append(dict(ms=gap / 1000, starts_utc_ns=absolute(last_end),
                             next_activity=event["name"],
                             next_correlated_launch_after_gap_start_ms=(
                                 (launch["ts"] - last_end) / 1000 if launch else None)))
        last_end = max(last_end, end(event))
    result = dict(
        kernel_count=sum(e["cat"] == "kernel" for e in gpu),
        memcpy_count=sum(e["cat"] == "gpu_memcpy" for e in gpu),
        memcpy_bytes=sum(e["args"].get("bytes", 0) for e in gpu if e["cat"] == "gpu_memcpy"),
        missing_launch_correlations=sum(e["args"].get("correlation") not in launches for e in gpu),
        first_gpu_utc_ns=absolute(first_gpu),
        last_gpu_utc_ns=absolute(last_gpu),
        gpu_span_ms=(last_gpu - first_gpu) / 1000,
        gpu_busy_union_ms=busy / 1000,
        gpu_internal_gap_ms=(last_gpu - first_gpu - busy) / 1000,
        pre_rs_first_gpu_to_entry_ms=(scatter["ts"] - first_gpu) / 1000,
        pre_rs_active_gpu_union_ms=pre_rs_busy / 1000,
        pre_rs_gpu_internal_gap_ms=(scatter["ts"] - first_gpu - pre_rs_busy) / 1000,
        largest_gpu_gaps=sorted(gaps, key=lambda g: -g["ms"])[:3],
        marker_start_utc_ns=absolute(start),
        marker_stop_utc_ns=absolute(stop),
        nccl_overlap_with_other_activity_ms=nccl_overlap / 1000,
        allgather_start_utc_ns=absolute(gather["ts"]),
        allgather_end_utc_ns=absolute(end(gather)),
        reduce_scatter_start_utc_ns=absolute(scatter["ts"]),
        reduce_scatter_end_utc_ns=absolute(end(scatter)),
        cuda_sync_ms={name: sum(durations) / 1000 for name, durations in waits.items()},
        phases=phase_stats,
    )
    return result, signatures


def series_attempt(attempt):
    invocation = read(attempt / "invocation.json")
    status = read(attempt / "status.json")
    if status["status"] != "success":
        raise ValueError(f"Cannot summarize incomplete attempt: {attempt}")
    bench = benchmark(attempt, invocation)
    ranks = []
    kernel_signatures = defaultdict(dict)
    timing_by_rank = []
    measured_by_rank = []
    physical_indices = invocation.get("physical_gpu_index_by_logical_rank", list(range(8)))
    for rank in range(8):
        path = attempt / f"timing-{rank}.json"
        timing = read(path)
        timer_results = read(attempt / f"rank-{rank}.json")["measured"]
        timing_by_rank.append(timing)
        measured_by_rank.append(timer_results)
        if len(timing) != invocation["samples"]:
            raise ValueError(f"Unexpected timer count in {path}")
        sample_index = int(invocation["settings"]["C8_PHASE_PROFILE_SAMPLE"]) - 1
        if [i for i, sample in enumerate(timing) if sample["profiled"]] != [sample_index]:
            raise ValueError(f"Unexpected profiler sample index in {path}")
        profiled = timing[sample_index]
        result, signatures = profile_trace(attempt / f"trace-{rank}.json", profiled)
        physical_device = attempt / f"physical-device-{rank}.json"
        ranks.append(dict(rank=rank, physical_gpu_index=physical_indices[rank], timing=profiled,
                          physical_device=read(physical_device) if physical_device.is_file() else None,
                          **result))
        for signature, count in signatures.items():
            kernel_signatures[signature][rank] = count

    traced = bench["steps_ms"][sample_index]
    unprofiled_indices = [i for i in range(invocation["samples"]) if i != sample_index]
    unprofiled_slowest_rank_metrics = [
        measured_by_rank[bench["max_wall_ranks"][i]][i] for i in unprofiled_indices
    ]
    gather_finishes = [row["allgather_end_utc_ns"] for row in ranks]
    unprofiled = [value for index, value in enumerate(bench["steps_ms"]) if index != sample_index]
    peak = {}
    for key in ("allocated_bytes.all.peak", "reserved_bytes.all.peak"):
        peak[key] = max(row[key] for timing in timing_by_rank for row in timing if not row["profiled"]) / 2**30
    scatter_order = sorted(ranks, key=lambda row: row["reduce_scatter_start_utc_ns"])
    scatter_finishes = [row["reduce_scatter_end_utc_ns"] for row in ranks]
    routes = read(attempt / "activation-0.json")
    phase_aggregates = {}
    for phase in sorted(BUSY_CLASSES):
        values = [rank["phases"][phase]["sum_ms"] for rank in ranks]
        kernels = [rank["phases"][phase]["events"].get("kernel", 0) for rank in ranks]
        phase_aggregates[phase] = dict(median_ms=median(values), min_ms=min(values), max_ms=max(values),
                                       kernel_counts=sorted(set(kernels)))

    return dict(
        attempt=str(attempt.relative_to(PHASE)), raw_trace_path=str(attempt),
        source_runtime={key: invocation["provenance"][key] for key in (
            "host", "python", "torch", "torch_git_version", "torch_cuda", "nccl_version",
            "baseline_revision", "fixtures_sha256")}, status=status, request={
            key: invocation[key] for key in ("model", "batch", "seq", "chunks", "variant", "pair", "position",
                                              "manifests", "expected_routes", "reference", "reference_sha256")
            if key in invocation
        },
        routes=routes, traced_sample=sample_index + 1, traced_rankmax_ms=traced,
        traced_rankmax_rank=bench["max_wall_ranks"][sample_index],
        rankmax_step_series_ms=bench["steps_ms"],
        sample_windows_utc=[
            dict(sample=index + 1, profiled=index == sample_index,
                 starts_utc_ns=min(samples[index]["start_utc_ns"] for samples in timing_by_rank),
                 stops_utc_ns=max(samples[index]["stop_utc_ns"] for samples in timing_by_rank),
                 physical_gpu_6_start_utc_ns=timing_by_rank[physical_indices.index(6)][index]["start_utc_ns"],
                 physical_gpu_6_stop_utc_ns=timing_by_rank[physical_indices.index(6)][index]["stop_utc_ns"])
            for index in range(invocation["samples"])
        ],
        rankmax_unprofiled_median_ms=median(unprofiled),
        rankmax_pre_trace_median_ms=median(bench["steps_ms"][:sample_index]),
        rankmax_post_trace_median_ms=median(bench["steps_ms"][sample_index + 1:]),
        rankmax_unprofiled_vs_trace_pct=100 * (traced / median(unprofiled) - 1),
        unprofiled_host_submit_ms=median(row["submit_ms"] for row in unprofiled_slowest_rank_metrics),
        unprofiled_host_wait_ms=median(row["wait_ms"] for row in unprofiled_slowest_rank_metrics),
        unprofiled_native_total_gib=max(
            samples[i]["total_gib"] for samples in measured_by_rank for i in unprofiled_indices),
        unprofiled_allocated_reserved_gib=peak,
        retry_counts=bench["allocator_retry_count_process"],
        fixture=read(attempt / "rank-0.json")["initialization"],
        physical_tokens=invocation["seq"], folded_rows=2 * invocation["seq"] // invocation["chunks"],
        valid_tokens=bench["valid_tokens"],
        valid_mtp_tokens=bench["valid_mtp_tokens"], denominator=bench["denominator"],
        ag_exit_spread_ms=(max(gather_finishes) - min(gather_finishes)) / 1e6,
        rs_entry_spread_ms=(scatter_order[-1]["reduce_scatter_start_utc_ns"]
                            - scatter_order[0]["reduce_scatter_start_utc_ns"]) / 1e6,
        rs_exit_spread_ms=(max(scatter_finishes) - min(scatter_finishes)) / 1e6,
        rs_latest_rank=scatter_order[-1]["rank"],
        rs_latest_physical_gpu=scatter_order[-1]["physical_gpu_index"],
        phase_aggregates=phase_aggregates, ranks=ranks,
        kernel_signatures=[{**json.loads(sig), "counts_by_rank": counts}
                           for sig, counts in sorted(kernel_signatures.items())],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", nargs="+", type=Path)
    args = parser.parse_args()
    attempts = sorted(path.parent for series in args.series for path in series.glob("**/status.json"))
    if not attempts:
        parser.error("No completed trace attempts")
    print(json.dumps([series_attempt(attempt.resolve()) for attempt in attempts], indent=2))


if __name__ == "__main__":
    main()
