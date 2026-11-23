"""Validate the isolated Hopper receipts and report paired confidence intervals."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

from run_confidence import GPU_BY_CASE, PAIRS, SAMPLES, SEED, WARMUP
from summarize import TIMES


# Two-sided 95% critical value for the fixed 16-pair campaign.
T_CRITICAL_95 = {15: 2.131449545559323}
MEMORY = "total_gpu_gib"
NUMERIC_SIGNATURE = ("shape", "dtype", "sha256")
RESIDUAL = "GIFT_CUTE_NORM_FUSE_RESIDUAL"
STATS_ONLY = "GIFT_CUTE_NORM_SAC_STATS_ONLY"
WEIGHT_TAIL = "GIFT_FP8_QUANTIZE_WEIGHT_TAIL"
SWITCH_BY_CASE = {
    "norm-residual": RESIDUAL,
    "sac-residual": STATS_ONLY,
    "dense-fp8": WEIGHT_TAIL,
    "sac-plain": STATS_ONLY,
}
COMPILE = {"backend": "inductor", "fullgraph": True, "cudagraphs": False}
FAILURE_COUNTERS = {"num_alloc_retries": 0, "num_ooms": 0}
ENVIRONMENT_KEYS = (
    "hostname",
    "gpu",
    "torch_version",
    "torch_revision",
    "cuda_version",
    "deterministic_algorithms",
)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_plan(raw):
    plan = read_json(raw / "plan.json")
    protocol = {"pairs": PAIRS, "warmup": WARMUP, "samples": SAMPLES, "seed": SEED}
    if any(plan[key] != value for key, value in protocol.items()):
        raise ValueError("unexpected pair count, warmup, sample count, or seed")
    if plan["cases"].keys() != GPU_BY_CASE.keys():
        raise ValueError("unexpected cases in plan.json")
    uuid_by_gpu = {}
    for case, setup in plan["cases"].items():
        gpu = setup["gpu"]
        uuid = setup["uuid"]
        orders = setup["orders"]
        if gpu != GPU_BY_CASE[case] or not uuid.startswith("GPU-"):
            raise ValueError(f"unexpected GPU assignment for {case}")
        if uuid_by_gpu.setdefault(gpu, uuid) != uuid:
            raise ValueError(f"mismatched UUIDs for GPU {gpu}")
        if sorted(orders) != ["AB"] * (PAIRS // 2) + ["BA"] * (PAIRS // 2):
            raise ValueError(f"expected eight AB and eight BA pairs for {case}")
    if len(set(uuid_by_gpu.values())) != len(uuid_by_gpu):
        raise ValueError("physical GPUs in plan.json share a UUID")
    return plan


def expected_switches(case, arm):
    switches = {
        RESIDUAL: str(int(case == "sac-residual")),
        STATS_ONLY: "0",
        WEIGHT_TAIL: "0",
    }
    switches[SWITCH_BY_CASE[case]] = str(int(arm == "B"))
    return switches


def verify_trace(path, trace, arm, case, profiled):
    if not profiled:
        if trace is not None:
            raise ValueError(f"unexpected activation trace in {path}")
        return
    trace_path = path.with_suffix(".trace.json")
    if (
        not trace
        or Path(trace["path"]).name != trace_path.name
        or not trace_path.is_file()
        or sha256(trace_path) != trace["sha256"]
    ):
        raise ValueError(f"missing or altered activation trace for {path}")
    if not all(
        trace[phase]["gpu_work_events"] > 0 for phase in ("forward", "backward")
    ):
        raise ValueError(f"missing GPU work in activation trace for {path}")
    if case == "dense-fp8":
        if trace["kernel_activation"]["weight_tail"] is not (arm == "B"):
            raise ValueError(f"unexpected weight-tail activation in {path}")
    elif not trace["kernel_activation"] or not trace["custom_op_activation"]:
        raise ValueError(f"missing norm activation markers in {path}")


def summarize_process(path, row, expected, profiled):
    for key, value in expected.items():
        if row[key] != value:
            raise ValueError(f"unexpected {key} in {path}")
    if row["allocator_failure_deltas"] != FAILURE_COUNTERS:
        raise ValueError(f"measured allocator failure in {path}")
    if len(row["samples"]) != SAMPLES:
        raise ValueError(f"unexpected sample count in {path}")
    verify_trace(path, row["trace"], row["arm"], row["case"], profiled)

    process = {"receipt": str(path)}
    for key in (*TIMES, MEMORY):
        values = [sample[key] for sample in row["samples"]]
        if any(
            type(value) not in (float, int) or not math.isfinite(value) or value <= 0
            for value in values
        ):
            raise ValueError(f"nonpositive or nonfinite {key} sample in {path}")
        if key == MEMORY:
            process[key] = max(values)
        else:
            center = len(values) // 2
            early = median(values[:center])
            late = median(values[center:])
            process[key] = {
                "median": median(values),
                "early_median": early,
                "late_median": late,
                "late_vs_early_pct": 100 * (late / early - 1),
            }
    return process


def paired_mean_ci(values):
    center = mean(values)
    critical = T_CRITICAL_95[len(values) - 1]
    margin = critical * stdev(values) / math.sqrt(len(values))
    return center, center - margin, center + margin


def describe_effect(estimate, lower, upper):
    direction = "includes_zero"
    if upper < 0:
        direction = "lower"
    elif lower > 0:
        direction = "higher"
    return {"estimate": estimate, "ci95": [lower, upper], "direction": direction}


def summarize_timing(pairs, key):
    a = [pair["A"][key]["median"] for pair in pairs]
    b = [pair["B"][key]["median"] for pair in pairs]
    log_ratios = [math.log(right / left) for left, right in zip(a, b, strict=True)]
    interval = [100 * math.expm1(value) for value in paired_mean_ci(log_ratios)]
    return {
        "A_median_ms": median(a),
        "B_median_ms": median(b),
        "paired_delta_pct": describe_effect(*interval),
        "max_abs_late_vs_early_pct": {
            arm: max(abs(pair[arm][key]["late_vs_early_pct"]) for pair in pairs)
            for arm in "AB"
        },
    }


def summarize_memory(pairs):
    a = [pair["A"][MEMORY] for pair in pairs]
    b = [pair["B"][MEMORY] for pair in pairs]
    differences = [right - left for left, right in zip(a, b, strict=True)]
    return {
        "A_max_gib": max(a),
        "B_max_gib": max(b),
        "observed_max_delta_gib": max(b) - max(a),
        "paired_delta_gib": describe_effect(*paired_mean_ci(differences)),
    }


def numeric_signatures(signatures):
    return {
        name: {key: signature[key] for key in NUMERIC_SIGNATURE}
        for name, signature in signatures.items()
    }


def summarize_case(directory, case, setup):
    paths_by_pair = [
        [
            directory / f"{number:02d}-{position}-{arm}.json"
            for position, arm in enumerate(order, 1)
        ]
        for number, order in enumerate(setup["orders"], 1)
    ]
    expected_paths = {path for paths in paths_by_pair for path in paths}
    actual_paths = {
        path
        for path in directory.glob("*.json")
        if not path.name.endswith(".trace.json")
    }
    if actual_paths != expected_paths:
        missing = sorted(path.name for path in expected_paths - actual_paths)
        extra = sorted(path.name for path in actual_paths - expected_paths)
        raise ValueError(
            f"incomplete receipt sequence for {case}: {missing=}, {extra=}"
        )

    reference = read_json(paths_by_pair[0][0])
    signatures = reference["output_signatures"]
    if "y" not in signatures or len(signatures) <= 1:
        raise ValueError(f"missing output or gradient signatures for {case}")
    reference_values = numeric_signatures(signatures)
    reference_strides = {
        name: signature["stride"] for name, signature in signatures.items()
    }
    environment = {key: reference[key] for key in ENVIRONMENT_KEYS}
    expected = {
        **environment,
        "case": case,
        "cuda_visible_devices": setup["uuid"],
        "fixture": setup["fixture"],
        "resolved_config": setup["resolved_config"],
        "resolved_config_sha256": setup["resolved_config_sha256"],
        "compile": COMPILE,
        "warmup_calls": WARMUP,
    }
    pairs = []
    traces = {}
    stride_differences = []
    for number, (order, paths) in enumerate(
        zip(setup["orders"], paths_by_pair, strict=True), 1
    ):
        pair = {"number": number, "order": order}
        for arm, path in zip(order, paths, strict=True):
            row = read_json(path)
            actual_signatures = row["output_signatures"]
            if numeric_signatures(actual_signatures) != reference_values:
                raise ValueError(f"non-bitwise output or gradient in {path}")
            changed_strides = {
                name: signature["stride"]
                for name, signature in actual_signatures.items()
                if signature["stride"] != reference_strides[name]
            }
            if changed_strides:
                stride_differences.append(
                    {"receipt": str(path), "strides": changed_strides}
                )
            expected_arm = {
                **expected,
                "arm": arm,
                "switches": expected_switches(case, arm),
            }
            pair[arm] = summarize_process(path, row, expected_arm, profiled=number == 1)
            if number == 1:
                traces[arm] = row["trace"]
        pairs.append(pair)

    return {
        "environment": environment,
        "output_signatures": signatures,
        "output_signature_reference": str(paths_by_pair[0][0]),
        "stride_differences": stride_differences,
        "trace_activation": traces,
        "timings": {key: summarize_timing(pairs, key) for key in TIMES},
        MEMORY: summarize_memory(pairs),
        "pairs": pairs,
    }


def summarize(raw):
    plan = read_plan(raw)
    return {
        "plan": plan,
        "method": {
            "unit": "One matched A/B pair of fresh processes on one GPU",
            "timing_aggregation": "Median per process, then median of process medians per arm",
            "timing_effect": "Geometric mean of paired B/A median ratios, as percent change",
            "memory_metric": (
                "GIFT device_capacity_report estimate in GiB using allocator reserved peak "
                "plus report-time non-PyTorch memory"
            ),
            "memory_aggregation": "Maximum measured call per process, then maximum per arm",
            "memory_effect": "Arithmetic mean of paired B minus A process maxima, in GiB",
            "stability": "Compare medians of the first and last halves of measured calls",
            "confidence": (
                "Two-sided 95% Student-t interval using 16 pair effects and 15 degrees of "
                "freedom. Timing uses log(B/A), then converts the estimate and bounds to "
                "percent. Intervals assume independent, approximately normal pair effects. "
                "Intervals are per metric and have no multiplicity correction."
            ),
            "t_critical_95": T_CRITICAL_95[PAIRS - 1],
            "interpretation": (
                "Negative effects favor B. Intervals containing zero leave the direction "
                "unresolved at this confidence level. Results apply to the measured fixtures "
                "and GPUs."
            ),
            "timer_scopes": {
                "forward_ms": "CUDA event forward span including queue gaps",
                "backward_ms": "CUDA event backward span including queue gaps",
                "wall_ms": "Host elapsed time from submission start through CUDA synchronization",
                "host_submission_ms": "Host elapsed time through final CUDA event recording",
                "wall_drain_ms": "Host elapsed time in final CUDA synchronization",
            },
        },
        "cases": {
            case: summarize_case(raw / case, case, setup)
            for case, setup in plan["cases"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.raw), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
