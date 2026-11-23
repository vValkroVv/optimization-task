"""Validate the A–B–B–A receipts and summarize the component comparisons."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


CASES = ("norm-residual", "dense-fp8", "sac-plain", "sac-residual")
TIMES = ("forward_ms", "backward_ms", "wall_ms", "host_submission_ms", "wall_drain_ms")


def scalar_summary(a, b, key, reduce):
    rounds_a = [reduce(sample[key] for sample in row["samples"]) for row in a]
    rounds_b = [reduce(sample[key] for sample in row["samples"]) for row in b]
    value_a = reduce(sample[key] for row in a for sample in row["samples"])
    value_b = reduce(sample[key] for row in b for sample in row["samples"])
    spread = max(max(rounds_a) - min(rounds_a), max(rounds_b) - min(rounds_b))
    brackets = [right - left for left, right in zip(rounds_a, rounds_b, strict=True)]
    return {
        "A": value_a,
        "B": value_b,
        "delta": value_b - value_a,
        "delta_pct": 100 * (value_b - value_a) / value_a,
        "A_rounds": rounds_a,
        "B_rounds": rounds_b,
        "maximum_within_arm_spread": spread,
        "bracket_deltas": brackets,
        "resolved_beyond_repeat_spread": abs(value_b - value_a) > spread
        and min(brackets) * max(brackets) > 0,
    }


def summarize_case(directory, case):
    paths = [directory / f"{i}-{arm}.json" for i, arm in enumerate("ABBA", start=1)]
    rows = [json.loads(path.read_text()) for path in paths]
    baseline, fixed, repeated_fixed, repeated_baseline = rows
    if [row["arm"] for row in rows] != list("ABBA") or any(
        row["case"] != case for row in rows
    ):
        raise RuntimeError(f"wrong experiment sequence in {directory}")
    if any(row["warmup_calls"] != 10 or len(row["samples"]) != 30 for row in rows):
        raise RuntimeError(f"wrong warmup or sampling count in {directory}")
    for key in (
        "gpu",
        "cuda_visible_devices",
        "hostname",
        "torch_revision",
        "torch_version",
        "compile",
        "fixture",
        "resolved_config_sha256",
        "deterministic_algorithms",
    ):
        if any(row[key] != baseline[key] for row in rows):
            raise RuntimeError(f"changed {key} within {case}")
    switch_by_case = {
        "norm-residual": "GIFT_CUTE_NORM_FUSE_RESIDUAL",
        "dense-fp8": "GIFT_FP8_QUANTIZE_WEIGHT_TAIL",
        "sac-plain": "GIFT_CUTE_NORM_SAC_STATS_ONLY",
        "sac-residual": "GIFT_CUTE_NORM_SAC_STATS_ONLY",
    }
    switches = baseline["switches"]
    expected_fixed = {**switches, switch_by_case[case]: "1"}
    if (
        switches[switch_by_case[case]] != "0"
        or repeated_baseline["switches"] != switches
        or any(row["switches"] != expected_fixed for row in (fixed, repeated_fixed))
    ):
        raise RuntimeError(f"mismatched A/B switch settings in {case}")
    if any(any(row["allocator_failure_deltas"].values()) for row in rows):
        raise RuntimeError(f"measured allocator failure in {case}")
    for row in (baseline, fixed):
        if (
            not row["trace"]
            or hashlib.sha256(Path(row["trace"]["path"]).read_bytes()).hexdigest()
            != row["trace"]["sha256"]
        ):
            raise RuntimeError(
                f"missing or altered activation trace for {case} {row['arm']}"
            )
    tensors = baseline["output_signatures"]
    if any(row["output_signatures"].keys() != tensors.keys() for row in rows):
        raise RuntimeError(f"different output or gradient tensors in {case}")
    parity = {
        name: all(
            all(
                row["output_signatures"][name][key] == expected[key]
                for key in ("sha256", "shape", "dtype")
            )
            for row in rows
        )
        for name, expected in tensors.items()
    }
    if not all(parity.values()):
        raise RuntimeError(f"non-bitwise output or gradient in {case}: {parity}")
    a = [baseline, repeated_baseline]
    b = [fixed, repeated_fixed]
    return {
        "case": case,
        "gpu_uuid": baseline["cuda_visible_devices"],
        "receipt_paths": [str(path) for path in paths],
        "bitwise_parity": parity,
        **{key: scalar_summary(a, b, key, median) for key in TIMES},
        "total_gpu_gib": scalar_summary(a, b, "total_gpu_gib", max),
        "trace_activation": {row["arm"]: row["trace"] for row in (baseline, fixed)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    print(
        json.dumps(
            [summarize_case(args.directory / case, case) for case in args.cases],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
