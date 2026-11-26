"""Summarize complete eight-rank LM-head replays and their FP64 comparisons."""

import argparse
import json
import math
from pathlib import Path
import re
from statistics import median
import struct


RANKS = tuple(range(8))
VOCAB = 128256
CASES = {("100b", 2): (2, 4096), ("100b", 8): (2, 4096), ("700b", 8): (3, 7168)}
TIMING_COLUMNS = ("ms", "submit_ms", "wait_ms", "total_gib")
LOSS_KEYS = {"loss_ce": "loss_ce", "loss_mtp": "loss_mtp", "total": "total_loss"}
NUMERICAL_COMPONENTS = ("dhidden", "dweight_shard", *LOSS_KEYS)
ERROR_COLUMNS = ("max_abs", "relative_l2")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_numerics(row):
    values = row["numerics"]
    require(set(values) == {*NUMERICAL_COMPONENTS, "reference_cache"}, "unexpected numerical components")
    require(bool(values["reference_cache"]), "missing FP64 reference cache identity")
    for component in NUMERICAL_COMPONENTS:
        metric = values[component]
        scalar = component in LOSS_KEYS
        require(set(metric) == (set(ERROR_COLUMNS) | ({"actual", "reference"} if scalar else set())),
                f"unexpected numerical columns for {component}")
        for key, value in metric.items():
            require(finite(value) and (key in ("actual", "reference") or value >= 0),
                    f"invalid numerical value for {component}.{key}")
        if scalar:
            raw = row.get("total_loss") if component == "total" else row["components"][LOSS_KEYS[component]]
            require(metric["actual"] == raw and metric["max_abs"] == abs(raw - metric["reference"]),
                    f"inconsistent recorded scalar for {component}")


def load_run(path):
    status = path / "exit-code"
    if not status.is_file():
        return {"status": "missing_exit_code"}, []
    code = int(status.read_text().strip())
    if code:
        return {"status": "failed", "exit_code": code}, []
    invocation = json.loads((path / "invocation.json").read_text())
    require("--nproc-per-node=8" in invocation, "invocation must request eight ranks")

    def option(flag):
        return invocation[invocation.index(flag) + 1]

    model, chunks, variant = option("--model"), int(option("--chunks")), option("--variant")
    observed = {
        int(match.group(1)) for p in path.glob("rank-*.json")
        if (match := re.fullmatch(r"rank-(0|[1-9][0-9]*)\.json", p.name))
    }
    require(observed == set(RANKS) and len(list(path.glob("rank-*.json"))) == len(RANKS),
            f"expected exactly ranks 0 through 7, found {sorted(observed)}")
    samples = int(option("--samples"))
    require(samples > 0, "sample count must be positive")
    batch, width = CASES[model, chunks]
    shape = (2 * batch * 4096 // chunks, width, VOCAB)
    needs_oracle = "--reference" in invocation
    rows = []
    for rank in RANKS:
        row = json.loads((path / f"rank-{rank}.json").read_text())
        require((row["rank"], row["model"], row["chunks"], row["variant"]) == (rank, model, chunks, variant),
                f"rank {rank} identity disagrees with invocation")
        require((row["rows"], row["width"], row["vocab"]) == shape, f"rank {rank} unexpected active shape")
        require(row["smoke_collectives"] == {"gather": 1, "scatter": 1}, f"rank {rank} unexpected collectives")
        require(type(row["bf16_direct"]) is bool and row["staged_chunks"] in ([True] * chunks, [False] * chunks),
                f"rank {rank} invalid activation flags")
        require(len(row["measured"]) == samples, f"rank {rank} sample count disagrees with invocation")
        for step, sample in enumerate(row["measured"]):
            require(set(TIMING_COLUMNS) <= sample.keys() and all(finite(sample[key]) for key in TIMING_COLUMNS),
                    f"rank {rank} sample {step} has invalid timing columns")
            require(sample["ms"] > 0 and sample["total_gib"] > 0 and sample["submit_ms"] >= 0 and sample["wait_ms"] >= 0,
                    f"rank {rank} sample {step} has invalid timings or memory")
            require(math.isclose(sample["ms"], sample["submit_ms"] + sample["wait_ms"], rel_tol=0, abs_tol=1e-9),
                    f"rank {rank} sample {step} submit and wait disagree with wall time")
        require(not needs_oracle or "numerics" in row, f"rank {rank} missing requested FP64 comparison")
        if "numerics" in row:
            validate_numerics(row)
        rows.append(row)

    first = rows[0]
    shared = ("source", "source_hash", "initialization", "bf16_direct", "staged_chunks",
              "gradient_divide_factor", "force_sum_reduction_for_comms")
    require(all(row.get(field) == first.get(field) for row in rows for field in shared),
            "inconsistent source, fixture, activation, or reduction across ranks")
    require(re.fullmatch(r"[0-9a-f]{64}", first["source_hash"]) is not None, "invalid source hash")
    oracle_available = all("numerics" in row for row in rows)
    if oracle_available:
        require(all(row["numerics"]["reference_cache"] == first["numerics"]["reference_cache"] and all(
            row["numerics"][name]["reference"] == first["numerics"][name]["reference"] for name in LOSS_KEYS
        ) for row in rows), "inconsistent FP64 references across ranks")
    require(oracle_available or not any("numerics" in row for row in rows), "inconsistent FP64 coverage across ranks")
    verified_sum = first.get("gradient_divide_factor") == 1.0 and first.get("force_sum_reduction_for_comms") is True
    return {
        "status": "complete", "exit_code": 0, "model": model, "chunks": chunks, "variant": variant,
        "folded_rows_per_chunk": shape[0], "width": width, "vocab": VOCAB,
        "source_hash": first["source_hash"], "fixture": first["initialization"],
        "bf16_direct": first["bf16_direct"], "staged_chunks": first["staged_chunks"],
        "gradient_divide_factor": first.get("gradient_divide_factor"),
        "force_sum_reduction_for_comms": first.get("force_sum_reduction_for_comms"),
        "verified_sum": verified_sum, "oracle_available": oracle_available,
        "reference_cache": first["numerics"]["reference_cache"] if oracle_available else None,
        "timing": summarize_timing(rows),
        "worst_rank_errors": error_maxima(rows) if oracle_available else None,
    }, rows


def summarize_timing(rows):
    steps = []
    for samples in zip(*(row["measured"] for row in rows), strict=True):
        rank = max(RANKS, key=lambda index: samples[index]["ms"])
        step = {"rank": rank, **{key: samples[rank][key] for key in ("ms", "submit_ms", "wait_ms")}}
        step["total_gib"] = max(sample["total_gib"] for sample in samples)
        steps.append(step)
    return {
        "wall_ms": median(step["ms"] for step in steps),
        "submit_ms_on_slowest_ranks": median(step["submit_ms"] for step in steps),
        "wait_ms_on_slowest_ranks": median(step["wait_ms"] for step in steps),
        "total_gib": max(step["total_gib"] for step in steps),
        "steps": steps,
    }


def error_maxima(rows):
    return {name: {
        key: max(row["numerics"][name][key] for row in rows) for key in ERROR_COLUMNS
    } for name in NUMERICAL_COMPONENTS}


def float32_ulp_distance(a, b):
    def ordered_bits(value):
        bits = struct.unpack(">i", struct.pack(">f", value))[0]
        return bits if bits >= 0 else -(1 << 31) - bits

    return abs(ordered_bits(a) - ordered_bits(b))


def compare_run(cell, rows, reference, reference_rows):
    errors_available = cell["oracle_available"] and reference["oracle_available"]
    matched = cell["fixture"] == reference["fixture"]
    same_reference = cell["reference_cache"] == reference["reference_cache"] and (
        not errors_available or all(rows[0]["numerics"][name]["reference"] == reference_rows[0]["numerics"][name]["reference"]
                                    for name in LOSS_KEYS)
    )
    verified_sum = cell["verified_sum"] and reference["verified_sum"]
    grades = {}
    violations = []
    if matched and same_reference and errors_available:
        for component in NUMERICAL_COMPONENTS:
            if component == "dweight_shard" and not verified_sum:
                grades[component] = "unverified_reduction"
                continue
            previous_count = len(violations)
            for row, ref_row in zip(rows, reference_rows, strict=True):
                metrics = row["numerics"][component]
                ref = ref_row["numerics"][component]
                for metric in ERROR_COLUMNS:
                    value, limit = metrics[metric], ref[metric]
                    if value <= limit:
                        continue
                    detail = {"component": component, "rank": row["rank"], "metric": metric,
                              "error": value, "reference_error": limit, "increase": value - limit,
                              "relative_increase": (value - limit) / limit if limit else None}
                    if component in LOSS_KEYS:
                        detail["actual"] = metrics["actual"]
                        detail["reference_actual"] = ref["actual"]
                        detail["fp32_output_ulp_distance"] = float32_ulp_distance(metrics["actual"], ref["actual"])
                    violations.append(detail)
            grades[component] = "no_worse" if len(violations) == previous_count else "regression"
    available = errors_available and matched and same_reference and verified_sum
    if not errors_available:
        unavailable_reason = "missing numerical comparison"
    elif not matched or not same_reference:
        unavailable_reason = "different fixture or FP64 reference"
    elif not verified_sum:
        unavailable_reason = "SUM reduction with divide factor 1 is unverified"
    else:
        unavailable_reason = None
    only_one_ulp = bool(violations) and all(item.get("fp32_output_ulp_distance") == 1 for item in violations)
    if not available:
        classification = "unavailable"
    elif not violations:
        classification = "no_worse"
    elif only_one_ulp:
        classification = "one_ulp_scalar_regressions_only"
    else:
        classification = "regression"
    wall, ref_wall = (item["timing"]["wall_ms"] for item in (cell, reference))
    memory, ref_memory = (item["timing"]["total_gib"] for item in (cell, reference))
    return {
        "run": reference["run"], "same_fixture": matched, "classification": classification,
        "strict_no_worse": not violations if available else None, "unavailable_reason": unavailable_reason,
        "components": grades, "violations": violations,
        "wall_change_pct": 100 * (wall / ref_wall - 1) if matched else None,
        "total_gib_change_pct": 100 * (memory / ref_memory - 1) if matched else None,
    }


def summarize(series, final_arm, baseline_arm, reference_series=None):
    runs = []
    lookup = {}
    for directory in series:
        require(directory.is_dir(), f"missing series directory {directory}")
        paths = sorted(path for path in directory.iterdir() if path.is_dir() and path.name.startswith(("100b-c", "700b-c")))
        require(bool(paths), f"no head runs found in {directory}")
        for path in paths:
            try:
                cell, rows = load_run(path)
            except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
                cell, rows = {"status": "invalid", "error": str(error)}, []
            cell.update(run=f"{directory.name}/{path.name}", series=directory.name)
            runs.append((cell, rows))
            if rows:
                key = (directory.name, cell["model"], cell["chunks"], cell["variant"])
                require(key not in lookup, f"duplicate receipt for {key}")
                lookup[key] = (cell, rows)
    for cell, rows in runs:
        if not rows:
            continue
        for label, arm in (("vs_final", final_arm), ("vs_baseline", baseline_arm)):
            case = (cell["model"], cell["chunks"], arm)
            anchor = (reference_series or cell["series"], *case)
            reference = lookup.get(anchor)
            if reference is None and reference_series is None:
                reference = lookup.get((series[0].name, *case))
            cell[label] = (compare_run(cell, rows, *reference) if reference is not None else
                           {"classification": "unavailable", "strict_no_worse": None,
                            "unavailable_reason": f"missing {arm} comparison run"})
    return {
        "timing_scope": "Isolated host wall, including the queued CUDA work and synchronization. Median of per-step maximum wall across eight ranks. Submit and wait use each step's slowest wall rank. Device kernel and communication times are unavailable.",
        "memory_scope": "Maximum of reported GIFT total GiB across all steps and ranks.",
        "numerical_rule": "Each of five components must have no larger maximum absolute or relative L2 error on any rank. One-ULP scalar output differences are identified separately and fail a strict regression check when error increases.",
        "runs": [cell for cell, _ in runs],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", nargs="+", type=Path)
    parser.add_argument("--reference-series", help="Series directory name supplying both comparison arms")
    parser.add_argument("--final-arm", default="cuda13")
    parser.add_argument("--baseline-arm", default="baseline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.series, args.final_arm, args.baseline_arm, args.reference_series)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
