from __future__ import annotations

import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from metric_sampler.contract.metrics import ACTIVATION_LOG2_BINS, ENTROPY_TOKEN_STRIDE

REL_TOL = 1e-4
LOSS_CE_REL_TOL = 1e-3
TOKEN_COUNT_REL_TOL = 1e-6
EXPONENT_HISTOGRAM_REL_TOL = 1e-5
WHOLE_NUMBER_TOL = 1e-3
LOG2_RANGE_TOL = 1e-6
SMALLEST_MEAN_SQUARE = 1e-300
WIDTH_DIGITS = 3
PROBLEMS_SHOWN = 4
# Bin i of an activation histogram holds the values whose log2 magnitude is i - BIN_EXPONENT_OFFSET.
BIN_EXPONENT_OFFSET = -int(ACTIVATION_LOG2_BINS.start)
TOKEN_COVERING_POINTS = ("residual", "attn_out", "mlp_out", "final_norm")
EQUAL_WIDTH_POINTS = ("residual", "attn_out", "mlp_out")


def load_rows(window_dirs: list[Path], family: str, names: dict[int, str]) -> list[dict]:
    rows = []
    for window_dir in window_dirs:
        path = window_dir / f"{family}.parquet"
        if path.exists():
            for row in pq.read_table(path, partitioning=None).to_pylist():
                row["name"] = names[row["metric_id"]]
                rows.append(row)
    return rows


def is_close(first: float, second: float, tolerance: float) -> bool:
    return abs(first - second) <= tolerance * max(abs(first), abs(second), 1.0)


def activation_problems(metrics: dict, tokens_per_microbatch: int) -> list[str]:
    problems = []
    microbatches = len({row["microbatch"] for row in metrics.get("loss/ce", [])})
    widths = {}
    for moments in (row for name, rows in metrics.items() if name.endswith("/moments") for row in rows):
        point = moments["name"].split("/")[1]
        layer = moments["layer"]
        _, sum_of_squares, nonfinite, finite = moments["values"]
        root_mean_square_rows = [
            row for row in metrics.get(f"act/{point}/token_rms_hist", []) if row["layer"] == layer
        ]
        if not root_mean_square_rows:
            problems.append(f"act/{point} layer {layer}: moments without token_rms_hist")
            continue
        root_mean_square_histogram = root_mean_square_rows[0]["values"]
        tokens = sum(root_mean_square_histogram)
        expected_tokens = tokens_per_microbatch * microbatches * moments["contributors"]
        if point in TOKEN_COVERING_POINTS and not is_close(tokens, expected_tokens, TOKEN_COUNT_REL_TOL):
            problems.append(f"act/{point} layer {layer}: rms hist tokens {tokens:.0f} != {expected_tokens}")
        width = (finite + nonfinite) / max(tokens, 1)
        if point in EQUAL_WIDTH_POINTS:
            widths.setdefault(round(width, WIDTH_DIGITS), []).append(f"{point}@{layer}")
        if abs(width - round(width)) > WHOLE_NUMBER_TOL:
            problems.append(f"act/{point} layer {layer}: elements per token {width:.4f} is not whole")
        problems.extend(
            token_range_problems(point, layer, root_mean_square_histogram, sum_of_squares, finite)
        )
        problems.extend(exponent_problems(metrics, point, layer, finite))
    if len(widths) > 1:
        problems.append(f"activation widths differ: { {width: names[:3] for width, names in widths.items()} }")
    return problems


def token_range_problems(
    point: str, layer: int, root_mean_square_histogram: list, sum_of_squares: float, finite: float
) -> list[str]:
    nonzero = [index for index, count in enumerate(root_mean_square_histogram) if count > 0]
    if finite > 0 and nonzero:
        log2_root_mean_square = 0.5 * math.log2(max(sum_of_squares / finite, SMALLEST_MEAN_SQUARE))
        low, high = nonzero[0] - BIN_EXPONENT_OFFSET, nonzero[-1] - BIN_EXPONENT_OFFSET + 1
        if not low - LOG2_RANGE_TOL <= log2_root_mean_square <= high + LOG2_RANGE_TOL:
            return [f"act/{point} layer {layer}: log2 rms {log2_root_mean_square:.3f} "
                    f"outside token range [{low}, {high}]"]
    return []


def exponent_problems(metrics: dict, point: str, layer: int, finite: float) -> list[str]:
    problems = []
    exponent_rows = [row for row in metrics.get(f"act/{point}/exp_hist", []) if row["layer"] == layer]
    if exponent_rows:
        exponent_histogram = exponent_rows[0]["values"]
        if not is_close(sum(exponent_histogram), finite, EXPONENT_HISTOGRAM_REL_TOL):
            problems.append(
                f"act/{point} layer {layer}: exp hist {sum(exponent_histogram):.0f} != finite {finite:.0f}"
            )
        amax = max((row["values"][0] for row in metrics.get(f"act/{point}/amax", []) if row["layer"] == layer),
                   default=0.0)
        top = max((index for index, count in enumerate(exponent_histogram) if count > 0), default=None)
        if (top is not None and top < len(exponent_histogram) - 1
                and amax >= 2.0 ** (top - BIN_EXPONENT_OFFSET + 1)):
            problems.append(
                f"act/{point} layer {layer}: amax {amax} above top exponent bin {top - BIN_EXPONENT_OFFSET}"
            )
    return problems


def step_problems(metrics: dict, token_bins: dict, top_k: int, tokens_per_microbatch: int) -> list[str]:
    problems = []
    stream_count = sum(metrics["loss/stream_count"][0]["values"])
    stream_sum = sum(metrics["loss/stream_sum"][0]["values"])
    histogram = metrics["loss/token_hist_by_stream"][0]["values"]
    if sum(histogram) != stream_count:
        problems.append(f"hist tokens {sum(histogram)} != stream count {stream_count}")
    problems.extend(expert_loss_problems(metrics, top_k, stream_count, stream_sum))
    problems.extend(loss_mean_problems(metrics, token_bins, histogram, stream_count, stream_sum))
    problems.extend(entropy_histogram_problems(metrics, tokens_per_microbatch))
    problems.extend(expert_histogram_problems(metrics))
    problems.extend(activation_problems(metrics, tokens_per_microbatch))
    problems.extend(grad_clip_problems(metrics))
    return problems


def expert_loss_problems(metrics: dict, top_k: int, stream_count: int, stream_sum: float) -> list[str]:
    problems = []
    mtp_tokens = {row["layer"]: sum(row["values"]) for row in metrics.get("loss/mtp_token_hist", [])}
    for row in metrics.get("expert_loss/count", []):
        layer = row["layer"]
        expected = top_k * mtp_tokens.get(layer, stream_count)
        if sum(row["values"]) != expected:
            problems.append(f"layer {layer}: expert count {sum(row['values'])} != top_k * tokens {expected}")
    for row in metrics.get("expert_loss/sum", []):
        if row["layer"] not in mtp_tokens and not is_close(sum(row["values"]) / top_k, stream_sum, REL_TOL):
            problems.append(f"layer {row['layer']}: expert loss sum / top_k {sum(row['values']) / top_k:.4f} "
                            f"!= stream sum {stream_sum:.4f}")
    return problems


def loss_mean_problems(
    metrics: dict, token_bins: dict, histogram: list, stream_count: int, stream_sum: float
) -> list[str]:
    problems = []
    exact_mean = stream_sum / stream_count if stream_count else 0.0
    loss_ce = [row["values"][0] for row in metrics.get("loss/ce", [])]
    if loss_ce and not is_close(exact_mean, sum(loss_ce) / len(loss_ce), LOSS_CE_REL_TOL):
        problems.append(f"stream mean {exact_mean:.5f} vs loss/ce mean {sum(loss_ce) / len(loss_ce):.5f}")

    bin_count = token_bins["count"]
    folded = [sum(histogram[stream * bin_count + bin_index] for stream in range(len(histogram) // bin_count))
              for bin_index in range(bin_count)]
    histogram_mean = sum(count * (token_bins["start"] + token_bins["width"] * (index + 0.5))
                         for index, count in enumerate(folded)) / max(stream_count, 1)
    if abs(histogram_mean - exact_mean) > token_bins["width"] / 2:
        problems.append(f"hist mean {histogram_mean:.4f} vs exact {exact_mean:.4f}")
    return problems


def entropy_histogram_problems(metrics: dict, tokens_per_microbatch: int) -> list[str]:
    problems = []
    if "router/entropy_hist" in metrics:
        microbatches = len({row["microbatch"] for row in metrics["loss/ce"]})
        ranks = metrics["loss/stream_count"][0]["contributors"]
        for row in metrics["router/entropy_hist"]:
            expected = math.ceil(tokens_per_microbatch / ENTROPY_TOKEN_STRIDE) * microbatches * ranks
            if sum(row["values"]) != expected:
                problems.append(f"layer {row['layer']}: entropy hist total {sum(row['values'])} != {expected}")
    return problems


def expert_histogram_problems(metrics: dict) -> list[str]:
    problems = []
    if "expert_loss/hist" in metrics:
        counts = {row["layer"]: sum(row["values"]) for row in metrics["expert_loss/count"]}
        for name in ("expert_loss/hist", "expert_loss/stream_count"):
            for row in metrics[name]:
                if sum(row["values"]) != counts[row["layer"]]:
                    problems.append(f"layer {row['layer']}: {name} total {sum(row['values'])} "
                                    f"!= expert count {counts[row['layer']]}")
    return problems


def grad_clip_problems(metrics: dict) -> list[str]:
    return ["grad/clip_total_norm differs across ranks" for row in metrics.get("grad/clip_total_norm", [])
            if not row["identical"]]


def main(run_dir: str, top_k: int, tokens_per_microbatch: int) -> None:
    attempt_dirs = sorted(Path(path) for path in glob.glob(f"{run_dir}/attempt=*"))
    failures = 0
    checked_steps = 0
    detailed_steps = 0
    for attempt_dir in attempt_dirs:
        registry = json.loads((attempt_dir / "metrics.json").read_text())
        names = {metric["metric_id"]: metric["name"] for metric in registry}
        bins = {metric["name"]: metric["bins"] for metric in registry if metric.get("bins")}
        run_path = Path(run_dir)
        collected_attempt = (
            run_path.parent.parent / "_collected" / run_path.parent.name / run_path.name / attempt_dir.name
        )
        window_dirs = sorted(
            Path(path) for root in (attempt_dir, collected_attempt) for path in glob.glob(f"{root}/node=*/w*")
        )
        rows = []
        for family in ("scalars", "expert_counts", "router_vectors"):
            rows.extend(load_rows(window_dirs, family, names))

        by_step = defaultdict(lambda: defaultdict(list))
        for row in rows:
            by_step[(row["node"], row["step"])][row["name"]].append(row)

        for (node, step), metrics in sorted(by_step.items()):
            if "loss/stream_count" not in metrics:
                continue
            checked_steps += 1
            if "expert_loss/hist" in metrics:
                detailed_steps += 1
            problems = step_problems(metrics, bins["loss/token_hist_by_stream"], top_k, tokens_per_microbatch)
            if problems:
                failures += 1
                print(f"FAIL {node} step {step}: " + "; ".join(problems[:PROBLEMS_SHOWN]))

        logged = sorted(
            path for root in (attempt_dir, collected_attempt) for path in glob.glob(f"{root}/logged/w*/logged.parquet")
        )
        logged_rows = sum(pq.read_metadata(path).num_rows for path in logged)
        print(f"{attempt_dir.name}: logged windows {len(logged)}, rows {logged_rows}")
        streams_path = attempt_dir / "streams.json"
        if streams_path.exists():
            print(f"{attempt_dir.name}: streams {len(json.loads(streams_path.read_text()))}")

    print(f"checked {checked_steps} node-steps ({detailed_steps} detailed), failures {failures}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
