"""Input parsing and output helpers for stability comparison."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from heapq import heapreplace
from pathlib import Path
from typing import Any
import yaml

Series = list[tuple[int, float]]
_MAX_VIO_DROP_POINTS: int | None = None
_MAX_VIO_TOP_POINTS: int | None = None
_RUN_DETAILS_FIELDNAMES = [
    "setup_name",
    "run_name",
    "lr",
    "status",
    "max_step",
    "failed_criteria",
    "first_failure_step",
    "loss_tag",
    "notes",
]

_LR_RE = re.compile(
    r"(?P<prefix>(?:^|[-_])lr[_-]?)(?P<lr>[0-9]+(?:\.[0-9]+)?(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)
_MAX_VIO_LAYER_RE = re.compile(
    r"^dispatcher_layer_experts_max_vio_batch_no_pad_no_prefix_avg/layer_(?P<layer>\d+)$"
)
_LOSS_EXACT_PRIORITY = {
    "loss": 100,
    "train/loss": 100,
    "loss/train": 100,
    "metrics/train/loss": 100,
    "train/metrics/loss": 100,
    "train/total_loss": 95,
    "total_loss": 95,
}


@dataclass
class LayerVioSummary:
    """Compact per-layer view sufficient for the current instability rule."""

    earliest_points: Series = field(default_factory=list)
    top_points: Series = field(default_factory=list)


@dataclass
class RunTrace:
    """Standardized scalar view consumed by collapse criteria."""

    run_name: str
    run_path: Path
    lr: float
    setup_name: str
    max_step: int = 0
    activation_amax: Series = field(default_factory=list)
    loss: Series = field(default_factory=list)
    max_vio_layers: dict[int, LayerVioSummary] = field(default_factory=dict)
    max_vio_theoretical_limit: float | None = None
    loss_tag: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunEvaluation:
    run_name: str
    setup_name: str
    lr: float
    status: str
    max_step: int
    failed_criteria: str
    first_failure_step: int | None
    loss_tag: str
    notes: str


def parse_run_identity(path: Path) -> tuple[str, float]:
    """Return setup name and LR parsed from a run directory name."""

    name = path.name
    matches = list(_LR_RE.finditer(name))
    if not matches:
        raise ValueError(f"could not parse lr from run name: {name}")

    match = matches[-1]
    lr = float(match.group("lr"))
    setup_name = f"{name[: match.start()]}{name[match.end() :]}"
    setup_name = re.sub(r"[-_]{2,}", "-", setup_name).strip("-_")
    return setup_name, lr


def discover_run_dirs(input_dir: Path) -> list[Path]:
    """Find TensorBoard run directories under the given input directory."""

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    if _has_direct_event_files(input_dir):
        return [input_dir]

    run_dirs = [
        child
        for child in input_dir.iterdir()
        if child.is_dir()
        and (_has_event_files(child) or (child / "config.yaml").exists())
    ]
    return sorted(run_dirs, key=lambda path: path.name)


def _has_event_files(path: Path) -> bool:
    return any(path.glob("events.out.tfevents.*")) or any(
        path.rglob("events.out.tfevents.*")
    )


def _has_direct_event_files(path: Path) -> bool:
    return any(path.glob("events.out.tfevents.*"))


def discover_event_files(run_dir: Path) -> list[Path]:
    """List TensorBoard event files for a single run directory."""

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if event_files:
        return event_files
    return sorted(run_dir.rglob("events.out.tfevents.*"))


def _extract_theoretical_max_vio(run_dir: Path) -> float | None:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None

    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}

    overrides = config.get("model", {}).get("config_overrides", {})
    num_routed_experts = overrides.get("num_routed_experts")
    num_experts_per_tok = overrides.get("num_experts_per_tok")
    if num_routed_experts is None or num_experts_per_tok is None:
        return None

    routed = float(num_routed_experts)
    per_tok = float(num_experts_per_tok)
    if per_tok <= 0:
        raise ValueError(f"invalid num_experts_per_tok in {config_path}")
    return routed / per_tok - 1.0


def initialize_run_trace(run_dir: Path) -> RunTrace:
    """Build a trace with metadata loaded from the run directory."""

    setup_name, lr = parse_run_identity(run_dir)
    trace = RunTrace(run_dir.name, run_dir, lr, setup_name)
    try:
        trace.max_vio_theoretical_limit = _extract_theoretical_max_vio(run_dir)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        trace.warnings.append(f"config read failed: {exc}")
    if trace.max_vio_theoretical_limit is None:
        trace.warnings.append("no max_vio config bound")
    return trace


def _extract_scalar(value: Any) -> float | None:
    if value.HasField("simple_value"):
        return float(value.simple_value)
    if value.HasField("tensor"):
        tensor = value.tensor
        if tensor.float_val:
            return float(tensor.float_val[0])
        if tensor.double_val:
            return float(tensor.double_val[0])
        if tensor.int_val:
            return float(tensor.int_val[0])
    return None


def _validate_max_vio_point_limits(
    max_vio_drop_points: int,
    max_vio_top_points: int,
) -> tuple[int, int]:
    if max_vio_drop_points < 0:
        raise ValueError("max_vio_drop_points must be non-negative")
    if max_vio_top_points <= 0:
        raise ValueError("max_vio_top_points must be positive")
    if max_vio_top_points < max_vio_drop_points:
        raise ValueError("max_vio_top_points must be >= max_vio_drop_points")
    return max_vio_drop_points, max_vio_top_points


def configure_max_vio_point_limits(
    *,
    max_vio_drop_points: int,
    max_vio_top_points: int,
) -> None:
    """Configure retained max-vio points from CLI/runtime parameters."""

    global _MAX_VIO_DROP_POINTS, _MAX_VIO_TOP_POINTS
    _MAX_VIO_DROP_POINTS, _MAX_VIO_TOP_POINTS = _validate_max_vio_point_limits(
        max_vio_drop_points,
        max_vio_top_points,
    )


def _resolve_max_vio_point_limits(
    max_vio_drop_points: int | None,
    max_vio_top_points: int | None,
) -> tuple[int, int]:
    if max_vio_drop_points is None:
        max_vio_drop_points = _MAX_VIO_DROP_POINTS
    if max_vio_top_points is None:
        max_vio_top_points = _MAX_VIO_TOP_POINTS
    if max_vio_drop_points is None or max_vio_top_points is None:
        raise RuntimeError("max_vio point limits are not configured")
    return _validate_max_vio_point_limits(
        max_vio_drop_points,
        max_vio_top_points,
    )


def _is_activation_tag(tag: str) -> bool:
    if "{}" in tag:
        return False
    return tag.startswith(("activations_amax/", "activations/amax/"))


def _is_loss_tag(tag: str) -> bool:
    return "loss" in tag.lower()


def _merge_step_max(dst: dict[int, float], step: int, value: float) -> None:
    value = abs(value)
    if step not in dst or value > dst[step]:
        dst[step] = value


def _parse_max_vio_layer(tag: str) -> int | None:
    match = _MAX_VIO_LAYER_RE.match(tag)
    if match is None:
        return None
    return int(match.group("layer"))


def _append_earliest_point(
    earliest_points: Series,
    point: tuple[int, float],
    *,
    max_vio_drop_points: int,
) -> None:
    earliest_points.append(point)
    if len(earliest_points) > (max_vio_drop_points * 2):
        earliest_points.sort()
        del earliest_points[max_vio_drop_points:]


def _append_top_point(
    top_points: list[tuple[float, int]],
    point: tuple[int, float],
    *,
    max_vio_top_points: int,
) -> None:
    step, value = point
    heap_item = (value, step)
    if len(top_points) < max_vio_top_points:
        top_points.append(heap_item)
        if len(top_points) == max_vio_top_points:
            top_points.sort()
        return

    if heap_item > top_points[0]:
        heapreplace(top_points, heap_item)


def _merge_layer_summaries(
    summaries: list[LayerVioSummary],
    *,
    max_vio_drop_points: int,
    max_vio_top_points: int,
) -> LayerVioSummary:
    earliest_points: Series = []
    top_candidates: list[tuple[int, float]] = []

    for summary in summaries:
        earliest_points.extend(summary.earliest_points)
        top_candidates.extend(summary.top_points)

    earliest_points.sort()
    top_candidates.sort(key=lambda point: point[1], reverse=True)
    return LayerVioSummary(
        earliest_points=earliest_points[:max_vio_drop_points],
        top_points=top_candidates[:max_vio_top_points],
    )


def _loss_tag_score(tag: str, series: Series, max_step: int) -> tuple[int, int, int]:
    low = tag.lower()
    score = _LOSS_EXACT_PRIORITY.get(low, 0)
    if "train" in low and "loss" in low:
        score += 50
    if low.endswith("loss") or "/loss" in low:
        score += 30
    if "eval" in low or "val" in low:
        score -= 40
    if any(token in low for token in ("z_loss", "aux", "router", "mtp")):
        score -= 25
    last_step = series[-1][0] if series else 0
    reaches_end = int(last_step >= max_step)
    return score, reaches_end, len(series)


def _select_loss(losses: dict[str, Series], max_step: int) -> tuple[str | None, Series]:
    if not losses:
        return None, []
    tag = max(losses, key=lambda key: _loss_tag_score(key, losses[key], max_step))
    return tag, losses[tag]


def read_event_file_partial(
    path: Path,
    *,
    max_vio_drop_points: int | None = None,
    max_vio_top_points: int | None = None,
) -> dict[str, Any]:
    """Read one event file and aggregate only scalars used by stability checks."""

    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader  # noqa: PLC0415

    max_vio_drop_points, max_vio_top_points = _resolve_max_vio_point_limits(
        max_vio_drop_points,
        max_vio_top_points,
    )
    max_step = 0
    activation_amax: dict[int, float] = {}
    layer_earliest_points: dict[int, Series] = {}
    layer_top_points: dict[int, list[tuple[float, int]]] = {}
    losses: dict[str, Series] = {}

    for event in EventFileLoader(str(path)).Load():
        max_step = max(max_step, int(event.step))
        if not event.HasField("summary"):
            continue

        for value in event.summary.value:
            tag = value.tag
            scalar = _extract_scalar(value)
            if scalar is None:
                continue

            step = int(event.step)
            layer = _parse_max_vio_layer(tag)
            if layer is not None:
                point = (step, scalar)
                _append_earliest_point(
                    layer_earliest_points.setdefault(layer, []),
                    point,
                    max_vio_drop_points=max_vio_drop_points,
                )
                _append_top_point(
                    layer_top_points.setdefault(layer, []),
                    point,
                    max_vio_top_points=max_vio_top_points,
                )
                continue

            if _is_activation_tag(tag):
                _merge_step_max(activation_amax, step, scalar)
            elif _is_loss_tag(tag):
                losses.setdefault(tag, []).append((step, scalar))

    return {
        "max_step": max_step,
        "activation_amax": activation_amax,
        "losses": losses,
        "max_vio_layers": {
            layer: LayerVioSummary(
                earliest_points=sorted(layer_earliest_points.get(layer, []))[
                    :max_vio_drop_points
                ],
                top_points=sorted(
                    ((step, value) for value, step in top_points),
                    key=lambda point: point[1],
                    reverse=True,
                )[:max_vio_top_points],
            )
            for layer, top_points in layer_top_points.items()
        },
    }


def _merge_event_file_data(
    partials: list[dict[str, Any]],
    *,
    max_vio_drop_points: int | None = None,
    max_vio_top_points: int | None = None,
) -> dict[str, Any]:
    max_vio_drop_points, max_vio_top_points = _resolve_max_vio_point_limits(
        max_vio_drop_points,
        max_vio_top_points,
    )
    max_step = 0
    activation_amax: dict[int, float] = {}
    max_vio_layers: dict[int, list[LayerVioSummary]] = {}
    losses: dict[str, Series] = {}

    for partial in partials:
        max_step = max(max_step, partial["max_step"])
        for step, value in partial["activation_amax"].items():
            _merge_step_max(activation_amax, step, value)
        for layer, series in partial["max_vio_layers"].items():
            max_vio_layers.setdefault(layer, []).append(series)
        for tag, series in partial["losses"].items():
            losses.setdefault(tag, []).extend(series)

    return {
        "max_step": max_step,
        "activation_amax": sorted(activation_amax.items()),
        "losses": {tag: sorted(series) for tag, series in losses.items()},
        "max_vio_layers": {
            layer: _merge_layer_summaries(
                summaries,
                max_vio_drop_points=max_vio_drop_points,
                max_vio_top_points=max_vio_top_points,
            )
            for layer, summaries in max_vio_layers.items()
        },
    }


def apply_event_file_partials(
    trace: RunTrace,
    partials: list[dict[str, Any]],
    *,
    max_vio_drop_points: int | None = None,
    max_vio_top_points: int | None = None,
) -> RunTrace:
    """Populate a trace from already-read event file partials."""

    if not partials:
        if "no TensorBoard event files" not in trace.warnings:
            trace.warnings.append("no TensorBoard event files")
        return trace

    merged = _merge_event_file_data(
        partials,
        max_vio_drop_points=max_vio_drop_points,
        max_vio_top_points=max_vio_top_points,
    )
    loss_tag, loss = _select_loss(merged["losses"], merged["max_step"])

    trace.max_step = merged["max_step"]
    trace.activation_amax = merged["activation_amax"]
    trace.max_vio_layers = merged["max_vio_layers"]
    trace.loss_tag = loss_tag
    trace.loss = loss

    if not trace.max_vio_layers:
        trace.warnings.append("no dispatcher layer max_vio tags")
    if not trace.activation_amax:
        trace.warnings.append("no activation amax tags")
    if not trace.loss:
        trace.warnings.append("no loss tags")
    return trace


def load_run_trace(
    run_dir: Path,
    *,
    max_vio_drop_points: int | None = None,
    max_vio_top_points: int | None = None,
) -> RunTrace:
    """Load one run directory into the standardized trace format."""

    max_vio_drop_points, max_vio_top_points = _resolve_max_vio_point_limits(
        max_vio_drop_points,
        max_vio_top_points,
    )
    event_files = discover_event_files(run_dir)
    trace = initialize_run_trace(run_dir)
    if not event_files:
        trace.warnings.append("no TensorBoard event files")
        return trace

    partials = [
        read_event_file_partial(
            path,
            max_vio_drop_points=max_vio_drop_points,
            max_vio_top_points=max_vio_top_points,
        )
        for path in event_files
    ]
    return apply_event_file_partials(
        trace,
        partials,
        max_vio_drop_points=max_vio_drop_points,
        max_vio_top_points=max_vio_top_points,
    )


def format_lr(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.8g}"


def _parse_optional_int(value: str | int | float | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _read_existing_evaluations(details_path: Path) -> list[RunEvaluation]:
    if not details_path.exists():
        return []

    with details_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            RunEvaluation(
                run_name=row["run_name"],
                setup_name=row["setup_name"],
                lr=float(row["lr"]),
                status=row["status"],
                max_step=int(row["max_step"]),
                failed_criteria=row["failed_criteria"],
                first_failure_step=_parse_optional_int(row["first_failure_step"]),
                loss_tag=row["loss_tag"],
                notes=row["notes"],
            )
            for row in reader
        ]


def _merge_evaluations(
    existing: list[RunEvaluation],
    updates: list[RunEvaluation],
) -> list[RunEvaluation]:
    merged = {evaluation.run_name: evaluation for evaluation in existing}
    for evaluation in updates:
        previous = merged.get(evaluation.run_name)
        if previous is None or previous.status == "?":
            merged[evaluation.run_name] = evaluation
    return list(merged.values())


def evaluate_trace_status(
    trace: RunTrace, criteria_results, min_full_steps: int
) -> RunEvaluation:
    failed = [result for result in criteria_results if result.failed]
    if failed:
        status = "broken"
    elif trace.max_step >= min_full_steps and all(
        result.failed is not None for result in criteria_results
    ):
        status = "stable"
    else:
        status = "?"

    first_failure_step = min(
        (result.first_step for result in failed if result.first_step is not None),
        default=None,
    )
    failed_criteria = (
        ", ".join(f"{result.name} ({result.details})" for result in failed) or "-"
    )
    notes = "; ".join(trace.warnings)
    if status == "?" and (
        trace.max_step < min_full_steps
        or any(result.failed is None for result in criteria_results)
    ):
        notes = "; ".join(filter(None, [notes, f"max_step < {min_full_steps}"]))

    return RunEvaluation(
        run_name=trace.run_name,
        setup_name=trace.setup_name,
        lr=trace.lr,
        status=status,
        max_step=trace.max_step,
        failed_criteria=failed_criteria,
        first_failure_step=first_failure_step,
        loss_tag=trace.loss_tag or "-",
        notes=notes or "-",
    )


def build_summary_rows(evaluations: list[RunEvaluation]) -> list[dict[str, str]]:
    by_setup: dict[str, list[RunEvaluation]] = {}
    for evaluation in evaluations:
        by_setup.setdefault(evaluation.setup_name, []).append(evaluation)

    rows: list[dict[str, str]] = []
    for setup_name, setup_runs in sorted(by_setup.items()):
        stable_lrs = [run.lr for run in setup_runs if run.status == "stable"]
        broken_runs = [run for run in setup_runs if run.status == "broken"]
        broken_lrs = [run.lr for run in broken_runs]
        pending_lrs = [run.lr for run in setup_runs if run.status == "?"]
        min_broken_run = (
            min(broken_runs, key=lambda run: run.lr) if broken_runs else None
        )
        rows.append(
            {
                "setup": setup_name,
                "max_stable_lr": format_lr(max(stable_lrs) if stable_lrs else None),
                "min_broken_lr": format_lr(min(broken_lrs) if broken_lrs else None),
                "min_broken_criteria": (
                    min_broken_run.failed_criteria if min_broken_run else "-"
                ),
                "pending_lrs": (
                    ", ".join(format_lr(lr) for lr in sorted(pending_lrs))
                    if pending_lrs
                    else "-"
                ),
                "runs": str(len(setup_runs)),
            }
        )
    return rows


def format_markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No runs found."

    headers = list(rows[0].keys())
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    sep = "| " + " | ".join("-" * widths[header] for header in headers) + " |"
    out = [
        "| " + " | ".join(header.ljust(widths[header]) for header in headers) + " |",
        sep,
    ]
    for row in rows:
        out.append(
            "| "
            + " | ".join(row[header].ljust(widths[header]) for header in headers)
            + " |"
        )
    return "\n".join(out)


def write_outputs(
    output_dir: Path,
    result_name: str,
    summary_rows: list[dict[str, str]],
    evaluations: list[RunEvaluation],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{result_name}.csv"
    details_path = output_dir / f"{result_name}_runs.csv"
    md_path = output_dir / f"{result_name}.md"

    output_evaluations = _merge_evaluations(
        _read_existing_evaluations(details_path),
        evaluations,
    )
    summary_rows = build_summary_rows(output_evaluations)

    if summary_rows:
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_path.write_text(
            "setup,max_stable_lr,min_broken_lr,min_broken_criteria,pending_lrs,runs\n"
        )

    with details_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_RUN_DETAILS_FIELDNAMES)
        writer.writeheader()
        for evaluation in sorted(
            output_evaluations, key=lambda item: (item.setup_name, item.lr)
        ):
            writer.writerow(
                {
                    "setup_name": evaluation.setup_name,
                    "run_name": evaluation.run_name,
                    "lr": format_lr(evaluation.lr),
                    "status": evaluation.status,
                    "max_step": evaluation.max_step,
                    "failed_criteria": evaluation.failed_criteria,
                    "first_failure_step": evaluation.first_failure_step or "",
                    "loss_tag": evaluation.loss_tag,
                    "notes": evaluation.notes,
                }
            )

    md_path.write_text(format_markdown_table(summary_rows) + "\n")
    return summary_path, details_path, md_path
