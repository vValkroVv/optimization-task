"""Compare stability experiments from TensorBoard logs.

The input directory is expected to contain one subdirectory per run. Runs are
grouped by the run name with the numeric ``lr_*`` fragment removed.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from traceback import format_exc
from time import perf_counter
from typing import TypeVar
import random

from src.criteria import evaluate_collapse
from src.io_utils import (
    RunTrace,
    apply_event_file_partials,
    build_summary_rows,
    configure_max_vio_point_limits,
    discover_event_files,
    discover_run_dirs,
    evaluate_trace_status,
    initialize_run_trace,
    read_event_file_partial,
    write_outputs,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
_DEFAULT_MAX_VIO_DROP_POINTS = 10
_DEFAULT_MAX_VIO_TOP_K = 40
_PROGRESS_BAR_WIDTH = 30
T = TypeVar("T")


def _resolve_worker_count(file_count: int, requested_workers: int | None) -> int:
    cpu_count = os.cpu_count() or 1
    print(f"Total number of CPU: {cpu_count}")
    worker_cap = min(requested_workers or 1, cpu_count)
    return max(1, min(file_count, worker_cap))


def _split_evenly(items: list[T], chunk_count: int) -> list[list[T]]:
    random.shuffle(items)
    base_size, remainder = divmod(len(items), chunk_count)
    chunks: list[list[T]] = []
    start = 0
    for idx in range(chunk_count):
        chunk_size = base_size + (1 if idx < remainder else 0)
        end = start + chunk_size
        chunks.append(items[start:end])
        start = end
    return chunks


def _render_progress(completed: int, total: int) -> str:
    if total <= 0:
        return "Event files: [------------------------------] 0/0 (100.0%)"
    ratio = completed / total
    filled = int(_PROGRESS_BAR_WIDTH * ratio)
    bar = "#" * filled + "-" * (_PROGRESS_BAR_WIDTH - filled)
    return f"Event files: [{bar}] {completed}/{total} ({ratio * 100:5.1f}%)"


def _read_event_file_batch(
    event_files: list[tuple[str, Path]],
    max_vio_drop_points: int,
    max_vio_top_points: int,
) -> dict[str, list[dict]]:
    grouped_partials: dict[str, list[dict]] = defaultdict(list)
    for run_name, event_file in event_files:
        try:
            grouped_partials[run_name].append(
                read_event_file_partial(
                    event_file,
                    max_vio_drop_points=max_vio_drop_points,
                    max_vio_top_points=max_vio_top_points,
                )
            )
        except Exception as exc:
            print(
                "\n"
                f"failed to read TensorBoard event file for run={run_name} "
                f"path={event_file}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            print(format_exc(), flush=True)
    return dict(grouped_partials)


def _read_processed_run_names(details_path: Path) -> set[str]:
    if not details_path.exists():
        return set()

    with details_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["run_name"]
            for row in reader
            if row.get("run_name") and row.get("status") != "?"
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate setup stability from TensorBoard event logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory with TensorBoard run dirs."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV/Markdown artifacts are written.",
    )
    parser.add_argument(
        "result_name",
        nargs="?",
        default="stability_exps_res",
        help="Base name for output artifacts.",
    )
    parser.add_argument("--min-full-steps", type=int, default=16000)
    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help=argparse.SUPPRESS
    )
    parser.add_argument("--max-vio-threshold-share", type=float, default=0.2)
    parser.add_argument(
        "--max-vio-drop-points",
        type=int,
        default=_DEFAULT_MAX_VIO_DROP_POINTS,
        help="Drop this many leading points from each dispatcher layer max_vio series.",
    )
    parser.add_argument(
        "--max-vio-top-k",
        type=int,
        default=_DEFAULT_MAX_VIO_TOP_K,
        help="Average this many largest remaining values per layer.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="When True, check if the output directory already exists and run evaluations only if it does not.",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=1e-6,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--loss-relative-increase",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--loss-absolute-increase",
        type=float,
        default=0.2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--persistent-points",
        type=int,
        default=3,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--persistent-step-span",
        type=int,
        default=400,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max concurrent event-file reader processes. Defaults to CPU count capped by file count.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    max_vio_top_points = args.max_vio_drop_points + args.max_vio_top_k
    configure_max_vio_point_limits(
        max_vio_drop_points=args.max_vio_drop_points,
        max_vio_top_points=max_vio_top_points,
    )

    processed_runs = set()
    if args.use_cache:
        if os.path.exists(args.output_dir):
            runs_csv = args.output_dir / f"{args.result_name}_runs.csv"
            processed_runs = _read_processed_run_names(runs_csv)

    started = perf_counter()

    run_dirs = discover_run_dirs(args.input_dir)
    if not run_dirs:
        raise SystemExit(f"error: no run directories found in {args.input_dir}")

    run_dirs = [
        run_dir
        for run_dir in run_dirs
        if run_dir.name not in processed_runs
        and not ("-lr_0.1-" in run_dir.name or "-lr_0.5-" in run_dir.name)
    ]

    for run_dir in run_dirs:
        print(run_dir.name)

    print(
        f"Found {len(run_dirs)} run(s) in {args.input_dir} which are not in cache",
        flush=True,
    )
    traces: dict[str, RunTrace] = {}
    skipped_runs: list[tuple[str, str]] = []
    event_files: list[tuple[str, Path]] = []

    for run_dir in run_dirs:
        try:
            trace = initialize_run_trace(run_dir)
        except ValueError as exc:
            skipped_runs.append((run_dir.name, str(exc)))
            continue

        run_event_files = discover_event_files(run_dir)
        if not run_event_files:
            trace.warnings.append("no TensorBoard event files")
        else:
            event_files.extend(
                (run_dir.name, event_file) for event_file in run_event_files
            )
        traces[run_dir.name] = trace

    for run_name, reason in skipped_runs:
        print(f"skip {run_name}: {reason}", flush=True)

    total_files = len(event_files)
    if total_files:
        worker_count = _resolve_worker_count(total_files, args.workers)
        print(
            f"Reading {total_files} event file(s) with {worker_count} worker process(es)",
            flush=True,
        )
        file_batches = _split_evenly(event_files, worker_count)
        partials_by_run: dict[str, list[dict]] = defaultdict(list)
        completed_files = 0
        print(_render_progress(completed_files, total_files), flush=True)

        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            future_to_batch_size = {
                pool.submit(
                    _read_event_file_batch,
                    batch,
                    args.max_vio_drop_points,
                    max_vio_top_points,
                ): len(batch)
                for batch in file_batches
                if batch
            }
            for future in as_completed(future_to_batch_size):
                completed_files += future_to_batch_size[future]
                print(
                    f"\r{_render_progress(completed_files, total_files)}",
                    end="",
                    flush=True,
                )
                for run_name, partials in future.result().items():
                    partials_by_run[run_name].extend(partials)
        print(flush=True)

        for run_name, trace in traces.items():
            if (
                run_name not in partials_by_run
                and "no TensorBoard event files" in trace.warnings
            ):
                continue
            apply_event_file_partials(
                trace,
                partials_by_run.get(run_name, []),
                max_vio_drop_points=args.max_vio_drop_points,
                max_vio_top_points=max_vio_top_points,
            )
    else:
        print(
            "No event files found; evaluating runs from directory metadata only.",
            flush=True,
        )

    evaluations = []
    for idx, trace in enumerate(
        sorted(traces.values(), key=lambda item: item.run_name), start=1
    ):
        criteria_results = evaluate_collapse(
            trace,
            warmup_steps=args.warmup_steps,
            max_vio_threshold_share=args.max_vio_threshold_share,
            max_vio_drop_points=args.max_vio_drop_points,
            max_vio_top_k=args.max_vio_top_k,
            activation_threshold=args.activation_threshold,
            loss_relative_increase=args.loss_relative_increase,
            loss_absolute_increase=args.loss_absolute_increase,
            min_points=args.persistent_points,
            min_step_span=args.persistent_step_span,
        )
        evaluation = evaluate_trace_status(
            trace,
            criteria_results,
            min_full_steps=args.min_full_steps,
        )
        evaluations.append(evaluation)
        print(
            f"[{idx}/{len(traces)}] {trace.run_name} "
            f"lr={evaluation.lr:g} status={evaluation.status} "
            f"max_step={evaluation.max_step} failed={evaluation.failed_criteria}",
            flush=True,
        )

    summary_rows = build_summary_rows(evaluations)
    summary_path, details_path, markdown_path = write_outputs(
        args.output_dir,
        args.result_name,
        summary_rows,
        evaluations,
    )
    print("\n" + markdown_path.read_text().rstrip(), flush=True)

    elapsed = perf_counter() - started
    print(f"\nSaved summary: {summary_path}", flush=True)
    print(f"Saved per-run details: {details_path}", flush=True)
    print(f"Saved markdown table: {markdown_path}", flush=True)
    print(f"Done in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
