"""CLI: build a per-layer metrics dashboard from TensorBoard runs.

Usage examples
--------------
# Single run
python3 build_dashboard.py out/ pretrain-v1/

# Two-run comparison
python3 build_dashboard.py out/ pretrain-v1/ pretrain-v2/

# All runs in a directory (one subplot per subdirectory)
python3 build_dashboard.py out/ --runs-dir /path/to/experiments/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined per-layer metrics dashboard from TensorBoard run directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory where the HTML is written."
    )
    parser.add_argument(
        "run_paths",
        type=Path,
        nargs="*",
        help="TensorBoard run directories (1 or more).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        metavar="DIR",
        help="Directory whose immediate subdirectories are treated as run paths.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="Human-readable labels for each run (must match number of run_paths).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        metavar="N",
        help="Subsample slider positions to at most N steps (default: 500). "
        "Lower values reduce HTML size and improve browser performance with many runs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Max CPU workers for parallel event-file parsing (default: all cores up to 16).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.runs_dir and args.run_paths:
        raise SystemExit("error: run_paths and --runs-dir are mutually exclusive")
    if args.runs_dir:
        run_paths = sorted(p for p in args.runs_dir.iterdir() if p.is_dir())
        if not run_paths:
            raise SystemExit(f"error: no subdirectories found in {args.runs_dir}")
        print(f"Found {len(run_paths)} runs")
    else:
        run_paths = args.run_paths or []
        if not run_paths:
            raise SystemExit("error: provide run_paths or --runs-dir")
        print(f"Found {run_paths} path")

    if args.labels and len(args.labels) != len(run_paths):
        raise SystemExit(
            f"error: --labels count ({len(args.labels)}) must match run count ({len(run_paths)})"
        )

    from dashboard.builder import build_dashboard_for_paths

    build_dashboard_for_paths(
        args.output_dir,
        run_paths,
        labels=args.labels,
        max_steps=args.max_steps,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    from time import perf_counter

    _t0 = perf_counter()
    main()
    print(f"\nDone in {perf_counter() - _t0:.2f}s", flush=True)
