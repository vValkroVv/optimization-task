"""Cross-run comparison of analyze.json checkpoints — where do N runs differ?

The ``report`` subcommand already diffs *iterations of one run* (DELTAS /
trajectory). This compares *different runs* at a comparable checkpoint: it aligns
tensors by FQN across the given analyze.json files and ranks each
``(tensor, metric)`` by how much the runs disagree, so the analysis can cite
concrete diffs — "``spectral.stable_rank`` of ``moe_expert/gate`` at layer 17 is
1.8 in RUN_A vs 28 in RUN_B/C" — instead of reading three reports side by side.

Each file is one run. Run / iteration labels are parsed from the path
(``…/<run>/<iter>/analyze.json``); pass ``--labels`` to override run names. When
the files span more than one shared iteration the comparison is repeated per
iteration and a divergence *trajectory* is emitted, so you can see *when* the
runs started to diverge — the weights analog of ``instability.compare --steps``.

Outputs (stdout text report by default; ``--csv-dir`` also writes):

* ``run_comparison_diffs.csv``       — per (tensor, metric) at the latest shared
  iteration, ranked by divergence, one value column per run.
* ``run_comparison_group_summary.csv`` — per (role, proj, block, metric): how
  much that family differs and which run sits highest.
* ``run_comparison_trajectory.csv``  — only when >1 shared iteration: group-level
  divergence per iteration.

Runs are compared by exact tensor FQN — they must be the same model.
``--runs A B C`` style: pass 2+ analyze.json files.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

from weights.report import (
    ANALYZE_CURATED,
    _analyze_record_metrics,
    _fmt,
    _hdr,
    _short_iter_label,
    _table,
    detect_kind,
)

_EPS = 1e-12
_ITER_NUM_RE = re.compile(r"(?:ba|iter[_-]?|step[_-]?|ep\d+_ba)(\d+)")


def _run_iter_labels(path: Path, override: str | None) -> tuple[str, str]:
    """Best-effort (run, iter) from a path like ``…/<run>/<iter>/analyze.json``.

    ``iter`` is the first ``ba###`` / ``ep#_ba###`` / ``iter_###`` token found in
    the path; ``run`` is the directory just above that iter dir (or the parent if
    no iter token). ``override`` forces the run label.
    """
    parts = path.parts
    iter_label = "single"
    iter_idx = None
    for i, part in enumerate(parts):
        if _ITER_NUM_RE.search(part):
            iter_label = _short_iter_label(part)
            iter_idx = i
            break
    if override:
        run = override
    elif iter_idx is not None and iter_idx > 0:
        run = parts[iter_idx - 1]
    elif len(parts) >= 2:
        run = parts[-2]
    else:
        run = path.stem
    return run, iter_label


def _iter_sort_key(it: str) -> tuple[int, Any]:
    m = _ITER_NUM_RE.search(it)
    return (0, int(m.group(1))) if m else (1, it)


def load_files(
    files: list[Path], labels: list[str] | None
) -> list[tuple[str, str, dict[str, dict]]]:
    """Return [(run, iter, tensors)] for every analyze.json passed."""
    if labels and len(labels) != len(files):
        raise SystemExit(
            f"--labels count ({len(labels)}) must match files ({len(files)})"
        )
    loaded: list[tuple[str, str, dict[str, dict]]] = []
    for i, p in enumerate(files):
        import json

        blob = json.loads(p.read_text())
        if detect_kind(blob) != "analyze":
            raise SystemExit(f"{p}: not an analyze.json (compare needs analyze files)")
        run, it = _run_iter_labels(p, labels[i] if labels else None)
        loaded.append((run, it, blob.get("tensors", {})))
    return loaded


# ---------------------------------------------------------------------------
# Core divergence
# ---------------------------------------------------------------------------
def _record_id(rec: dict) -> tuple[str, str, str, Any]:
    proj = str(rec.get("projection") or rec.get("proj") or "-")
    block = str(rec.get("block") or "main")
    return str(rec.get("role", "?")), proj, block, rec.get("layer", "-")


def compute_diffs(
    run_tensors: dict[str, dict[str, dict]], *, all_metrics: bool
) -> list[dict[str, Any]]:
    """Per (tensor, metric) cross-run divergence at one iteration.

    Divergence is the relative spread ``(max-min)/(mean|·|)`` of the metric across
    the runs that have the tensor. ``highest_run`` / ``lowest_run`` + the metric's
    direction tell you which run is the bad one.
    """
    runs = list(run_tensors)
    curated = {m: (d, lbl) for m, d, lbl in ANALYZE_CURATED}

    # tensor FQN -> {run: flat metric dict}
    common: dict[str, dict[str, dict[str, float]]] = {}
    for run, tensors in run_tensors.items():
        for name, rec in tensors.items():
            common.setdefault(name, {})[run] = _analyze_record_metrics(rec)

    rows: list[dict[str, Any]] = []
    for name, per_run in common.items():
        if len(per_run) < 2:
            continue
        # representative record for role/proj/layer/block
        ref_rec = next(rt[name] for rt in run_tensors.values() if name in rt)
        role, proj, block, layer = _record_id(ref_rec)
        metric_names = (
            sorted({m for d in per_run.values() for m in d})
            if all_metrics
            else [m for m, _, _ in ANALYZE_CURATED]
        )
        for metric in metric_names:
            vals = {r: d[metric] for r, d in per_run.items() if metric in d}
            finite = {r: v for r, v in vals.items() if math.isfinite(v)}
            if len(finite) < 2:
                continue
            v = list(finite.values())
            spread = max(v) - min(v)
            mean_abs = sum(abs(x) for x in v) / len(v) + _EPS
            direction = curated.get(metric, ("high", ""))[0]
            row: dict[str, Any] = {
                "tensor": name,
                "role": role,
                "proj": proj,
                "block": block,
                "layer": layer,
                "metric": metric,
                "direction": direction,
                "divergence": spread / mean_abs,
                "spread": spread,
                "highest_run": max(finite, key=finite.get),
                "lowest_run": min(finite, key=finite.get),
                # the run that's *worst* by the metric's own direction
                "worst_run": (
                    min(finite, key=finite.get)
                    if direction == "low"
                    else max(finite, key=lambda r: abs(finite[r]))
                    if direction == "abs"
                    else max(finite, key=finite.get)
                ),
            }
            for r in runs:
                row[f"val__{r}"] = vals.get(r, math.nan)
            rows.append(row)
    rows.sort(key=lambda r: -r["divergence"])
    return rows


def group_summary(diffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (role, proj, block, metric): median/max divergence + dominant run."""
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for r in diffs:
        buckets.setdefault((r["role"], r["proj"], r["block"], r["metric"]), []).append(
            r
        )
    out = []
    for (role, proj, block, metric), rs in buckets.items():
        divs = sorted(r["divergence"] for r in rs)
        median = divs[len(divs) // 2]
        worst_runs = [r["worst_run"] for r in rs]
        mode = max(set(worst_runs), key=worst_runs.count)
        out.append(
            {
                "role": role,
                "proj": proj,
                "block": block,
                "metric": metric,
                "n_tensors": len(rs),
                "median_divergence": median,
                "max_divergence": max(divs),
                "worst_run_mode": mode,
            }
        )
    out.sort(key=lambda r: -r["max_divergence"])
    return out


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def _print_report(
    runs: list[str],
    it: str,
    diffs: list[dict[str, Any]],
    gsummary: list[dict[str, Any]],
    top_n: int,
) -> None:
    _hdr(f"CROSS-RUN COMPARISON  @ {it}   ({len(runs)} runs: {', '.join(runs)})")
    if not diffs:
        print("  (no common tensors/metrics across >= 2 runs)")
        return

    print("\n  -- most divergent families (role / proj / block × metric) --")
    rows = [
        [
            f"{g['role']}/{g['proj']}/{g['block']}",
            g["metric"],
            str(g["n_tensors"]),
            _fmt(g["median_divergence"]),
            _fmt(g["max_divergence"]),
            g["worst_run_mode"],
        ]
        for g in gsummary[:top_n]
    ]
    _table(rows, ["group", "metric", "n", "med div", "max div", "worst run"])

    # Per curated metric, the top divergent tensors with per-run values.
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for d in diffs:
        by_metric.setdefault(d["metric"], []).append(d)
    for metric, _dir, label in ANALYZE_CURATED:
        ds = by_metric.get(metric)
        if not ds:
            continue
        # Skip a metric whose value is identical across runs for every tensor —
        # nothing to compare there.
        if all(d["spread"] == 0 for d in ds):
            continue
        print(f"\n  -- {metric}  ({label}; worse = {_dir}) --")
        rows = []
        for d in ds[:top_n]:
            rows.append(
                [_fmt(d["divergence"])]
                + [_fmt(d[f"val__{r}"]) for r in runs]
                + [
                    str(d["layer"]),
                    f"{d['role']}/{d['proj']}/{d['block']}",
                    d["worst_run"],
                ]
            )
        _table(rows, ["div"] + runs + ["layer", "group", "worst"])


def _write_csvs(
    csv_dir: Path,
    runs: list[str],
    diffs: list[dict[str, Any]],
    gsummary: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        cols = list(rows[0])
        with (csv_dir / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    _dump("run_comparison_diffs.csv", diffs)
    _dump("run_comparison_group_summary.csv", gsummary)
    if trajectory:
        _dump("run_comparison_trajectory.csv", trajectory)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(
    files: list[Path],
    *,
    labels: list[str] | None = None,
    top_n: int = 12,
    all_metrics: bool = False,
    csv_dir: Path | None = None,
    out: Path | None = None,
) -> None:
    if len(files) < 2:
        raise SystemExit("compare needs at least 2 analyze.json files.")

    loaded = load_files(files, labels)
    runs = list(dict.fromkeys(run for run, _, _ in loaded))
    if len(runs) < 2:
        raise SystemExit(
            f"compare needs >= 2 distinct runs; got {runs}. "
            "Use --labels A B ... to disambiguate same-named parent dirs."
        )

    # Group by iteration; only iterations shared by >= 2 runs are comparable.
    by_iter: dict[str, dict[str, dict[str, dict]]] = {}
    for run, it, tensors in loaded:
        by_iter.setdefault(it, {})[run] = tensors
    shared = sorted(
        (it for it, rt in by_iter.items() if len(rt) >= 2), key=_iter_sort_key
    )
    if not shared:
        raise SystemExit(
            "no iteration is shared by >= 2 runs — pass the same checkpoint per run "
            "(or --labels to fix run identity)."
        )
    final_it = shared[-1]

    # Trajectory: group-level max divergence per shared iteration.
    trajectory: list[dict[str, Any]] = []
    if len(shared) > 1:
        for it in shared:
            for g in group_summary(compute_diffs(by_iter[it], all_metrics=all_metrics)):
                trajectory.append({"iter": it, **g})

    diffs = compute_diffs(by_iter[final_it], all_metrics=all_metrics)
    gsummary = group_summary(diffs)

    streams = [sys.stdout]
    fh = None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fh = out.open("w")
        streams.append(fh)

    class _Tee:
        def write(self, s: str) -> int:
            for st in streams:
                st.write(s)
            return len(s)

        def flush(self) -> None:
            for st in streams:
                st.flush()

    orig = sys.stdout
    sys.stdout = _Tee()  # type: ignore[assignment]
    try:
        if len(shared) > 1:
            print(f"[compare] shared iterations: {shared}  (headline @ {final_it})")
        _print_report(runs, final_it, diffs, gsummary, top_n)
        if len(shared) > 1:
            _print_trajectory(runs, shared, trajectory, top_n)
    finally:
        sys.stdout = orig
        if fh:
            fh.close()

    if csv_dir:
        _write_csvs(csv_dir, runs, diffs, gsummary, trajectory)
        print(f"\n[compare] wrote CSVs into {csv_dir}", flush=True)


def _print_trajectory(
    runs: list[str], shared: list[str], trajectory: list[dict[str, Any]], top_n: int
) -> None:
    """Show how the top diverging families evolve across shared iterations."""
    if not trajectory:
        return

    # Pick the families with the largest divergence at any iteration.
    def key(r):
        return (r["role"], r["proj"], r["block"], r["metric"])

    peak: dict[tuple, float] = {}
    for r in trajectory:
        peak[key(r)] = max(peak.get(key(r), 0.0), r["max_divergence"])
    top_families = sorted(peak, key=lambda k: -peak[k])[:top_n]

    _hdr("DIVERGENCE TRAJECTORY  (max divergence per family across iterations)")
    by_fam_iter = {(key(r), r["iter"]): r["max_divergence"] for r in trajectory}
    rows = []
    for fam in top_families:
        role, proj, block, metric = fam
        rows.append(
            [f"{role}/{proj}/{block}", metric]
            + [_fmt(by_fam_iter.get((fam, it), float("nan"))) for it in shared]
        )
    _table(rows, ["group", "metric"] + shared)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="weights compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="2+ analyze.json files, one per run (optionally per (run, iter)).",
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="run label per file (overrides path-derived run names).",
    )
    ap.add_argument("--top-n", type=int, default=12, help="rows per table (default 12)")
    ap.add_argument(
        "--all-metrics",
        action="store_true",
        help="compare every numeric metric, not just the curated diagnostic set.",
    )
    ap.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="also write run_comparison_*.csv into this directory.",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="also tee the text report to this file."
    )
    args = ap.parse_args(argv)
    run(
        args.files,
        labels=args.labels,
        top_n=args.top_n,
        all_metrics=args.all_metrics,
        csv_dir=args.csv_dir,
        out=args.out,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
