"""Cross-run comparison: where do N runs actually differ?

Given N explicit runs, aligns every stability monitor on a common step grid,
compares each ``(tag, layer)`` across runs and ranks by divergence. The point is
to let the analysis cite concrete diffs — "run A's layer-3 gmm_down
``activations_amax`` ends at 47 vs B/C at 12 (3.9x spread)" — instead of
eyeballing side-by-side.

By default the comparison is at a single **reference step** — the latest step
*every* run reached, so a run that crashed at 5k isn't compared against one that
ran to 50k. Pass ``--steps 5000 10000 25000`` (explicit checkpoints) or
``--n-steps K`` (K evenly-spaced) to compare at several steps and see *when* the
runs start to diverge.

Outputs to ``<out-dir>/``:

* ``run_comparison_diffs.csv`` — headline: one row per ``(tag, group, layer)`` at
  the final comparison step, ranked by divergence, with each run's value
  (``val__<run>``), the spread, trajectory ``peak_*``, and (for exactly two runs)
  the signed A−B difference.
* ``run_comparison_trajectory.csv`` — long format over every comparison step
  (written only when >1 step) — shows divergence emerging.
* ``run_comparison_group_summary.csv`` — per monitor group: how much that family
  differs and which run tends to sit highest.
* ``max_vio_comparison.csv`` + ``max_vio_by_layer.csv`` — headline max_vio per run
  and per layer at the final comparison step.
* ``figures/`` — for ``instability.story``: ``00_max_vio_overlay`` +
  ``00b_max_vio_panels`` (side-by-side, one panel per run), per-group divergence,
  divergence trajectory, and for each top-K diverging series both an overlay
  (``02_top*``) and **side-by-side panels** (``03_top*_panels`` — run name as the
  panel title, shared y-axis, so comparison needs no giant legend). Run labels in
  every figure are the short aliases printed as the "run key".

Runs are compared by *exact tag name* — they must be the same model. Use
``--runs A B C`` (2+ runs required).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from instability.config import add_common_cli_args, resolve, select_runs
from instability.features import monitor_group
from instability.tb_helpers import (
    build_max_vio_series,
    load_run_scalars,
    parse_layer_from_tag,
)

_EPS = 1e-12
_MAX_GRID = 1000  # cap grid points used for peak/figure computations
_TOP_K_FIGURES = 12  # overlay figures for the most-divergent series


def _interp(steps: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if steps.size == 0:
        return np.full(grid.size, np.nan)
    return np.interp(grid.astype(float), steps.astype(float), vals.astype(float))


def _subsample(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.size <= n:
        return arr
    idx = np.linspace(0, arr.size - 1, n).round().astype(int)
    return arr[np.unique(idx)]


def load_runs(
    root: Path, runs: list[str], max_workers: int = 8
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for i, run in enumerate(runs, start=1):
        print(f"[{i}/{len(runs)}] loading {run}", flush=True)
        out[run] = load_run_scalars(str(root / run), max_workers=max_workers)
    return out


def _run_horizon(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> int:
    """Last step this run reached across any loaded tag."""
    last = 0
    for s, _ in series.values():
        if s.size:
            last = max(last, int(s[-1]))
    return last


def _candidate_tags(
    series_by_run: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> list[str]:
    """Tags that carry a known monitor group and appear in >= 2 runs."""
    counts: dict[str, int] = {}
    for series in series_by_run.values():
        for tag in series:
            counts[tag] = counts.get(tag, 0) + 1
    return sorted(
        t for t, c in counts.items() if c >= 2 and monitor_group(t) != "other"
    )


def compute_diffs(
    series_by_run: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    steps: list[int],
    grid: np.ndarray,
) -> pd.DataFrame:
    """Per (tag, group, layer, step) divergence across runs.

    Long format: one row per (tag, comparison step). ``divergence`` is the
    relative spread across runs *at that step*; ``peak_*`` columns are
    trajectory-level (over the full common grid) and repeat across a tag's
    rows so a single-step query still carries them.
    """
    runs = list(series_by_run)
    two_runs = len(runs) == 2
    rows: list[dict[str, Any]] = []
    for tag in _candidate_tags(series_by_run):
        present = {
            run: series[tag] for run, series in series_by_run.items() if tag in series
        }
        if len(present) < 2:
            continue
        group = monitor_group(tag)
        layer = parse_layer_from_tag(tag)

        # Trajectory-level peak spread over the full common grid (step-independent).
        aligned = {run: _interp(*sv, grid) for run, sv in present.items()}
        stack = np.vstack([aligned[run] for run in aligned])
        col_spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        if np.isfinite(col_spread).any():
            j = int(np.nanargmax(col_spread))
            peak_spread = float(col_spread[j])
            peak_step = int(grid[j])
        else:
            peak_spread, peak_step = math.nan, -1
        peak_rel = peak_spread / (float(np.nanmean(np.abs(stack))) + _EPS)

        for step in steps:
            step_vals = {
                run: float(_interp(sv[0], sv[1], np.array([step]))[0])
                for run, sv in present.items()
            }
            finite = {run: x for run, x in step_vals.items() if math.isfinite(x)}
            if len(finite) < 2:
                continue
            vals = list(finite.values())
            spread = max(vals) - min(vals)
            mean_abs = sum(abs(x) for x in vals) / len(vals) + _EPS
            row: dict[str, Any] = {
                "tag": tag,
                "group": group,
                "layer": layer,
                "step": int(step),
                "n_runs": len(finite),
                "divergence": spread / mean_abs,
                "spread": spread,
                "peak_spread": peak_spread,
                "peak_step": peak_step,
                "peak_rel_divergence": peak_rel,
                "highest_run": max(finite, key=finite.get),
                "lowest_run": min(finite, key=finite.get),
            }
            for run in runs:
                row[f"val__{run}"] = step_vals.get(run, math.nan)
            if two_runs:
                a, b = runs
                row["signed_diff"] = step_vals.get(a, math.nan) - step_vals.get(
                    b, math.nan
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["step", "divergence", "peak_rel_divergence"], ascending=False
        ).reset_index(drop=True)
    return df


def group_summary(diffs: pd.DataFrame) -> pd.DataFrame:
    if diffs.empty:
        return pd.DataFrame()
    rows = []
    for group, g in diffs.groupby("group"):
        top = g.iloc[0]
        rows.append(
            {
                "group": group,
                "n_tags": int(len(g)),
                "median_divergence": float(g["divergence"].median()),
                "max_divergence": float(g["divergence"].max()),
                "most_divergent_tag": top["tag"],
                "most_divergent_layer": top["layer"],
                "highest_run_mode": g["highest_run"].mode().iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values("max_divergence", ascending=False)


def max_vio_tables(
    series_by_run: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    ref_step: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Per-run max_vio headline + per-layer table at the reference step."""
    head_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    overlays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run, series in series_by_run.items():
        steps, vio, per_layer = build_max_vio_series(series)
        overlays[run] = (steps, vio)
        if steps.size == 0:
            head_rows.append({"run": run, "has_vio": False})
            continue
        vio_at_ref = float(_interp(steps, vio, np.array([ref_step]))[0])
        head_rows.append(
            {
                "run": run,
                "has_vio": True,
                "max_vio_at_ref": vio_at_ref,
                "max_vio_own_last": float(vio[-1]),
                "max_vio_max": float(np.nanmax(vio)),
                "max_vio_p95": float(np.nanpercentile(vio, 95)),
                "own_last_step": int(steps[-1]),
            }
        )
        for layer, (s, v) in per_layer.items():
            layer_rows.append(
                {
                    "run": run,
                    "layer": layer,
                    "vio_at_ref": float(_interp(s, v, np.array([ref_step]))[0]),
                    "vio_max": float(np.nanmax(v)),
                }
            )
    return pd.DataFrame(head_rows), pd.DataFrame(layer_rows), overlays


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def short_labels(runs: list[str]) -> dict[str, str]:
    """Strip the shared prefix/suffix tokens so legends/titles stay readable.

    Run names are often huge and 90% identical
    (``PapayaSeek-1B_10B-lr_explosion-z_loss-lr_0.005-PapayaSet_5_3_1-…``). We
    split on ``-``/``_`` and drop the leading and trailing tokens common to all
    runs, leaving just the distinguishing middle (``0.005``). Falls back to
    ``R1, R2, …`` if that collides or empties out.
    """
    if len(runs) < 2:
        return {r: r[:28] for r in runs}
    split = [re.split(r"[-_]", r) for r in runs]
    pre = 0
    while all(len(s) > pre + 1 and s[pre] == split[0][pre] for s in split):
        pre += 1
    suf = 0
    while all(
        len(s) > pre + suf + 1 and s[-1 - suf] == split[0][-1 - suf] for s in split
    ):
        suf += 1
    short: dict[str, str] = {}
    for r, s in zip(runs, split):
        mid = s[pre : len(s) - suf] if suf else s[pre:]
        short[r] = ("-".join(mid) or r)[:28]
    if len(set(short.values())) < len(runs):
        short = {r: f"R{i + 1}" for i, r in enumerate(runs)}
    return short


def _panels(plt, run_series, short, colors, title, out_path, ref_step=None) -> bool:
    """Side-by-side small multiples — one panel per run, shared y-axis.

    The run name is the panel *title* (short label), so there is no giant legend;
    a shared y-axis makes the runs directly comparable at a glance.
    """
    import math as _math

    items = [(r, s, v) for r, (s, v) in run_series.items() if s.size]
    if not items:
        return False
    n = len(items)
    ncols = min(4, n)
    nrows = _math.ceil(n / ncols)
    finite = [v[np.isfinite(v)] for _, _, v in items]
    allv = (
        np.concatenate([f for f in finite if f.size])
        if any(f.size for f in finite)
        else np.zeros(0)
    )
    ylim = None
    if allv.size:
        lo, hi = float(np.percentile(allv, 1)), float(np.percentile(allv, 99))
        ylim = (lo, hi) if hi > lo else None
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 2.5 * nrows),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for i, (r, s, v) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        ax.plot(s, v, color=colors[r], lw=1.4)
        if ref_step is not None:
            ax.axvline(ref_step, color="k", ls="--", lw=0.8, alpha=0.4)
        ax.set_title(short[r], fontsize=8)
        ax.tick_params(labelsize=7)
        if ylim:
            ax.set_ylim(ylim)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _save_figures(
    fig_dir: Path,
    headline: pd.DataFrame,
    trajectory: pd.DataFrame,
    series_by_run: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    grid: np.ndarray,
    mv_overlays: dict[str, tuple[np.ndarray, np.ndarray]],
    ref_step: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = list(series_by_run)
    short = short_labels(runs)
    cmap = plt.get_cmap("tab10")
    colors = {run: cmap(i % 10) for i, run in enumerate(runs)}

    # 00 — global max_vio overlay (short labels keep the legend small)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for run, (s, v) in mv_overlays.items():
        if s.size:
            ax.plot(s, v, label=short[run], color=colors[run], lw=1.5)
    ax.axvline(ref_step, color="k", ls="--", lw=1, alpha=0.5, label="ref step")
    ax.set(xlabel="step", ylabel="max_vio", title="max_vio — per run (overlay)")
    ax.legend(fontsize=8, loc="upper left", ncol=max(1, len(runs) // 8))
    fig.savefig(fig_dir / "00_max_vio_overlay.png", dpi=130)
    plt.close(fig)

    # 00b — max_vio side-by-side (one panel per run)
    _panels(
        plt,
        mv_overlays,
        short,
        colors,
        "max_vio — side by side per run",
        fig_dir / "00b_max_vio_panels.png",
        ref_step=ref_step,
    )

    if headline.empty:
        return

    # 01 — divergence by monitor group
    gs = group_summary(headline)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(gs))), constrained_layout=True)
    ax.barh(gs["group"], gs["max_divergence"], color="#d62728", alpha=0.85)
    ax.invert_yaxis()
    ax.set(
        xlabel="max relative divergence", title="Cross-run divergence by monitor group"
    )
    fig.savefig(fig_dir / "01_divergence_by_group.png", dpi=130)
    plt.close(fig)

    # 01b — divergence trajectory: when does each top series start to diverge?
    steps_present = sorted(trajectory["step"].unique()) if not trajectory.empty else []
    if len(steps_present) > 1:
        top_tags = headline.head(_TOP_K_FIGURES)["tag"].tolist()
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for tag in top_tags:
            sub = trajectory[trajectory["tag"] == tag].sort_values("step")
            label = f"{monitor_group(tag)} L{parse_layer_from_tag(tag)}"
            ax.plot(
                sub["step"], sub["divergence"], marker="o", ms=3, lw=1.3, label=label
            )
        ax.set(
            xlabel="step",
            ylabel="relative divergence",
            title="Divergence over training — top series",
        )
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(fig_dir / "01b_divergence_trajectory.png", dpi=130)
        plt.close(fig)

    # 02.. / 03..: each top-divergent series, as an overlay AND side-by-side panels
    for rank, (_, row) in enumerate(headline.head(_TOP_K_FIGURES).iterrows()):
        tag = row["tag"]
        layer = row["layer"]
        safe = tag.replace("/", "_").replace(".", "_")[:80]
        ttl = f"{row['group']}  L{layer}  (divergence={row['divergence']:.2f})\n{tag}"

        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        for run, series in series_by_run.items():
            if tag in series:
                ax.plot(
                    grid,
                    _interp(*series[tag], grid),
                    label=short[run],
                    color=colors[run],
                    lw=1.5,
                )
        ax.axvline(ref_step, color="k", ls="--", lw=1, alpha=0.4)
        ax.set(xlabel="step", ylabel="value", title=ttl)
        ax.legend(fontsize=8, ncol=max(1, len(runs) // 8))
        fig.savefig(fig_dir / f"02_top{rank:02d}_{safe}.png", dpi=120)
        plt.close(fig)

        run_series = {
            run: (grid, _interp(*series[tag], grid))
            for run, series in series_by_run.items()
            if tag in series
        }
        _panels(
            plt,
            run_series,
            short,
            colors,
            ttl,
            fig_dir / f"03_top{rank:02d}_panels_{safe}.png",
            ref_step=ref_step,
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _resolve_steps(
    grid: np.ndarray, ref_step: int, steps: list[int] | None, n_steps: int | None
) -> list[int]:
    """Resolve the comparison checkpoints.

    Explicit ``--steps`` win (clamped to the common range); else ``--n-steps``
    evenly-spaced checkpoints up to the reference step; else the single ref step.
    """
    lo = int(grid[0])
    if steps:
        kept = sorted({int(s) for s in steps if lo <= s <= ref_step})
        dropped = [s for s in steps if not (lo <= s <= ref_step)]
        if dropped:
            print(
                f"[compare] dropped out-of-range --steps {dropped} "
                f"(common range is [{lo}, {ref_step}])",
                flush=True,
            )
        return kept or [ref_step]
    if n_steps and n_steps > 1:
        raw = np.linspace(lo, ref_step, n_steps)
        snapped = sorted({int(grid[int(np.abs(grid - r).argmin())]) for r in raw})
        return snapped or [ref_step]
    return [ref_step]


def run_comparison(
    root: Path,
    out_dir: Path,
    runs: list[str],
    *,
    steps: list[int] | None = None,
    n_steps: int | None = None,
    max_workers: int = 8,
    make_figures: bool = True,
) -> pd.DataFrame:
    if len(runs) < 2:
        raise SystemExit("compare needs at least 2 runs (--runs A B ...).")

    series_by_run = load_runs(root, runs, max_workers=max_workers)

    horizons = {run: _run_horizon(s) for run, s in series_by_run.items()}
    nonzero = [h for h in horizons.values() if h > 0]
    if not nonzero:
        raise SystemExit("no run produced any steps — nothing to compare.")
    ref_step = min(nonzero)  # latest step every run reached

    union_steps: set[int] = set()
    for series in series_by_run.values():
        for s, _ in series.values():
            union_steps.update(int(x) for x in s if x <= ref_step)
    grid = _subsample(np.array(sorted(union_steps), dtype=np.int64), _MAX_GRID)
    if grid.size == 0:
        raise SystemExit("empty common step grid — runs have no overlapping range.")

    cmp_steps = _resolve_steps(grid, ref_step, steps, n_steps)
    final_step = cmp_steps[-1]
    print(
        f"[compare] reference step = {ref_step}  (per-run horizons: {horizons})\n"
        f"[compare] comparison steps = {cmp_steps}",
        flush=True,
    )

    trajectory = compute_diffs(series_by_run, cmp_steps, grid)
    headline = (
        trajectory[trajectory["step"] == final_step]
        .sort_values(["divergence", "peak_rel_divergence"], ascending=False)
        .reset_index(drop=True)
        if not trajectory.empty
        else trajectory
    )
    gsummary = group_summary(headline)
    mv_head, mv_layer, mv_overlays = max_vio_tables(series_by_run, final_step)

    out_dir.mkdir(parents=True, exist_ok=True)
    headline.to_csv(out_dir / "run_comparison_diffs.csv", index=False)
    gsummary.to_csv(out_dir / "run_comparison_group_summary.csv", index=False)
    mv_head.to_csv(out_dir / "max_vio_comparison.csv", index=False)
    mv_layer.to_csv(out_dir / "max_vio_by_layer.csv", index=False)
    if len(cmp_steps) > 1:
        trajectory.to_csv(out_dir / "run_comparison_trajectory.csv", index=False)

    if make_figures:
        _save_figures(
            out_dir / "figures",
            headline,
            trajectory,
            series_by_run,
            grid,
            mv_overlays,
            final_step,
        )

    # Run-key: short alias → full name, so the report can use compact labels
    # (figures already use these aliases as panel titles / legend entries).
    short = short_labels(runs)
    print("\n[compare] run key (use these short labels in the report):", flush=True)
    for run in runs:
        print(f"    {short[run]:<28}  =  {run}", flush=True)

    print(
        f"\n[compare] {len(headline)} comparable series across {len(runs)} runs "
        f"at step {final_step}. Top divergences:",
        flush=True,
    )
    for _, row in headline.head(8).iterrows():
        per_run = "  ".join(
            f"{r}={row[f'val__{r}']:.3g}" for r in runs if f"val__{r}" in row
        )
        print(
            f"  {row['group']:>16}  L{row['layer']}  div={row['divergence']:.2f}  "
            f"[{per_run}]  → {row['tag']}",
            flush=True,
        )
    print(f"\nWrote into {out_dir}", flush=True)
    return headline


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_cli_args(ap)
    ap.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        metavar="STEP",
        help="Explicit checkpoints to compare at (e.g. --steps 5000 10000 25000). "
        "Clamped to the range all runs cover. Default: the single latest shared step.",
    )
    ap.add_argument(
        "--n-steps",
        type=int,
        default=None,
        metavar="K",
        help="Compare at K evenly-spaced checkpoints up to the reference step "
        "(ignored if --steps is given). Use to see when runs start to diverge.",
    )
    ap.add_argument(
        "--no-figures", action="store_true", help="skip PNG overlays (CSV only)."
    )
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()
    cfg = resolve(args)

    runs = select_runs(cfg, cfg.eda) if (args.runs or args.runs_pattern) else None
    if not runs:
        raise SystemExit(
            "compare needs an explicit run set: pass --runs A B C "
            "or --runs-pattern 'glob' (2+ runs)."
        )
    if len(runs) < 2:
        raise SystemExit(f"compare needs 2+ runs, got {len(runs)}: {runs}")
    print(f"[compare] {len(runs)} runs:")
    for r in runs:
        print(f"  - {r}")

    run_comparison(
        cfg.tb_root,
        cfg.output_dir,
        runs,
        steps=args.steps,
        n_steps=args.n_steps,
        max_workers=args.max_workers,
        make_figures=not args.no_figures,
    )


if __name__ == "__main__":
    _cli()
