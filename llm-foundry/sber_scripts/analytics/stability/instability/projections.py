"""Per-projection (router / shared_expert / up / down) event-study + threshold plots.

Stage 3. Берёт `run_summary.csv` + `run_inventory.csv` от `eda.py`,
выборку ранов делает через `select_runs(cfg, cfg.projections)`. Для
каждого рана: derived projection-ratios, event-study около growth_onset,
threshold-классификатор `max_vio≥4`, и медианные профили в PNG.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from instability.config import (
    ResolvedConfig,
    add_common_cli_args,
    resolve,
    select_runs,
)
from instability.eda import interp
from instability.features import family, growth_onset, monitor_group, projection_group
from instability.tb_helpers import (
    build_max_vio_series,
    load_run_scalars,
    parse_layer_from_tag,
)
from instability.temporal import (
    add_derived_signals,
    transform_series,
)


def summary_tags_for_run(
    root: Path, run: str, max_files: int | None = None
) -> list[str]:
    """Return sorted list of relevant tag names for the run.

    Reuses ``tb_helpers.load_run_scalars`` so the parse hits the shared
    disk cache (same cache report.py / eda.py / temporal.py warm up).
    ``max_files`` is honored only as a sampling hint — we cache the full
    parse, then optionally subset.
    """
    s = load_run_scalars(str(root / run))
    return sorted(s.keys())


def tag_inventory(root: Path, out_dir: Path, runs: list[str]) -> pd.DataFrame:
    rows = []
    for i, run in enumerate(runs, 1):
        print(f"[tags {i}/{len(runs)}] {run}", flush=True)
        for tag in summary_tags_for_run(root, run):
            if not any(
                s in tag
                for s in ["activations_", "l2_norm/", "moe_", "router", "shared"]
            ):
                continue
            rows.append(
                {
                    "run": run,
                    "tag": tag,
                    "layer": parse_layer_from_tag(tag),
                    "monitor_group": monitor_group(tag),
                    "projection": projection_group(tag),
                }
            )
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(out_dir / "projection_tag_inventory.csv", index=False)
    return df


def load_projection_run(
    root: Path, run: str, max_workers: int = 4
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    # Use the shared default-prefix filter so this hits the same cache
    # as report.py / eda.py / temporal.py on the same TB.
    s = load_run_scalars(str(root / run), max_workers=max_workers)
    steps, y, _ = build_max_vio_series(s)
    signals: dict[str, np.ndarray] = {}
    if len(steps) == 0:
        return steps, y, signals
    for tag, (st, vals) in s.items():
        if "dispatcher_layer_experts_max_vio" in tag:
            continue
        if "old_moe_routing" in tag:
            continue
        if len(st) < 30 or np.nanstd(vals) < 1e-12:
            continue
        signals[tag] = interp(st, vals, steps)
    signals = add_projection_derived(add_derived_signals(signals))
    return steps, y, signals


def add_projection_derived(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(signals)
    by_layer: dict[int, dict[str, np.ndarray]] = {}
    for tag, vals in signals.items():
        if not tag.startswith("activations_"):
            continue
        layer = parse_layer_from_tag(tag)
        if layer is None:
            continue
        proj = projection_group(tag)
        stat = "amax" if "activations_amax" in tag else "amean"
        by_layer.setdefault(layer, {})[f"{stat}:{proj}"] = vals
    for layer, d in by_layer.items():
        for stat in ["amax", "amean"]:
            up = d.get(f"{stat}:up")
            down = d.get(f"{stat}:down")
            router = d.get(f"{stat}:router")
            if up is not None and down is not None:
                out[f"derived/projection_ratio/{stat}/up_over_down/layer_{layer}"] = (
                    np.where(np.abs(down) > 1e-15, up / down, np.nan)
                )
            if router is not None and up is not None:
                out[f"derived/projection_ratio/{stat}/router_over_up/layer_{layer}"] = (
                    np.where(np.abs(up) > 1e-15, router / up, np.nan)
                )
    return out


def threshold_summary(
    values: np.ndarray, y: np.ndarray, horizon: int = 5, threshold: float = 4.0
) -> dict[str, float]:
    vals = values.astype(float)
    yy = y.astype(float)
    n = len(vals)
    # `future[i]` = True iff max(y[i+1 : i+1+horizon]) >= threshold.
    # Vectorized via a forward-shifted running max over the next `horizon`
    # samples. NaN-safe; emits no warnings on empty slices.
    future = np.zeros(n, dtype=bool)
    if n > 1 and horizon > 0:
        crossed = np.where(np.isfinite(yy) & (yy >= threshold))[0]
        for k in crossed:
            lo = max(0, k - horizon)
            future[lo:k] = True
    m = np.isfinite(vals)
    if m.sum() < 30 or future[m].sum() < 2:
        return {}
    x = vals[m]
    ev = future[m]
    qs = [0.5, 0.75, 0.9, 0.95]
    best: dict[str, float] = {"best_f1": -1.0}
    for q in qs:
        thr = float(np.nanquantile(x, q))
        pred = x >= thr
        tp = float(np.sum(pred & ev))
        fp = float(np.sum(pred & ~ev))
        fn = float(np.sum(~pred & ev))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        lift = precision / max(float(np.mean(ev)), 1e-12)
        if f1 > best["best_f1"]:
            best = {
                "best_quantile": q,
                "best_threshold": thr,
                "best_precision": precision,
                "best_recall": recall,
                "best_f1": f1,
                "best_lift": lift,
                "event_rate": float(np.mean(ev)),
                "n": float(m.sum()),
            }
    return best


def projection_temporal_tables(
    root: Path, out_dir: Path, runs: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    thresh_rows = []
    for i, run in enumerate(runs, 1):
        print(f"[projection {i}/{len(runs)}] {run}", flush=True)
        steps, y, signals = load_projection_run(root, run)
        if len(steps) < 50:
            continue
        onset = growth_onset(steps, y)
        if np.isfinite(onset):
            pre = (steps >= onset - 500) & (steps < onset)
            base = steps < max(steps[0] + 500, onset - 1000)
        else:
            pre = np.arange(len(steps)) >= max(0, len(steps) - len(steps) // 5)
            base = np.arange(len(steps)) < max(5, len(steps) // 5)
        for tag, vals in signals.items():
            proj = projection_group(tag)
            fam = family(tag)
            layer = parse_layer_from_tag(tag)
            if fam in {"other", "routing_load"}:
                continue
            tr = transform_series(vals)
            for rep in [
                "level_log",
                "level_rel_early",
                "rolling_slope_log",
                "rolling_vol_log",
                "rolling_z",
            ]:
                x = tr[rep]
                b = x[base]
                p = x[pre]
                bmean = float(np.nanmean(b)) if np.isfinite(b).sum() else math.nan
                pmean = float(np.nanmean(p)) if np.isfinite(p).sum() else math.nan
                bstd = float(np.nanstd(b)) if np.isfinite(b).sum() else math.nan
                event_rows.append(
                    {
                        "run": run,
                        "tag": tag,
                        "family": fam,
                        "projection": proj,
                        "layer": layer,
                        "representation": rep,
                        "onset_step": onset,
                        "pre_minus_baseline": pmean - bmean
                        if np.isfinite(pmean) and np.isfinite(bmean)
                        else math.nan,
                        "pre_z_vs_baseline": (pmean - bmean) / (bstd + 1e-12)
                        if np.isfinite(pmean)
                        and np.isfinite(bmean)
                        and np.isfinite(bstd)
                        else math.nan,
                    }
                )
            if fam in {
                "param_norm",
                "update_param_ratio",
                "grad_param_ratio",
                "effective_lr",
                "update_grad_gap",
            } or tag.startswith("activations_amax"):
                ts = threshold_summary(vals, y, horizon=5, threshold=4.0)
                if ts:
                    ts.update(
                        {
                            "run": run,
                            "tag": tag,
                            "family": fam,
                            "projection": proj,
                            "layer": layer,
                        }
                    )
                    thresh_rows.append(ts)
    event = pd.DataFrame(event_rows)
    thresh = pd.DataFrame(thresh_rows)
    event.to_csv(out_dir / "projection_event_summary.csv", index=False)
    thresh.to_csv(out_dir / "absolute_threshold_summary.csv", index=False)
    return event, thresh


PLOT_SPECS: list[tuple[str, str]] = [
    ("update_norm", "moe_block"),
    ("update_param_ratio", "moe_block"),
    ("param_norm", "moe_block"),
    ("effective_lr", "moe_block"),
    ("update_grad_gap", "moe_block"),
    ("activation_amax", "up"),
    ("activation_amax", "down"),
    ("activation_amax", "router"),
    ("activation_amax", "shared_expert"),
    ("activation_amean", "up"),
    ("activation_amean", "down"),
    ("activation_amean", "router"),
    ("activation_amean", "shared_expert"),
    ("update_activation_ratio", "up"),
    ("update_activation_ratio", "down"),
    ("update_activation_ratio", "moe_block"),
    ("grad_activation_ratio", "up"),
    # residual-connection tracking: how much attn vs mlp write into the residual,
    # and the residual stream itself, split by sublayer
    ("residual_contribution", "attention"),
    ("residual_contribution", "moe_block"),
    ("residual_stream", "attention"),
    ("residual_stream", "moe_block"),
    ("layer_contrast", "other"),
]


def make_event_plots(root: Path, out_dir: Path, runs: list[str]) -> pd.DataFrame:
    plots = out_dir / "event_study_plots"
    plots.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, run in enumerate(runs, 1):
        print(f"[plot-load {i}/{len(runs)}] {run}", flush=True)
        steps, y, signals = load_projection_run(root, run)
        if len(steps) < 50:
            continue
        onset = growth_onset(steps, y)
        if not np.isfinite(onset):
            continue
        rel = steps - onset
        grid = np.arange(-1000, 1001, 100)
        for fam, proj in PLOT_SPECS:
            candidates = []
            for tag, vals in signals.items():
                if family(tag) != fam:
                    continue
                if proj != "other" and projection_group(tag) != proj:
                    continue
                candidates.append((tag, vals))
            if not candidates:
                continue
            mat_z = []
            mat_level = []
            for _, vals in candidates:
                tr = transform_series(vals)
                mat_z.append(
                    np.interp(
                        grid,
                        rel.astype(float),
                        tr["rolling_z"],
                        left=np.nan,
                        right=np.nan,
                    )
                )
                mat_level.append(
                    np.interp(
                        grid,
                        rel.astype(float),
                        tr["level_rel_early"],
                        left=np.nan,
                        right=np.nan,
                    )
                )
            for representation, mat in [
                ("rolling_z", mat_z),
                ("level_rel_early", mat_level),
            ]:
                prof = np.nanmedian(np.vstack(mat), axis=0)
                for g, v in zip(grid, prof):
                    rows.append(
                        {
                            "run": run,
                            "family": fam,
                            "projection": proj,
                            "representation": representation,
                            "rel_step": g,
                            "value": v,
                        }
                    )
    prof = pd.DataFrame(rows)
    prof.to_csv(out_dir / "event_plot_profiles.csv", index=False)
    if prof.empty:
        return prof
    for (fam, proj, representation), g in prof.groupby(
        ["family", "projection", "representation"]
    ):
        p = g.groupby("rel_step")["value"].agg(["median", "count"]).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(p["rel_step"], p["median"], marker="o", linewidth=1.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{fam} / {proj}: {representation} around growth_onset")
        ax.set_xlabel("train step relative to growth_onset")
        ax.set_ylabel(f"median {representation} across runs/tags")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / f"{fam}__{proj}__{representation}.png", dpi=150)
        plt.close(fig)
    return prof


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_cli_args(ap)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument(
        "--skip-inventory",
        action="store_true",
        help="Не пересобирать projection_tag_inventory.csv (медленно)",
    )
    args = ap.parse_args()
    cfg: ResolvedConfig = resolve(args)
    out = cfg.output_dir

    summary_csv = out / "run_summary.csv"
    inv_csv = out / "run_inventory.csv"
    summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else None
    inv_df = pd.read_csv(inv_csv) if inv_csv.exists() else None
    if cfg.runs_override is None and cfg.runs_pattern is None and summary_df is None:
        raise SystemExit(
            f"{summary_csv} not found. Run `python -m instability.eda` first or pass --runs/--runs-pattern."
        )

    runs = select_runs(cfg, cfg.projections, inv_df=inv_df, summary_df=summary_df)
    if not runs:
        raise SystemExit("Список ранов для анализа пуст.")
    print(f"[projections] {len(runs)} runs:")
    for r in runs:
        print(f"  - {r}")

    if not args.skip_inventory:
        tag_inventory(cfg.tb_root, out, runs)
    projection_temporal_tables(cfg.tb_root, out, runs)
    if not args.skip_plots:
        make_event_plots(cfg.tb_root, out, runs)


if __name__ == "__main__":
    _cli()
