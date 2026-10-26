"""Run inventory + per-run max_vio summary + monitor↔max_vio correlations.

Stage 1. Walks the TB-root, parses feature flags from run names/configs,
builds the global ``max_vio`` series for each run, and computes
correlations between every MoE/router/activation/grad scalar and ``max_vio``.
Outputs CSVs to ``<output_dir>/`` (set via ``--out-dir``).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from instability.config import (
    ResolvedConfig,
    add_common_cli_args,
    resolve,
)
from instability.features import (
    FEATURE_PATTERNS,
    KEY_CONFIG_PATHS,
    flatten_keys,
    growth_onset,
    infer_features_from_name,
    monitor_group,
    nested_get,
    read_config,
)
from instability.tb_helpers import (
    argmax_layer_histogram,
    build_max_vio_series,
    load_run_scalars,
    parse_layer_from_tag,
)


def run_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith("__")
    ):
        events = list(run_dir.glob("events.out.tfevents*"))
        cfg = read_config(run_dir)
        row: dict[str, Any] = {
            "run": run_dir.name,
            "path": str(run_dir),
            "n_event_files": len(events),
            "has_config": bool(cfg),
        }
        row.update(infer_features_from_name(run_dir.name))
        for label, path in KEY_CONFIG_PATHS:
            row[label] = nested_get(cfg, path)
        flat = flatten_keys(cfg)
        for key, value in flat.items():
            low = key.lower()
            if any(
                s in low
                for s in [
                    "router",
                    "expert",
                    "norm",
                    "clipping",
                    "warmup",
                    "liger",
                    "yarn",
                ]
            ):
                if key not in row and isinstance(
                    value, (str, int, float, bool, type(None))
                ):
                    row[f"cfg.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def interp(
    src_steps: np.ndarray, src_vals: np.ndarray, query_steps: np.ndarray
) -> np.ndarray:
    if src_steps.size == 0:
        return np.full(query_steps.size, np.nan)
    return np.interp(
        query_steps.astype(float), src_steps.astype(float), src_vals.astype(float)
    )


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return math.nan
    x, y = a[m], b[m]
    if np.nanstd(x) < 1e-15 or np.nanstd(y) < 1e-15:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def first_threshold_step(
    steps: np.ndarray, values: np.ndarray, threshold: float
) -> float:
    idx = np.where(values >= threshold)[0]
    return float(steps[int(idx[0])]) if idx.size else math.nan


def tag_value(
    series: dict[str, tuple[np.ndarray, np.ndarray]], tag: str, steps: np.ndarray
) -> np.ndarray:
    if tag not in series:
        return np.full(steps.size, np.nan)
    return interp(*series[tag], steps)


def summarize_one_run(
    root: Path, run: str, max_workers: int = 8
) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = root / run
    series = load_run_scalars(str(run_dir), max_workers=max_workers)
    steps, max_vio, per_layer = build_max_vio_series(series)
    row: dict[str, Any] = {
        "run": run,
        "n_tags_loaded": len(series),
        "has_vio": steps.size > 0,
        "n_vio_steps": int(steps.size),
        "first_step": float(steps[0]) if steps.size else math.nan,
        "last_step": float(steps[-1]) if steps.size else math.nan,
    }
    corr_rows: list[dict[str, Any]] = []
    if steps.size == 0:
        return row, pd.DataFrame(corr_rows)

    row.update(
        {
            "max_vio_min": float(np.nanmin(max_vio)),
            "max_vio_median": float(np.nanmedian(max_vio)),
            "max_vio_p95": float(np.nanpercentile(max_vio, 95)),
            "max_vio_p99": float(np.nanpercentile(max_vio, 99)),
            "max_vio_max": float(np.nanmax(max_vio)),
            "max_vio_last": float(max_vio[-1]),
            "max_vio_growth_x": float(
                max_vio[-1] / np.nanmedian(max_vio[: max(5, steps.size // 10)])
            ),
            "first_vio_ge_1_5": first_threshold_step(steps, max_vio, 1.5),
            "first_vio_ge_2": first_threshold_step(steps, max_vio, 2.0),
            "first_vio_ge_4": first_threshold_step(steps, max_vio, 4.0),
            "growth_onset_step": growth_onset(steps, max_vio),
        }
    )
    _, argmax = argmax_layer_histogram(series)
    if argmax:
        top_layer, top_share = sorted(argmax.items(), key=lambda kv: -kv[1])[0]
        row["argmax_vio_top_layer"] = top_layer
        row["argmax_vio_top_layer_share"] = top_share

    layer_values = []
    for layer, (st, va) in per_layer.items():
        v = interp(st, va, steps)
        row[f"vio_L{layer}_max"] = float(np.nanmax(v))
        row[f"vio_L{layer}_last"] = float(v[-1])
        layer_values.append((layer, v))
    if layer_values:
        early = [np.nanmean(v[: max(5, len(v) // 10)]) for _, v in layer_values]
        late = [
            np.nanmean(v[max(0, len(v) - max(5, len(v) // 10)) :])
            for _, v in layer_values
        ]
        row["vio_layer_mean_early"] = float(np.nanmean(early))
        row["vio_layer_mean_late"] = float(np.nanmean(late))
        row["vio_layer_mean_growth_x"] = (
            float(np.nanmean(late) / np.nanmean(early))
            if np.nanmean(early)
            else math.nan
        )

    metric_specs: list[tuple[str, str]] = []
    for tag in series:
        low = tag.lower()
        if "dispatcher_layer_experts_max_vio" in tag:
            continue
        if any(
            s in low
            for s in [
                "moe_seq_aux_loss",
                "moe_z_loss_router",
                "old_moe_routing",
                "l2_norm/grad",
                "l2_norm/update",
                "activations_amax",
                "activations_amean",
            ]
        ):
            metric_specs.append((tag, tag))

    for tag, _label in metric_specs:
        vals = tag_value(series, tag, steps)
        corr = safe_corr(vals, max_vio)
        if not np.isfinite(corr):
            continue
        layer = parse_layer_from_tag(tag)
        corr_rows.append(
            {
                "run": run,
                "tag": tag,
                "layer": layer,
                "corr_with_max_vio": corr,
                "metric_max": float(np.nanmax(vals)),
                "metric_p95": float(np.nanpercentile(vals, 95)),
                "metric_last": float(vals[-1]),
            }
        )

    return row, pd.DataFrame(corr_rows)


def feature_table(
    inv: pd.DataFrame, summary: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    df = inv.merge(summary, on="run", how="left")
    feat_cols = [name for name, _ in FEATURE_PATTERNS if name in df]
    rows = []
    for feat in feat_cols:
        mask = df[feat].fillna(False).astype(bool)
        rows.append(
            {
                "feature": feat,
                "n_runs": int(mask.sum()),
                "n_with_vio": int(df.loc[mask, "has_vio"].fillna(False).sum())
                if "has_vio" in df
                else 0,
                "median_max_vio_max": float(df.loc[mask, "max_vio_max"].median())
                if mask.any()
                else math.nan,
                "median_last_step": float(df.loc[mask, "last_step"].median())
                if mask.any()
                else math.nan,
                "share_vio_ge_2": float((df.loc[mask, "max_vio_max"] >= 2).mean())
                if mask.any()
                else math.nan,
                "share_vio_ge_4": float((df.loc[mask, "max_vio_max"] >= 4).mean())
                if mask.any()
                else math.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["n_runs", "feature"], ascending=[False, True])
    out.to_csv(out_dir / "feature_summary.csv", index=False)
    return out


def build_summary(
    root: Path,
    out_dir: Path,
    *,
    runs: list[str] | None = None,
    min_event_files: int = 2,
    max_runs: int | None = None,
    max_workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Если `runs` задан — анализируем ровно их (выход: sample_*.csv).
    Иначе берём все раны с `n_event_files >= min_event_files`.
    """
    inv = run_inventory(root)
    sample_mode = runs is not None
    if sample_mode:
        candidates_names = [r for r in dict.fromkeys(runs) if (root / r).is_dir()]
    else:
        candidates = inv[inv["n_event_files"] >= min_event_files].sort_values(
            ["n_event_files", "run"], ascending=[False, True]
        )
        if max_runs:
            candidates = candidates.head(max_runs)
        candidates_names = candidates["run"].tolist()

    rows = []
    corr_parts = []
    for i, run in enumerate(candidates_names, start=1):
        print(f"[{i}/{len(candidates_names)}] {run}", flush=True)
        row, corr = summarize_one_run(root, run, max_workers=max_workers)
        rows.append(row)
        if not corr.empty:
            corr_parts.append(corr)
    summary = pd.DataFrame(rows)
    corr_df = pd.concat(corr_parts, ignore_index=True) if corr_parts else pd.DataFrame()

    inv.to_csv(out_dir / "run_inventory.csv", index=False)
    summary_name = "sample_run_summary.csv" if sample_mode else "run_summary.csv"
    corr_name = (
        "sample_metric_correlations.csv" if sample_mode else "metric_correlations.csv"
    )
    summary.to_csv(out_dir / summary_name, index=False)
    corr_df.to_csv(out_dir / corr_name, index=False)
    return inv, summary, corr_df


def feature_monitor_correlation_summary(
    out_dir: Path,
    *,
    min_runs_with_feature: int = 2,
    min_runs_without_feature: int = 5,
    min_vio_steps: int = 50,
    min_max_vio: float = 4.0,
) -> pd.DataFrame:
    """Compare monitor→max_vio correlations for runs with vs without each feature."""
    inv = pd.read_csv(out_dir / "run_inventory.csv")
    summary = pd.read_csv(out_dir / "run_summary.csv")
    corr = pd.read_csv(out_dir / "metric_correlations.csv")
    corr = corr.merge(
        summary[["run", "n_vio_steps", "max_vio_max"]], on="run", how="left"
    )
    corr = corr[
        (corr["n_vio_steps"] >= min_vio_steps) & (corr["max_vio_max"] >= min_max_vio)
    ].copy()
    corr["group"] = corr["tag"].map(monitor_group)
    corr["abs_corr"] = corr["corr_with_max_vio"].abs()

    feat_cols = [name for name, _ in FEATURE_PATTERNS if name in inv.columns]
    run_features = inv[["run", *feat_cols]].copy()
    corr = corr.merge(run_features, on="run", how="left")

    rows: list[dict[str, Any]] = []
    for feat in feat_cols:
        feat_run_counts = run_features.set_index("run")[feat].fillna(False).astype(bool)
        runs_with = set(feat_run_counts[feat_run_counts].index)
        runs_without = set(feat_run_counts[~feat_run_counts].index)
        for group, g in corr.groupby("group"):
            with_g = g[g["run"].isin(runs_with)]
            without_g = g[g["run"].isin(runs_without)]
            n_with_runs = with_g["run"].nunique()
            n_without_runs = without_g["run"].nunique()
            if (
                n_with_runs < min_runs_with_feature
                or n_without_runs < min_runs_without_feature
            ):
                continue
            med_with = float(with_g["abs_corr"].median())
            med_without = float(without_g["abs_corr"].median())
            rows.append(
                {
                    "feature": feat,
                    "group": group,
                    "n_runs_with": int(n_with_runs),
                    "n_runs_without": int(n_without_runs),
                    "n_tags_with": int(len(with_g)),
                    "n_tags_without": int(len(without_g)),
                    "median_abs_corr_with": med_with,
                    "median_abs_corr_without": med_without,
                    "delta_median_abs_corr": med_with - med_without,
                    "mean_abs_corr_with": float(with_g["abs_corr"].mean()),
                    "mean_abs_corr_without": float(without_g["abs_corr"].mean()),
                    "max_abs_corr_with": float(with_g["abs_corr"].max()),
                    "max_abs_corr_without": float(without_g["abs_corr"].max()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["feature", "delta_median_abs_corr"], ascending=[True, False]
        )
    out.to_csv(out_dir / "feature_monitor_correlation_summary.csv", index=False)
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_cli_args(ap)
    ap.add_argument(
        "--feature-corr",
        action="store_true",
        help="Не пересчитывать summary — только feature_monitor_correlation_summary из готовых CSV",
    )
    args = ap.parse_args()
    cfg: ResolvedConfig = resolve(args)
    out = cfg.output_dir

    if args.feature_corr:
        feature_monitor_correlation_summary(out)
        return

    # Выборка ранов: если CLI задал --runs/--runs-pattern — используем их (sample-mode).
    # Иначе — все раны с n_event_files >= min_event_files (полный inventory).
    if cfg.runs_override is not None:
        runs = cfg.runs_override
    elif cfg.runs_pattern is not None:
        import fnmatch

        runs = [
            p.name
            for p in sorted(cfg.tb_root.iterdir())
            if p.is_dir() and fnmatch.fnmatch(p.name, cfg.runs_pattern)
        ]
    else:
        runs = None

    inv, summary, _ = build_summary(
        cfg.tb_root,
        out,
        runs=runs,
        min_event_files=cfg.eda.min_event_files,
        max_runs=cfg.eda.max_runs,
        max_workers=cfg.eda.max_workers,
    )
    feature_table(inv, summary, out)
    print("\nWrote into", out)


if __name__ == "__main__":
    _cli()
