"""Temporal causality + event-study around growth onset.

Stage 2. Reads `run_summary.csv` + `run_inventory.csv` from `eda.py` to
sample runs (preferred + filters from ``TEMPORAL_DEFAULTS`` in
``instability/config.py``), then per-run derives signals and asks the
same question three different ways:

* **Forward-lag predictive F-test** on *absolute levels* (regress the future
  level of one series on its own AR history plus the lagged levels of the
  other; the F-test on the added block is Granger-style but keeps the trend
  the reviewer cares about instead of differencing it away — see
  ``forward_predictive_test``). Binary verdict via p-values + BH-FDR, plus a
  ΔR² and the sign of the cumulative effect.
  Writes ``granger_results.csv`` + ``granger_group_summary.csv``.
* **Cross-correlation function** at lags in ``[-50, +50]`` (model-free,
  cheap; the sign of the peak directly tells which series leads).
  Writes ``ccf_results.csv`` + ``ccf_top_per_group.csv``.
* **Lasso AR-X regression** in both directions (interpretable
  coefficients per lag, asymmetry ratio tells direction strength).
  Writes ``lasso_arx_results.csv`` + ``lasso_top_per_group.csv``.

Plus event-study (pre-onset behaviour) → ``event_pre_onset_results.csv``.

Three independent answers to "что влияет на max_vio vs на что max_vio
влияет": if all three agree, the finding is solid; if they disagree,
there's something to dig into (likely a nonlinear effect or shared
confound).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from instability.config import (
    ResolvedConfig,
    add_common_cli_args,
    resolve,
    select_runs,
)
from instability.eda import interp
from instability.features import growth_onset, monitor_group
from instability.tb_helpers import (
    build_max_vio_series,
    load_run_scalars,
    parse_layer_from_tag,
)


EXCLUDE_SUBSTRINGS = (
    "old_moe_routing",
    "dispatcher_layer_experts_max_vio",
    "dispatcher_sum_metrics/experts_max_vio",
)


def candidate_tags(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[str]:
    tags = []
    for tag, (st, va) in series.items():
        if any(s in tag for s in EXCLUDE_SUBSTRINGS):
            continue
        group = monitor_group(tag)
        if group == "other":
            continue
        if len(st) < 30 or np.nanstd(va) < 1e-12:
            continue
        tags.append(tag)
    return tags


def add_derived_signals(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Derived monitors from already aligned raw monitors."""
    out = dict(signals)
    grad_re = re.compile(
        r"l2_norm/grad/(.*model\.layers\.(\d+)\.block_sparse_moe\.flatten)"
    )
    layer_series: dict[str, dict[int, np.ndarray]] = {
        "effective_lr": {},
        "update_norm": {},
        "grad_norm": {},
        "update_param_ratio": {},
        "grad_param_ratio": {},
    }
    for tag, grad in list(signals.items()):
        m = grad_re.match(tag)
        if not m:
            continue
        suffix = m.group(1)
        layer = m.group(2)
        layer_i = int(layer)
        upd_tag = f"l2_norm/update/{suffix}"
        param_tag = f"l2_norm/param/{suffix}"
        if upd_tag not in signals:
            continue
        upd = signals[upd_tag]
        eff = np.where(np.abs(grad) > 1e-15, upd / grad, np.nan)
        gap = np.log1p(np.maximum(upd, 0)) - np.log1p(np.maximum(grad, 0))
        out[f"derived/effective_lr/layer_{layer}"] = eff
        out[f"derived/update_minus_grad_log/layer_{layer}"] = gap
        layer_series["effective_lr"][layer_i] = eff
        layer_series["update_norm"][layer_i] = upd
        layer_series["grad_norm"][layer_i] = grad
        if param_tag in signals:
            param = signals[param_tag]
            upr = np.where(np.abs(param) > 1e-15, upd / param, np.nan)
            gpr = np.where(np.abs(param) > 1e-15, grad / param, np.nan)
            out[f"derived/update_param_ratio/layer_{layer}"] = upr
            out[f"derived/grad_param_ratio/layer_{layer}"] = gpr
            layer_series["update_param_ratio"][layer_i] = upr
            layer_series["grad_param_ratio"][layer_i] = gpr

        for act_name in [
            "activations_amax/model.model.layers.{layer}.block_sparse_moe.gmm_upgate_input.0",
            "activations_amax/model.model.layers.{layer}.block_sparse_moe.fused_swiglu_input.0",
            "activations_amean/model.model.layers.{layer}.block_sparse_moe.block_sparse_moe_output.0",
        ]:
            act_tag = act_name.format(layer=layer)
            if act_tag not in signals:
                continue
            act = signals[act_tag]
            safe = np.abs(act) > 1e-15
            short = (
                act_tag.split("block_sparse_moe.", 1)[-1]
                .replace("/", "_")
                .replace(".", "_")
            )
            out[f"derived/update_activation_ratio/{short}/layer_{layer}"] = np.where(
                safe, upd / act, np.nan
            )
            out[f"derived/grad_activation_ratio/{short}/layer_{layer}"] = np.where(
                safe, grad / act, np.nan
            )

    for name, per_layer in layer_series.items():
        if len(per_layer) < 4:
            continue
        layers = sorted(per_layer)
        mat = np.vstack([per_layer[L] for L in layers])
        med = np.nanmedian(mat, axis=0)
        mx = np.nanmax(mat, axis=0)
        out[f"derived/layer_contrast/{name}/max_over_median"] = np.where(
            np.abs(med) > 1e-15, mx / med, np.nan
        )
        early = [L for L in layers if L <= np.percentile(layers, 25)]
        late = [L for L in layers if L >= np.percentile(layers, 75)]
        if early and late:
            early_mean = np.nanmean(np.vstack([per_layer[L] for L in early]), axis=0)
            late_mean = np.nanmean(np.vstack([per_layer[L] for L in late]), axis=0)
            out[f"derived/layer_contrast/{name}/late_over_early"] = np.where(
                np.abs(early_mean) > 1e-15, late_mean / early_mean, np.nan
            )
        for L in [3, 4, 22, 23]:
            if L in per_layer:
                out[f"derived/layer_contrast/{name}/layer_{L}_over_median"] = np.where(
                    np.abs(med) > 1e-15, per_layer[L] / med, np.nan
                )
    return out


def saturation_step(
    steps: np.ndarray, y: np.ndarray, saturation: float = 0.8 * 47.0
) -> float:
    idx = np.where(y >= saturation)[0]
    return float(steps[int(idx[0])]) if idx.size else math.nan


def transform_series(x: np.ndarray, win: int = 11) -> dict[str, np.ndarray]:
    """Vectorized rolling stats on a 1-D series.

    Returns level/diff/log + rolling z-score, log-volatility, and a
    centered-window log-slope (window of length ``win``).
    """
    x = x.astype(float)
    n = len(x)
    eps = 1e-12
    early = x[: max(10, n // 10)]
    early_med = np.nanmedian(early)
    early_q95 = np.nanpercentile(early, 95) if np.isfinite(early).sum() else math.nan
    logx = np.log1p(np.maximum(x, 0))
    diff_log = np.r_[np.nan, np.diff(logx)]

    slope = np.full(n, np.nan)
    z = np.full(n, np.nan)
    vol = np.full(n, np.nan)
    if n >= 3:
        from numpy.lib.stride_tricks import sliding_window_view

        # Centered window of length up to `win`. We compute on the
        # max-length windows we can build with sliding_window_view and
        # assign them to positions [half, half + n_windows). Edge
        # positions keep NaN — matches old behaviour at the boundaries.
        w = min(win, n)
        half = w // 2
        windows_x = sliding_window_view(x, w)  # (n-w+1, w)
        windows_lx = sliding_window_view(logx, w)  # (n-w+1, w)
        # z-score: (x[i] - mean(window_i)) / std(window_i)
        mean_x = np.nanmean(windows_x, axis=1)
        std_x = np.nanstd(windows_x, axis=1)
        center_idx = np.arange(windows_x.shape[0]) + half
        # Guard against out-of-range when n < win (clipped above).
        center_idx = center_idx[center_idx < n]
        z[center_idx] = (x[center_idx] - mean_x[: center_idx.size]) / (
            std_x[: center_idx.size] + eps
        )
        vol[center_idx] = np.nanstd(windows_lx, axis=1)[: center_idx.size]
        # Rolling log-slope: fit logx vs [0..w-1] in each window.
        # slope = cov(t, lx) / var(t) with t = 0..w-1.
        t = np.arange(w, dtype=float)
        t_mean = t.mean()
        t_dev = t - t_mean
        t_var = float((t_dev**2).sum())
        if t_var > 0:
            lx_mean = np.nanmean(windows_lx, axis=1, keepdims=True)
            # nansum so windows with a few NaNs still yield a value.
            num = np.nansum((windows_lx - lx_mean) * t_dev, axis=1)
            slope[center_idx] = (num / t_var)[: center_idx.size]
    return {
        "level_log": logx,
        "level_rel_early": x / (early_med + eps),
        "level_high_early_q95": (x > early_q95).astype(float)
        if np.isfinite(early_q95)
        else np.full(n, np.nan),
        "diff_log": diff_log,
        "rolling_slope_log": slope,
        "rolling_z": z,
        "rolling_vol_log": vol,
    }


def stationarity_diagnostics(v: np.ndarray) -> dict[str, float]:
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return {
            "n": len(v),
            "trend_r2": math.nan,
            "acf1": math.nan,
            "half_mean_ratio": math.nan,
        }
    t = np.arange(len(v), dtype=float)
    _slope, _intercept, r, _, _ = stats.linregress(t, v)
    acf1 = (
        np.corrcoef(v[:-1], v[1:])[0, 1]
        if np.nanstd(v[:-1]) > 0 and np.nanstd(v[1:]) > 0
        else math.nan
    )
    h = len(v) // 2
    denom = abs(np.nanmean(v[:h])) + 1e-12
    return {
        "n": len(v),
        "trend_r2": float(r * r),
        "acf1": float(acf1),
        "half_mean_ratio": float(np.nanmean(v[h:]) / denom),
    }


def forward_predictive_test(y: np.ndarray, x: np.ndarray, lag: int) -> dict[str, float]:
    """Does the *absolute level* of x with a forward lag help predict the level of y?

    This replaces the differenced Granger test. The reviewer's objection to Granger was
    that it requires stationary inputs, so the trend is differenced away — but for
    instability the absolute level of max_vio (and of the predictors) is exactly what
    matters. Here we keep raw levels: we regress y[t] on its own past (AR baseline) plus
    the lagged levels of x, and test whether the x block adds predictive power. We report
    the F-test p-value, the gain in R^2 from adding x, and the sign of the cumulative x
    effect (does a higher x predict a higher or lower future y).
    """
    nan_out = {
        "p_value": math.nan,
        "f_stat": math.nan,
        "aic_unrestricted": math.nan,
        "bic_unrestricted": math.nan,
        "delta_r2": math.nan,
        "beta_x_sum": math.nan,
        "n_obs": 0,
    }
    m = np.isfinite(y) & np.isfinite(x)
    y = y[m].astype(float)
    x = x[m].astype(float)
    n = len(y)
    if n <= 3 * lag + 5 or lag < 1:
        return {**nan_out, "n_obs": n}
    target = y[lag:]
    cols_r = [np.ones(n - lag)]  # const + own (AR) lags
    cols_u = [np.ones(n - lag)]  # + lagged levels of x
    for k in range(1, lag + 1):
        yy = y[lag - k : n - k]
        cols_r.append(yy)
        cols_u.append(yy)
    for k in range(1, lag + 1):
        cols_u.append(x[lag - k : n - k])
    Xr = np.column_stack(cols_r)
    Xu = np.column_stack(cols_u)
    br, *_ = np.linalg.lstsq(Xr, target, rcond=None)
    bu, *_ = np.linalg.lstsq(Xu, target, rcond=None)
    er = target - Xr @ br
    eu = target - Xu @ bu
    ssr_r = float(np.sum(er * er))
    ssr_u = float(np.sum(eu * eu))
    sst = float(np.sum((target - np.mean(target)) ** 2))
    df_num = lag
    df_den = len(target) - Xu.shape[1]
    if df_den <= 0 or ssr_u <= 0 or sst <= 0:
        return {**nan_out, "n_obs": n}
    f = ((ssr_r - ssr_u) / df_num) / (ssr_u / df_den)
    p = float(stats.f.sf(max(f, 0.0), df_num, df_den))
    delta_r2 = (ssr_r - ssr_u) / sst  # R^2_unrestricted − R^2_restricted
    beta_x_sum = float(np.sum(bu[1 + lag :]))  # cumulative effect of the x lags
    sigma2 = ssr_u / len(target)
    aic = len(target) * math.log(max(sigma2, 1e-30)) + 2 * Xu.shape[1]
    bic = (
        len(target) * math.log(max(sigma2, 1e-30)) + math.log(len(target)) * Xu.shape[1]
    )
    return {
        "p_value": p,
        "f_stat": float(f),
        "aic_unrestricted": float(aic),
        "bic_unrestricted": float(bic),
        "delta_r2": float(delta_r2),
        "beta_x_sum": beta_x_sum,
        "n_obs": n,
    }


def choose_lag_and_predict(
    y: np.ndarray, x: np.ndarray, max_lag: int = 8
) -> dict[str, float]:
    """Level-based forward-lag predictive test with AIC/BIC lag selection."""
    n = np.isfinite(y + x).sum()
    max_lag = int(min(max_lag, max(1, n // 10)))
    rows = []
    for lag in range(1, max_lag + 1):
        res = forward_predictive_test(y, x, lag)
        res["lag"] = lag
        if np.isfinite(res["aic_unrestricted"]):
            rows.append(res)
    if not rows:
        return {
            "best_lag_aic": math.nan,
            "best_lag_bic": math.nan,
            "p_aic": math.nan,
            "p_bic": math.nan,
            "f_aic": math.nan,
            "f_bic": math.nan,
            "delta_r2_bic": math.nan,
            "beta_x_sum_bic": math.nan,
            "n_obs": n,
        }
    aic_best = min(rows, key=lambda r: r["aic_unrestricted"])
    bic_best = min(rows, key=lambda r: r["bic_unrestricted"])
    return {
        "best_lag_aic": aic_best["lag"],
        "best_lag_bic": bic_best["lag"],
        "p_aic": aic_best["p_value"],
        "p_bic": bic_best["p_value"],
        "f_aic": aic_best["f_stat"],
        "f_bic": bic_best["f_stat"],
        "delta_r2_bic": bic_best["delta_r2"],
        "beta_x_sum_bic": bic_best["beta_x_sum"],
        "n_obs": aic_best["n_obs"],
    }


def cross_correlation_analysis(
    y: np.ndarray,
    x: np.ndarray,
    *,
    max_lag: int = 50,
) -> dict[str, float]:
    """Pearson cross-correlation of ``x_{t-k}`` with ``y_t`` for k ∈ [-max_lag, +max_lag].

    Complements Granger: cheap, model-free, gives a direct "leads/lags"
    answer via the position of the peak. Sign of ``best_lag``:

    * **best_lag > 0** → ``x`` leads ``y`` (lagged ``x`` predicts current ``y``)
    * **best_lag < 0** → ``y`` leads ``x``
    * **best_lag ≈ 0** → synchronous (could be shared driver / confound)

    Returned dict has ``best_lag``, ``peak_corr`` (signed), ``peak_abs_corr``,
    ``direction`` (string), ``corr_at_0`` (synchronous baseline), ``n_obs``.
    """
    m = np.isfinite(y) & np.isfinite(x)
    y = y[m].astype(float)
    x = x[m].astype(float)
    n = len(y)
    if n < 2 * max_lag + 10:
        return {
            "best_lag": math.nan,
            "peak_corr": math.nan,
            "peak_abs_corr": math.nan,
            "direction": "insufficient_data",
            "corr_at_0": math.nan,
            "n_obs": n,
        }
    # Center; keep std-normalisation per-window to avoid drift bias.
    lags = list(range(-max_lag, max_lag + 1))
    corrs = np.full(len(lags), np.nan, dtype=float)
    for i, k in enumerate(lags):
        if k > 0:
            a, b = x[: n - k], y[k:]
        elif k < 0:
            a, b = x[-k:], y[: n + k]
        else:
            a, b = x, y
        if len(a) < 10:
            continue
        am = a - a.mean()
        bm = b - b.mean()
        sa = am.std(ddof=1)
        sb = bm.std(ddof=1)
        if sa < 1e-30 or sb < 1e-30:
            continue
        corrs[i] = float((am * bm).mean() / (sa * sb))
    if np.all(np.isnan(corrs)):
        return {
            "best_lag": math.nan,
            "peak_corr": math.nan,
            "peak_abs_corr": math.nan,
            "direction": "no_finite_corr",
            "corr_at_0": math.nan,
            "n_obs": n,
        }
    abs_c = np.abs(corrs)
    bi = int(np.nanargmax(abs_c))
    bl = int(lags[bi])
    if bl > 0:
        direction = "x_leads_y"
    elif bl < 0:
        direction = "y_leads_x"
    else:
        direction = "synchronous"
    corr_at_0 = float(corrs[max_lag]) if np.isfinite(corrs[max_lag]) else math.nan
    return {
        "best_lag": bl,
        "peak_corr": float(corrs[bi]),
        "peak_abs_corr": float(abs_c[bi]),
        "direction": direction,
        "corr_at_0": corr_at_0,
        "n_obs": int(n),
    }


def lasso_arx_analysis(
    y: np.ndarray,
    x: np.ndarray,
    *,
    ar_order: int = 5,
    x_lags: int = 10,
    cv_folds: int = 3,
) -> dict[str, float]:
    """Lasso-regularised AR-X regression in both directions.

    Forward model:  y_t = Σ a_i y_{t-i} + Σ β_l x_{t-l} + ε     (i=1..ar_order, l=1..x_lags)
    Backward model: x_t = Σ c_i x_{t-i} + Σ δ_l y_{t-l} + ε'

    ``β_l`` measures how much past ``x`` drives future ``y`` AFTER controlling
    for ``y``'s own past — same intuition as Granger but with explicit
    coefficients. ``δ_l`` is the reverse. Asymmetry tells direction:

    * ``asymmetry > 0`` → ``x`` drives ``y`` more than the reverse
    * ``asymmetry < 0`` → the reverse
    * close to 0 → no clear directionality

    Sklearn LassoCV picks the regularisation strength via CV. Sparsity in
    the exogenous block does feature selection across lags for free.
    """
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {
            "max_abs_beta_x_to_y": math.nan,
            "best_x_lag": math.nan,
            "sum_abs_beta_x_to_y": math.nan,
            "n_nonzero_x": math.nan,
            "r2_x_to_y": math.nan,
            "max_abs_delta_y_to_x": math.nan,
            "best_y_lag": math.nan,
            "sum_abs_delta_y_to_x": math.nan,
            "n_nonzero_y": math.nan,
            "r2_y_to_x": math.nan,
            "asymmetry_max": math.nan,
            "asymmetry_sum": math.nan,
            "n_obs": math.nan,
            "error": "sklearn_not_installed",
        }

    n = min(len(y), len(x))
    max_l = max(ar_order, x_lags)
    if n < max_l + 50:
        return {
            "max_abs_beta_x_to_y": math.nan,
            "best_x_lag": math.nan,
            "sum_abs_beta_x_to_y": math.nan,
            "n_nonzero_x": math.nan,
            "r2_x_to_y": math.nan,
            "max_abs_delta_y_to_x": math.nan,
            "best_y_lag": math.nan,
            "sum_abs_delta_y_to_x": math.nan,
            "n_nonzero_y": math.nan,
            "r2_y_to_x": math.nan,
            "asymmetry_max": math.nan,
            "asymmetry_sum": math.nan,
            "n_obs": n,
            "error": "insufficient_data",
        }

    y = y[:n].astype(float)
    x = x[:n].astype(float)

    def _fit_one(target: np.ndarray, exog: np.ndarray) -> dict[str, float] | None:
        N = n - max_l
        cols = []
        for i in range(1, ar_order + 1):
            cols.append(target[max_l - i : max_l - i + N])
        for k in range(1, x_lags + 1):
            cols.append(exog[max_l - k : max_l - k + N])
        Xd = np.column_stack(cols)
        yd = target[max_l : max_l + N]
        mask = np.isfinite(yd) & np.all(np.isfinite(Xd), axis=1)
        if mask.sum() < max_l + 20:
            return None
        Xd = Xd[mask]
        yd = yd[mask]
        if Xd.std(axis=0).min() < 1e-30:
            return None
        try:
            Xs = StandardScaler().fit_transform(Xd)
            model = LassoCV(cv=cv_folds, max_iter=5000, n_alphas=15).fit(Xs, yd)
        except Exception:  # noqa: BLE001
            return None
        coefs = model.coef_
        beta_ex = coefs[ar_order:]
        abs_ex = np.abs(beta_ex)
        score = float(model.score(Xs, yd))
        if abs_ex.size == 0 or abs_ex.max() == 0:
            best_lag = 0
        else:
            best_lag = int(np.argmax(abs_ex) + 1)
        return {
            "max_abs": float(abs_ex.max()) if abs_ex.size else math.nan,
            "best_lag": best_lag,
            "sum_abs": float(abs_ex.sum()),
            "n_nonzero": int((abs_ex > 1e-10).sum()),
            "r2": score,
            "n_obs": int(len(yd)),
        }

    fwd = _fit_one(y, x)
    bwd = _fit_one(x, y)
    if fwd is None or bwd is None:
        return {
            "max_abs_beta_x_to_y": math.nan,
            "best_x_lag": math.nan,
            "sum_abs_beta_x_to_y": math.nan,
            "n_nonzero_x": math.nan,
            "r2_x_to_y": math.nan,
            "max_abs_delta_y_to_x": math.nan,
            "best_y_lag": math.nan,
            "sum_abs_delta_y_to_x": math.nan,
            "n_nonzero_y": math.nan,
            "r2_y_to_x": math.nan,
            "asymmetry_max": math.nan,
            "asymmetry_sum": math.nan,
            "n_obs": n,
            "error": "fit_failed",
        }

    eps = 1e-30
    asym_max = (fwd["max_abs"] - bwd["max_abs"]) / (
        fwd["max_abs"] + bwd["max_abs"] + eps
    )
    asym_sum = (fwd["sum_abs"] - bwd["sum_abs"]) / (
        fwd["sum_abs"] + bwd["sum_abs"] + eps
    )
    return {
        "max_abs_beta_x_to_y": fwd["max_abs"],
        "best_x_lag": fwd["best_lag"],
        "sum_abs_beta_x_to_y": fwd["sum_abs"],
        "n_nonzero_x": fwd["n_nonzero"],
        "r2_x_to_y": fwd["r2"],
        "max_abs_delta_y_to_x": bwd["max_abs"],
        "best_y_lag": bwd["best_lag"],
        "sum_abs_delta_y_to_x": bwd["sum_abs"],
        "n_nonzero_y": bwd["n_nonzero"],
        "r2_y_to_x": bwd["r2"],
        "asymmetry_max": float(asym_max),
        "asymmetry_sum": float(asym_sum),
        "n_obs": int(fwd["n_obs"]),
    }


def bh_fdr(p: pd.Series) -> pd.Series:
    pvals = p.astype(float).to_numpy()
    out = np.full(len(pvals), np.nan)
    m = np.isfinite(pvals)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return pd.Series(out, index=p.index)
    order = idx[np.argsort(pvals[idx])]
    ranks = np.arange(1, len(order) + 1)
    q = pvals[order] * len(order) / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.clip(q, 0, 1)
    return pd.Series(out, index=p.index)


def load_aligned_run(
    root: Path, run: str, max_workers: int = 4
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    s = load_run_scalars(str(root / run), max_workers=max_workers)
    steps, y, _ = build_max_vio_series(s)
    signals: dict[str, np.ndarray] = {}
    if len(steps) == 0:
        return steps, y, signals
    for tag in candidate_tags(s):
        signals[tag] = interp(*s[tag], steps)
    signals = add_derived_signals(signals)
    return steps, y, signals


def analyze_run(
    root: Path,
    run: str,
    max_tags_per_group: int = 8,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    from time import perf_counter

    t0 = perf_counter()
    print("[temporal]   load + align: starting...", flush=True)
    steps, y_raw, signals = load_aligned_run(root, run)
    print(
        f"[temporal]   load + align: {perf_counter() - t0:.1f}s "
        f"({len(signals)} candidate signals, {len(steps)} steps)",
        flush=True,
    )
    if len(steps) < 50 or not signals:
        return [], [], []

    sat = saturation_step(steps, y_raw)
    onset = growth_onset(steps, y_raw)
    cutoff_step = sat if np.isfinite(sat) else steps[-1]
    keep = steps <= cutoff_step
    if keep.sum() < 50:
        keep = np.arange(len(steps)) < min(
            len(steps),
            max(
                50,
                np.where(steps <= cutoff_step)[0][-1] + 1 if keep.any() else len(steps),
            ),
        )

    steps_w = steps[keep]
    y = y_raw[keep]
    y_tr = transform_series(y)
    # Predictive test runs on the *absolute* level of max_vio (raw y), not a differenced
    # series — see forward_predictive_test for why. CCF + Lasso still use the differenced
    # series (diff_log), which is appropriate for those model-free / regularized views.
    y_level = y
    y_dyn = y_tr["diff_log"]

    tag_df = []
    for tag, vals in signals.items():
        v = vals[keep]
        tag_df.append(
            {
                "tag": tag,
                "group": monitor_group(tag),
                "std": float(np.nanstd(v)),
                "layer": parse_layer_from_tag(tag),
            }
        )
    tags = pd.DataFrame(tag_df)
    chosen = []
    for _group, g in tags.groupby("group"):
        chosen.extend(
            g.sort_values("std", ascending=False)
            .head(max_tags_per_group)["tag"]
            .tolist()
        )

    granger_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    ccf_rows: list[dict[str, Any]] = []
    lasso_rows: list[dict[str, Any]] = []

    print(
        f"[temporal]   Granger + CCF + Lasso ARX on {len(chosen)} chosen tag(s) "
        f"(max 8 per monitor group, granger max_lag=8, ccf max_lag=50, "
        f"lasso ar=5/x_lags=10)...",
        flush=True,
    )
    g_t0 = perf_counter()
    for tag in chosen:
        x = signals[tag][keep]
        x_tr = transform_series(x)
        group = monitor_group(tag)
        layer = parse_layer_from_tag(tag)

        if np.isfinite(onset):
            pre = (steps_w >= onset - 500) & (steps_w < onset)
            base = steps_w < max(steps_w[0] + 500, onset - 1000)
        else:
            pre = np.arange(len(steps_w)) >= max(0, len(steps_w) - len(steps_w) // 5)
            base = np.arange(len(steps_w)) < max(5, len(steps_w) // 5)
        for rep in [
            "level_log",
            "level_rel_early",
            "rolling_slope_log",
            "rolling_z",
            "rolling_vol_log",
        ]:
            xv = x_tr[rep]
            b = xv[base]
            p = xv[pre]
            bmean = float(np.nanmean(b)) if np.isfinite(b).sum() else math.nan
            pmean = float(np.nanmean(p)) if np.isfinite(p).sum() else math.nan
            bstd = float(np.nanstd(b)) if np.isfinite(b).sum() else math.nan
            event_rows.append(
                {
                    "run": run,
                    "tag": tag,
                    "group": group,
                    "layer": layer,
                    "onset_step": onset,
                    "saturation_step": sat,
                    "representation": rep,
                    "baseline_mean": bmean,
                    "pre_onset_mean": pmean,
                    "pre_minus_baseline": pmean - bmean
                    if np.isfinite(pmean) and np.isfinite(bmean)
                    else math.nan,
                    "pre_z_vs_baseline": (pmean - bmean) / (bstd + 1e-12)
                    if np.isfinite(pmean) and np.isfinite(bmean) and np.isfinite(bstd)
                    else math.nan,
                }
            )

        for rep in ["level_log", "diff_log", "rolling_slope_log", "rolling_z"]:
            d = stationarity_diagnostics(x_tr[rep])
            d.update(
                {
                    "run": run,
                    "tag": tag,
                    "group": group,
                    "layer": layer,
                    "representation": rep,
                }
            )
            stat_rows.append(d)

        # Forward-lag prediction on absolute levels (raw x → future max_vio, and back).
        x_level = x
        x_dyn = x_tr["diff_log"]  # differenced series for the CCF + Lasso views below
        xy = choose_lag_and_predict(y_level, x_level, max_lag=8)  # does x lead max_vio
        yx = choose_lag_and_predict(x_level, y_level, max_lag=8)  # does max_vio lead x
        granger_rows.append(
            {
                "run": run,
                "tag": tag,
                "group": group,
                "layer": layer,
                "x_to_y_p_bic": xy["p_bic"],
                "x_to_y_p_aic": xy["p_aic"],
                "x_to_y_lag_bic": xy["best_lag_bic"],
                "x_to_y_lag_aic": xy["best_lag_aic"],
                "x_to_y_f_bic": xy["f_bic"],
                "x_to_y_delta_r2_bic": xy["delta_r2_bic"],
                "x_to_y_beta_bic": xy["beta_x_sum_bic"],
                "y_to_x_p_bic": yx["p_bic"],
                "y_to_x_p_aic": yx["p_aic"],
                "y_to_x_lag_bic": yx["best_lag_bic"],
                "y_to_x_lag_aic": yx["best_lag_aic"],
                "y_to_x_f_bic": yx["f_bic"],
                "y_to_x_delta_r2_bic": yx["delta_r2_bic"],
                "y_to_x_beta_bic": yx["beta_x_sum_bic"],
                "n_obs": xy["n_obs"],
                "onset_step": onset,
                "saturation_step": sat,
            }
        )

        # Cross-correlation: cheap, model-free, sign of best_lag → direction.
        ccf = cross_correlation_analysis(y_dyn, x_dyn, max_lag=50)
        ccf_rows.append(
            {
                "run": run,
                "tag": tag,
                "group": group,
                "layer": layer,
                **ccf,
            }
        )

        # Lasso AR-X: explicit coefficients in both directions, asymmetry
        # tells which side dominates.
        lasso = lasso_arx_analysis(y_dyn, x_dyn, ar_order=5, x_lags=10)
        lasso_rows.append(
            {
                "run": run,
                "tag": tag,
                "group": group,
                "layer": layer,
                **lasso,
            }
        )

    print(
        f"[temporal]   Granger+CCF+Lasso done in {perf_counter() - g_t0:.1f}s "
        f"({len(granger_rows)} rows). Total analyze_run: "
        f"{perf_counter() - t0:.1f}s.",
        flush=True,
    )
    return granger_rows, event_rows, stat_rows, ccf_rows, lasso_rows


def run_temporal_analysis(
    root: Path,
    out_dir: Path,
    runs: list[str],
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    all_g: list[dict[str, Any]] = []
    all_e: list[dict[str, Any]] = []
    all_s: list[dict[str, Any]] = []
    all_c: list[dict[str, Any]] = []
    all_l: list[dict[str, Any]] = []
    for i, run in enumerate(runs, start=1):
        print(f"[{i}/{len(runs)}] {run}", flush=True)
        g, e, s, c, la = analyze_run(root, run)
        all_g.extend(g)
        all_e.extend(e)
        all_s.extend(s)
        all_c.extend(c)
        all_l.extend(la)

    granger = pd.DataFrame(all_g)
    event = pd.DataFrame(all_e)
    stat = pd.DataFrame(all_s)
    ccf = pd.DataFrame(all_c)
    lasso = pd.DataFrame(all_l)
    if not granger.empty:
        granger["x_to_y_q_bic"] = granger.groupby(["run", "group"], group_keys=False)[
            "x_to_y_p_bic"
        ].apply(bh_fdr)
        granger["y_to_x_q_bic"] = granger.groupby(["run", "group"], group_keys=False)[
            "y_to_x_p_bic"
        ].apply(bh_fdr)
        granger["direction"] = np.select(
            [
                (granger["x_to_y_q_bic"] < 0.1) & ~(granger["y_to_x_q_bic"] < 0.1),
                ~(granger["x_to_y_q_bic"] < 0.1) & (granger["y_to_x_q_bic"] < 0.1),
                (granger["x_to_y_q_bic"] < 0.1) & (granger["y_to_x_q_bic"] < 0.1),
            ],
            ["x_leads_y", "y_leads_x", "bidirectional"],
            default="none",
        )

    if not granger.empty:
        group_summary = (
            granger.groupby("group")
            .agg(
                n_tests=("tag", "count"),
                n_runs=("run", "nunique"),
                frac_x_leads_y=(
                    "direction",
                    lambda s: float((s == "x_leads_y").mean()),
                ),
                frac_bidirectional=(
                    "direction",
                    lambda s: float((s == "bidirectional").mean()),
                ),
                median_x_to_y_p=("x_to_y_p_bic", "median"),
                median_x_to_y_lag=("x_to_y_lag_bic", "median"),
                median_x_to_y_delta_r2=("x_to_y_delta_r2_bic", "median"),
            )
            .reset_index()
        )
    else:
        group_summary = pd.DataFrame()

    granger.to_csv(out_dir / "granger_results.csv", index=False)
    event.to_csv(out_dir / "event_pre_onset_results.csv", index=False)
    stat.to_csv(out_dir / "stationarity_diagnostics.csv", index=False)
    group_summary.to_csv(out_dir / "granger_group_summary.csv", index=False)
    ccf.to_csv(out_dir / "ccf_results.csv", index=False)
    lasso.to_csv(out_dir / "lasso_arx_results.csv", index=False)

    # Convenience top-driver summaries — what most strongly leads max_vio
    # vs. what max_vio most strongly leads. One row per (run, tag).
    if not ccf.empty:
        ccf_summary = (
            ccf.assign(abs_corr=lambda d: d["peak_abs_corr"])
            .sort_values("abs_corr", ascending=False)
            .groupby("group")
            .head(10)
            .reset_index(drop=True)
        )
        ccf_summary.to_csv(out_dir / "ccf_top_per_group.csv", index=False)
    if not lasso.empty:
        lasso_summary = (
            lasso.assign(
                drives_max_vio_strength=lambda d: d["max_abs_beta_x_to_y"],
                driven_by_max_vio_strength=lambda d: d["max_abs_delta_y_to_x"],
            )
            .sort_values("asymmetry_max", ascending=False)
            .groupby("group")
            .head(10)
            .reset_index(drop=True)
        )
        lasso_summary.to_csv(out_dir / "lasso_top_per_group.csv", index=False)

    inv_path = out_dir / "run_inventory.csv"
    if inv_path.exists() and not granger.empty and not event.empty:
        feature_conditioned_summary(out_dir, pd.read_csv(inv_path), granger, event)
    return granger, event, stat, group_summary, ccf, lasso


def feature_conditioned_summary(
    out_dir: Path,
    inv: pd.DataFrame,
    granger: pd.DataFrame,
    event: pd.DataFrame,
) -> pd.DataFrame:
    feat_cols = [
        c
        for c in [
            "postnorm_wrap",
            "aux_loss_free_adaptive",
            "router_bias_update",
            "router_z_loss",
            "sandwichnorm",
            "zero_centered_rms_norm",
            "zero_centered_gated_norm",
            "activation_clipping",
            "no_seq_len_warmup",
            "mla_scaling_factor",
            "shared_expert_sigmoid",
        ]
        if c in inv.columns
    ]
    flags = inv[["run", *feat_cols]].copy()
    granger = granger.merge(flags, on="run", how="left")
    event = event.merge(flags, on="run", how="left")

    rows: list[dict[str, Any]] = []
    for feat in feat_cols:
        for group, gg in granger.groupby("group"):
            with_g = gg[gg[feat].fillna(False).astype(bool)]
            without_g = gg[~gg[feat].fillna(False).astype(bool)]
            if with_g["run"].nunique() < 2 or without_g["run"].nunique() < 3:
                continue
            ev = event[
                (event["group"] == group)
                & (event["representation"] == "rolling_slope_log")
            ]
            ev_w = ev[ev[feat].fillna(False).astype(bool)]
            ev_wo = ev[~ev[feat].fillna(False).astype(bool)]
            rows.append(
                {
                    "feature": feat,
                    "group": group,
                    "n_runs_with": int(with_g["run"].nunique()),
                    "n_runs_without": int(without_g["run"].nunique()),
                    "frac_x_leads_y_with": float(
                        (with_g["direction"] == "x_leads_y").mean()
                    ),
                    "frac_x_leads_y_without": float(
                        (without_g["direction"] == "x_leads_y").mean()
                    ),
                    "delta_frac_x_leads_y": float(
                        (with_g["direction"] == "x_leads_y").mean()
                        - (without_g["direction"] == "x_leads_y").mean()
                    ),
                    "frac_bidirectional_with": float(
                        (with_g["direction"] == "bidirectional").mean()
                    ),
                    "frac_bidirectional_without": float(
                        (without_g["direction"] == "bidirectional").mean()
                    ),
                    "median_lag_with": float(with_g["x_to_y_lag_bic"].median()),
                    "median_lag_without": float(without_g["x_to_y_lag_bic"].median()),
                    "median_pre_slope_z_with": float(ev_w["pre_z_vs_baseline"].median())
                    if not ev_w.empty
                    else math.nan,
                    "median_pre_slope_z_without": float(
                        ev_wo["pre_z_vs_baseline"].median()
                    )
                    if not ev_wo.empty
                    else math.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["feature", "delta_frac_x_leads_y"], ascending=[True, False]
        )
    out.to_csv(out_dir / "feature_conditioned_temporal_summary.csv", index=False)
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_cli_args(ap)
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

    runs = select_runs(cfg, cfg.temporal, inv_df=inv_df, summary_df=summary_df)
    if not runs:
        raise SystemExit(
            "Empty run list. Tune preferred/filters in instability/config.py or pass --runs."
        )
    print(f"[temporal] {len(runs)} runs:")
    for r in runs:
        print(f"  - {r}")

    run_temporal_analysis(cfg.tb_root, out, runs)


if __name__ == "__main__":
    _cli()
