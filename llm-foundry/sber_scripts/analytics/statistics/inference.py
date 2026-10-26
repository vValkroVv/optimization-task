"""Statistical inference for baseline-vs-variant metric comparisons."""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as t_dist

from sber_scripts.analytics.statistics.basic import aggregate_location


def normalize_center_aggregation(center_aggregation: str) -> str:
    mode = str(center_aggregation).strip().lower()
    if mode not in {"mean", "median"}:
        raise ValueError("center_aggregation must be one of: mean, median")
    return mode


def normalize_comparison_mode(comparison_mode: str) -> str:
    mode = str(comparison_mode).strip().lower()
    if mode not in {"last", "best", "auc", "mean", "median"}:
        raise ValueError(
            "comparison_mode must be one of: last, best, auc, mean, median"
        )
    return mode


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = np.cumsum(weights_sorted)
    idx = int(np.searchsorted(cumulative, 0.5, side="left"))
    idx = min(max(idx, 0), len(values_sorted) - 1)
    return float(values_sorted[idx])


def bayesian_bootstrap_location_difference_draws(
    baseline_values: np.ndarray,
    group_values: np.ndarray,
    center_aggregation: str,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw Bayesian-bootstrap samples for location difference ``group - baseline``."""
    baseline = np.asarray(baseline_values, dtype=float)
    group = np.asarray(group_values, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    group = group[np.isfinite(group)]
    if baseline.size == 0 or group.size == 0:
        raise ValueError(
            "Both baseline_values and group_values must contain finite values"
        )
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")

    center_aggregation = normalize_center_aggregation(center_aggregation)
    baseline_weights = rng.dirichlet(
        np.ones(baseline.size, dtype=float), size=int(n_draws)
    )
    group_weights = rng.dirichlet(np.ones(group.size, dtype=float), size=int(n_draws))

    if center_aggregation == "median":
        baseline_center = np.asarray(
            [
                weighted_median(baseline, baseline_weights[idx])
                for idx in range(int(n_draws))
            ],
            dtype=float,
        )
        group_center = np.asarray(
            [weighted_median(group, group_weights[idx]) for idx in range(int(n_draws))],
            dtype=float,
        )
    else:
        baseline_center = np.sum(baseline_weights * baseline[None, :], axis=1)
        group_center = np.sum(group_weights * group[None, :], axis=1)

    return np.asarray(group_center - baseline_center, dtype=float)


def compute_bayesian_bootstrap_inference(
    *,
    baseline_scores: np.ndarray,
    group_scores: np.ndarray,
    center_aggregation: str,
    ci_level: float = 0.9,
    n_draws: int = 20_000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compute Bayesian-bootstrap confidence interval and tail probability."""
    baseline = np.asarray(baseline_scores, dtype=float)
    group = np.asarray(group_scores, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    group = group[np.isfinite(group)]

    out: dict[str, Any] = {
        "bayes_status": "skipped",
        "bayes_ci_lower": np.nan,
        "bayes_ci_upper": np.nan,
        "bayes_prob_delta_gt_zero": np.nan,
        "bayes_n_draws": int(n_draws),
        "bayes_random_seed": int(random_seed),
        "bayes_error": "",
    }

    try:
        if baseline.size < 1 or group.size < 1:
            raise ValueError("need at least 1 finite score in both groups")
        if not np.isfinite(ci_level) or ci_level <= 0.0 or ci_level >= 1.0:
            raise ValueError("ci_level must be in (0, 1)")
        if int(n_draws) < 1:
            raise ValueError("n_draws must be >= 1")

        rng_bayes = np.random.default_rng(int(random_seed) + 1)
        draws_bayes = bayesian_bootstrap_location_difference_draws(
            baseline,
            group,
            center_aggregation,
            int(n_draws),
            rng_bayes,
        )
        alpha = (1.0 - float(ci_level)) / 2.0
        bayes_ci_bounds = np.asarray(
            np.quantile(draws_bayes, [alpha, 1.0 - alpha]), dtype=float
        )
        out.update(
            {
                "bayes_status": "ok",
                "bayes_ci_lower": float(bayes_ci_bounds[0]),
                "bayes_ci_upper": float(bayes_ci_bounds[1]),
                "bayes_prob_delta_gt_zero": float(np.mean(draws_bayes > 0.0)),
                "bayes_error": "",
            }
        )
    except Exception as exc:
        out.update({"bayes_status": "error", "bayes_error": str(exc)})

    return out


def aggregate_curve_score(
    values: np.ndarray,
    steps: np.ndarray,
    comparison_mode: str,
    center_aggregation: str,
    *,
    higher_is_better: bool = True,
) -> float:
    """Aggregate one trajectory into a scalar score.

    ``best`` respects metric direction: ``max`` when ``higher_is_better``,
    ``min`` otherwise.
    """
    values = np.asarray(values, dtype=float)
    steps = np.asarray(steps, dtype=float)
    if values.size == 0:
        raise ValueError("values must be non-empty")

    mode = normalize_comparison_mode(comparison_mode)
    center_aggregation = normalize_center_aggregation(center_aggregation)

    if mode == "last":
        return float(values[-1])
    if mode == "best":
        return float(np.max(values)) if higher_is_better else float(np.min(values))
    if mode == "auc":
        if values.size < 2:
            return float(values[-1])
        trapz_fn = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz_fn(values, x=steps) / max(steps[-1] - steps[0], 1e-12))
    if mode == "median":
        return aggregate_location(values, center_aggregation="median")
    return aggregate_location(values, center_aggregation=center_aggregation)


def interpolate_run_values_on_steps(
    run_steps: np.ndarray,
    run_values: np.ndarray,
    reference_steps: np.ndarray,
) -> np.ndarray:
    """Interpolate one run on reference steps."""
    if run_steps.size == 0 or run_values.size == 0:
        raise ValueError("run_steps and run_values must be non-empty")
    if run_steps.size != run_values.size:
        raise ValueError("run_steps and run_values must have the same length")

    order = np.argsort(run_steps)
    steps_sorted = np.asarray(run_steps[order], dtype=float)
    values_sorted = np.asarray(run_values[order], dtype=float)

    unique_steps, unique_indices = np.unique(steps_sorted, return_index=True)
    unique_values = values_sorted[unique_indices]
    if unique_steps.size == 0:
        raise ValueError("run contains no valid steps")

    return np.interp(
        np.asarray(reference_steps, dtype=float),
        unique_steps,
        unique_values,
        left=np.nan,
        right=np.nan,
    )


def exact_permutation_test_location_difference(
    baseline_values: np.ndarray,
    group_values: np.ndarray,
    *,
    center_aggregation: str = "mean",
    n_max_exact: int = 200_000,
    n_approx_draws: int = 100_000,
    approximate_random_seed: int = 42,
) -> dict[str, float | int | str]:
    """Compute exact (or approximate) one-sided permutation test for improvement."""
    baseline = np.asarray(baseline_values, dtype=float)
    group = np.asarray(group_values, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    group = group[np.isfinite(group)]
    center_aggregation = normalize_center_aggregation(center_aggregation)

    n_baseline = int(baseline.size)
    n_group = int(group.size)
    if n_baseline < 1 or n_group < 1:
        raise ValueError("Permutation test requires >=1 value in each group")

    n_total = n_baseline + n_group
    n_total_permutations = int(comb(n_total, n_baseline))

    pooled = np.concatenate([baseline, group]).astype(float)
    observed_delta = aggregate_location(
        group, center_aggregation=center_aggregation
    ) - aggregate_location(baseline, center_aggregation=center_aggregation)

    if n_total_permutations <= int(n_max_exact):
        ge_count = 0
        for baseline_indices in combinations(range(n_total), n_baseline):
            baseline_mask: NDArray[np.bool_] = np.zeros(n_total, dtype=bool)
            baseline_mask[list(baseline_indices)] = True
            baseline_sample = pooled[baseline_mask]
            group_sample = pooled[~baseline_mask]
            delta = aggregate_location(
                group_sample, center_aggregation=center_aggregation
            ) - aggregate_location(
                baseline_sample, center_aggregation=center_aggregation
            )
            if float(delta) >= float(observed_delta):
                ge_count += 1
        p_value = float(ge_count / max(n_total_permutations, 1))
        mode = "exact"
        n_permutations = n_total_permutations
    else:
        rng = np.random.default_rng(int(approximate_random_seed))
        draws = int(max(1, n_approx_draws))
        ge_count = 0
        for _ in range(draws):
            permuted = rng.permutation(pooled)
            baseline_sample = permuted[:n_baseline]
            group_sample = permuted[n_baseline:]
            delta = aggregate_location(
                group_sample, center_aggregation=center_aggregation
            ) - aggregate_location(
                baseline_sample, center_aggregation=center_aggregation
            )
            if float(delta) >= float(observed_delta):
                ge_count += 1
        p_value = float((ge_count + 1) / (draws + 1))
        mode = "approximate"
        n_permutations = draws

    return {
        "mode": mode,
        "n_permutations": int(n_permutations),
        "n_total_permutations": int(n_total_permutations),
        "observed_delta": float(observed_delta),
        "p_value_one_sided_improvement": float(p_value),
    }


def welch_ttest_one_sided_improvement(
    *,
    mean_baseline: float,
    std_baseline: float,
    n_baseline: int,
    mean_group: float,
    std_group: float,
    n_group: int,
) -> float:
    """Welch t-test p-value for H1: oriented group mean > baseline mean."""
    n_baseline = int(n_baseline)
    n_group = int(n_group)
    if n_baseline < 2 or n_group < 2:
        return np.nan
    se_b = float(std_baseline) / np.sqrt(n_baseline)
    se_g = float(std_group) / np.sqrt(n_group)
    se_diff_sq = se_b**2 + se_g**2
    if se_diff_sq <= 0:
        return np.nan
    se_diff = np.sqrt(se_diff_sq)
    t_stat = (float(mean_group) - float(mean_baseline)) / se_diff

    df_num = se_diff_sq**2
    df_den = (se_b**4) / (n_baseline - 1) + (se_g**4) / (n_group - 1)
    if df_den <= 0:
        return np.nan
    df = df_num / df_den
    if not np.isfinite(df) or df <= 0:
        return np.nan

    return float(1.0 - t_dist.cdf(t_stat, df))


def compute_comparison_inference(
    *,
    baseline_scores: np.ndarray,
    group_scores: np.ndarray,
    center_aggregation: str,
    ci_level: float,
    resampling_draws: int,
    random_seed: int,
    permutation_n_max_exact: int = 200_000,
    permutation_n_approx_draws: int = 100_000,
) -> dict[str, Any]:
    """Compute all currently supported inference diagnostics."""
    baseline = np.asarray(baseline_scores, dtype=float)
    group = np.asarray(group_scores, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    group = group[np.isfinite(group)]
    out: dict[str, Any] = {
        "inference_ci_level": float(ci_level),
        "inference_resampling_draws": int(resampling_draws),
        "inference_status": "skipped_no_scores",
        "inference_error": "",
        "n_baseline": int(baseline.size),
        "n_group": int(group.size),
        "center_baseline": np.nan,
        "center_group": np.nan,
        "delta_oriented": np.nan,
        "welch_p_value_one_sided_improvement": np.nan,
        "perm_test_status": "skipped",
        "perm_test_mode": "",
        "perm_n_permutations": np.nan,
        "perm_n_permutations_total": np.nan,
        "perm_p_value_one_sided_improvement": np.nan,
        "perm_stat_observed": np.nan,
    }
    if baseline.size == 0 or group.size == 0:
        return out

    try:
        center_aggregation = normalize_center_aggregation(center_aggregation)
        center_baseline = aggregate_location(
            baseline, center_aggregation=center_aggregation
        )
        center_group = aggregate_location(group, center_aggregation=center_aggregation)
        out.update(
            {
                "inference_status": "ok",
                "center_baseline": center_baseline,
                "center_group": center_group,
                "delta_oriented": center_group - center_baseline,
                "welch_p_value_one_sided_improvement": welch_ttest_one_sided_improvement(
                    mean_baseline=float(np.mean(baseline)),
                    std_baseline=float(np.std(baseline, ddof=1))
                    if baseline.size > 1
                    else np.nan,
                    n_baseline=int(baseline.size),
                    mean_group=float(np.mean(group)),
                    std_group=float(np.std(group, ddof=1))
                    if group.size > 1
                    else np.nan,
                    n_group=int(group.size),
                ),
            }
        )
        try:
            permutation = exact_permutation_test_location_difference(
                baseline,
                group,
                center_aggregation=center_aggregation,
                n_max_exact=permutation_n_max_exact,
                n_approx_draws=permutation_n_approx_draws,
                approximate_random_seed=random_seed,
            )
            out.update(
                {
                    "perm_test_status": "ok",
                    "perm_test_mode": permutation["mode"],
                    "perm_n_permutations": permutation["n_permutations"],
                    "perm_n_permutations_total": permutation["n_total_permutations"],
                    "perm_p_value_one_sided_improvement": permutation[
                        "p_value_one_sided_improvement"
                    ],
                    "perm_stat_observed": permutation["observed_delta"],
                }
            )
        except Exception as exc:
            out.update({"perm_test_status": "error", "perm_test_error": str(exc)})

        out.update(
            compute_bayesian_bootstrap_inference(
                baseline_scores=baseline,
                group_scores=group,
                center_aggregation=center_aggregation,
                ci_level=ci_level,
                n_draws=resampling_draws,
                random_seed=random_seed,
            )
        )
    except Exception as exc:
        out.update({"inference_status": "error", "inference_error": str(exc)})
    return out
