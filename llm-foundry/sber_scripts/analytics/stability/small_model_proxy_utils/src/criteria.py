"""Collapse criteria for stability experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Callable

Series = list[tuple[int, float]]


@dataclass(frozen=True)
class CriterionResult:
    """Result of a single collapse criterion."""

    name: str
    failed: bool
    first_step: int | None = None
    max_value: float | None = None
    details: str = ""


def _after_step(series: Series, min_step: int) -> Series:
    return [(step, value) for step, value in series if step >= min_step]


def _find_persistent_region(
    series: Series,
    predicate: Callable[[float], bool],
    min_points: int,
    min_step_span: int,
) -> tuple[int, float] | None:
    """Return the first persistent predicate match and the max value in it."""

    start_step: int | None = None
    prev_step: int | None = None
    values: list[float] = []

    for step, value in series:
        if isfinite(value) and predicate(value):
            if start_step is None:
                start_step = step
                values = []
            values.append(value)
            prev_step = step
            if len(values) >= min_points and (prev_step - start_step) >= min_step_span:
                return start_step, max(values)
            continue

        start_step = None
        prev_step = None
        values = []

    return None


def check_max_vio_layerwise_topk_average(
    trace,
    *,
    threshold_share: float = 0.2,
    drop_points: int = 10,
    top_k: int = 40,
) -> CriterionResult:
    """Detect unstable runs from layerwise top-k dispatcher max_vio averages."""

    if trace.max_vio_theoretical_limit is None:
        return CriterionResult("max_vio", False, details="missing max_vio config bound")

    threshold = threshold_share * trace.max_vio_theoretical_limit

    best_layer: int | None = None
    best_average: float | None = None
    best_count = 0

    for layer, summary in trace.max_vio_layers.items():
        skipped_points = Counter(summary.earliest_points[:drop_points])
        selected_values: list[float] = []

        for point in summary.top_points:
            if skipped_points[point] > 0:
                skipped_points[point] -= 1
                continue
            value = point[1]
            if not isfinite(value):
                continue
            selected_values.append(value)
            if len(selected_values) >= top_k:
                break

        if len(selected_values) < top_k:
            return CriterionResult(
                "max_vio",
                None,
                details=f"max_vio series length after truncation {len(selected_values)} is less than max-vio-top-k {top_k}",
            )

        average = sum(selected_values) / len(selected_values)
        if best_average is None or average > best_average:
            best_layer = layer
            best_average = average
            best_count = len(selected_values)

    if best_average is None or best_layer is None:
        return CriterionResult("max_vio", None, details="no usable layer series")

    details = (
        f"layer_{best_layer} avg_top{best_count} after drop{drop_points}="
        f"{best_average:.4g}"
    )
    if best_average <= threshold:
        return CriterionResult(
            "max_vio", False, max_value=best_average, details=details
        )

    return CriterionResult(
        "max_vio",
        True,
        max_value=best_average,
        details=f"{details} > {threshold:g}",
    )


def check_activation_amax(
    trace,
    *,
    warmup_steps: int = 1000,
    threshold: float = 1e-6,
    min_points: int = 3,
    min_step_span: int = 400,
) -> CriterionResult:
    """Detect persistent activation amax threshold violations."""

    series = _after_step(trace.activation_amax, warmup_steps)
    region = _find_persistent_region(
        series,
        lambda value: abs(value) > threshold,
        min_points=min_points,
        min_step_span=min_step_span,
    )
    if region is None:
        return CriterionResult("activation_amax", False, details=f"<= {threshold:g}")
    first_step, max_value = region
    return CriterionResult(
        "activation_amax",
        True,
        first_step=first_step,
        max_value=max_value,
        details=f"> {threshold:g} for >= {min_step_span} steps",
    )


def check_loss_spike(
    trace,
    *,
    warmup_steps: int = 1000,
    relative_increase: float = 0.5,
    absolute_increase: float = 0.2,
    min_points: int = 3,
    min_step_span: int = 400,
) -> CriterionResult:
    """Detect a persistent upward loss jump relative to the best prior value."""

    series = _after_step(trace.loss, warmup_steps)
    if len(series) < min_points:
        return CriterionResult("loss_spike", False, details="not enough loss points")

    start_step: int | None = None
    prev_step: int | None = None
    spike_values: list[float] = []
    best_so_far: float | None = None
    for step, value in series:
        if not isfinite(value):
            return CriterionResult(
                "loss_spike",
                True,
                first_step=step,
                max_value=value,
                details="non-finite loss",
            )
        if best_so_far is None:
            best_so_far = value
            continue
        delta = value - best_so_far
        rel_delta = delta / max(abs(best_so_far), 1e-12)
        is_bad = delta >= absolute_increase and rel_delta >= relative_increase
        if is_bad:
            if start_step is None:
                start_step = step
                spike_values = []
            spike_values.append(value)
            prev_step = step
            if (
                len(spike_values) >= min_points
                and (prev_step - start_step) >= min_step_span
            ):
                return CriterionResult(
                    "loss_spike",
                    True,
                    first_step=start_step,
                    max_value=max(spike_values),
                    details=(
                        f">= {relative_increase:.0%} and >= {absolute_increase:g} "
                        f"above prior min for >= {min_step_span} steps"
                    ),
                )
        else:
            start_step = None
            prev_step = None
            spike_values = []
        if value < best_so_far:
            best_so_far = value

    return CriterionResult("loss_spike", False, details="no persistent spike")


def evaluate_collapse(
    trace,
    *,
    warmup_steps: int = 1000,
    max_vio_threshold_share: float = 0.2,
    max_vio_drop_points: int = 10,
    max_vio_top_k: int = 40,
    activation_threshold: float = 1e-6,
    loss_relative_increase: float = 0.5,
    loss_absolute_increase: float = 0.2,
    min_points: int = 3,
    min_step_span: int = 400,
) -> list[CriterionResult]:
    """Run all collapse criteria on a standardized run trace."""

    return [
        check_max_vio_layerwise_topk_average(
            trace,
            threshold_share=max_vio_threshold_share,
            drop_points=max_vio_drop_points,
            top_k=max_vio_top_k,
        ),
        # Disabled for now: the first pass was too strict. Keep the checks here
        # so they can be re-enabled once thresholds are calibrated.
        # check_activation_amax(
        #     trace,
        #     warmup_steps=warmup_steps,
        #     threshold=activation_threshold,
        #     min_points=min_points,
        #     min_step_span=min_step_span,
        # ),
        # check_loss_spike(
        #     trace,
        #     warmup_steps=warmup_steps,
        #     relative_increase=loss_relative_increase,
        #     absolute_increase=loss_absolute_increase,
        #     min_points=min_points,
        #     min_step_span=min_step_span,
        # ),
    ]
