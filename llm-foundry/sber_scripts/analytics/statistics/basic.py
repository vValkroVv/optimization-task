"""Basic numerical helpers shared by analysis components."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def smoothing(values: Iterable[float], coef: float) -> np.ndarray:
    """Apply TensorBoard-like exponential smoothing to a scalar series."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    coef = float(coef)
    if coef <= 0.0:
        return arr.copy()
    if coef >= 1.0:
        coef = min(coef, 0.999999)
    out = np.empty_like(arr, dtype=float)
    last = float(arr[0])
    out[0] = last
    for idx in range(1, arr.size):
        current = float(arr[idx])
        last = coef * last + (1.0 - coef) * current
        out[idx] = last
    return out


def aggregate_location(
    values: np.ndarray, *, center_aggregation: str = "mean"
) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("values must contain at least one finite value")
    mode = center_aggregation.strip().lower()
    if mode == "median":
        return float(np.median(finite))
    if mode == "mean":
        return float(np.mean(finite))
    raise ValueError(f"unsupported center_aggregation={center_aggregation!r}")
