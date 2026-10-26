"""Scalar logistic decoder for the rho=0 Mean-Var protocol."""

from __future__ import annotations

import math

VERDICTS = ("A>B", "B>A", "A=B")


def _sigmoid(value: float) -> float:
    # Both branches exponentiate a non-positive value, avoiding overflow.
    exp = math.exp(-abs(value))
    return 1.0 / (1.0 + exp) if value >= 0 else exp / (1.0 + exp)


def decode_pair(outputs: list[list[float]], protocol: dict) -> dict:
    if len(outputs) != 2 or any(len(row) != 2 for row in outputs):
        raise ValueError("backend must return exactly two [mu, sigma] vectors")
    if any(not math.isfinite(v) for row in outputs for v in row) or any(row[1] < protocol["sigma_floor"] for row in outputs):
        raise ValueError("backend returned nonfinite rewards or invalid sigma")
    margin = protocol["tie_margin"]
    difference_mu = outputs[0][0] - outputs[1][0]
    difference_sigma = math.hypot(outputs[0][1], outputs[1][1])
    lower_cdf = _sigmoid((-margin - difference_mu) / difference_sigma)
    upper_cdf = _sigmoid((margin - difference_mu) / difference_sigma)
    probabilities = [p + protocol["probability_floor"] for p in (1.0 - upper_cdf, lower_cdf, upper_cdf - lower_cdf)]
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    return {
        "verdict": VERDICTS[max(range(3), key=probabilities.__getitem__)],
        "probabilities": dict(zip(VERDICTS, probabilities, strict=True)),
        "rewards": {"A": outputs[0], "B": outputs[1]},
        "tie_margin": protocol["tie_margin"],
    }
