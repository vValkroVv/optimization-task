from __future__ import annotations

import math
from typing import Literal

ProResScheduleName = Literal["linear", "sqrt", "cosine", "power"]

PRORES_SCHEDULES = frozenset({"linear", "sqrt", "cosine", "power"})


def validate_prores_config(
    *,
    warmup_steps: int,
    start_value: float,
    schedule: str,
    power: float,
) -> None:
    if warmup_steps <= 0:
        raise ValueError(
            f"prores_warmup_steps must be positive, got {warmup_steps}."
        )
    if not 0.0 <= start_value <= 1.0:
        raise ValueError(
            f"prores_start_value must be in [0, 1], got {start_value}."
        )
    if schedule not in PRORES_SCHEDULES:
        available = ", ".join(sorted(PRORES_SCHEDULES))
        raise ValueError(
            f"Unknown prores_schedule={schedule!r}. Available schedules: {available}."
        )
    if power <= 0.0:
        raise ValueError(f"prores_power must be positive, got {power}.")


def prores_schedule_value(
    progress: float,
    *,
    schedule: ProResScheduleName = "linear",
    power: float = 1.0,
) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    if schedule == "linear":
        return progress
    if schedule == "sqrt":
        return math.sqrt(progress)
    if schedule == "cosine":
        return 1.0 - math.cos(math.pi * progress / 2.0)
    if schedule == "power":
        return progress**power
    available = ", ".join(sorted(PRORES_SCHEDULES))
    raise ValueError(
        f"Unknown prores_schedule={schedule!r}. Available schedules: {available}."
    )


def prores_layer_warmup_steps(
    *,
    warmup_steps: int,
    layer_idx: int,
    num_layers: int,
) -> float:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(
            f"layer_idx must be in [0, {num_layers}), got {layer_idx}."
        )
    return float(warmup_steps) * float(layer_idx + 1) / float(num_layers)


def prores_alpha(
    *,
    step: int,
    layer_idx: int,
    num_layers: int,
    warmup_steps: int,
    start_value: float = 0.0,
    schedule: ProResScheduleName = "linear",
    power: float = 1.0,
) -> float:
    validate_prores_config(
        warmup_steps=warmup_steps,
        start_value=start_value,
        schedule=schedule,
        power=power,
    )
    layer_warmup = prores_layer_warmup_steps(
        warmup_steps=warmup_steps,
        layer_idx=layer_idx,
        num_layers=num_layers,
    )
    progress = float(max(step, 0)) / layer_warmup
    scheduled = prores_schedule_value(progress, schedule=schedule, power=power)
    return start_value + (1.0 - start_value) * scheduled


def prores_alphas(
    *,
    step: int,
    num_layers: int,
    warmup_steps: int,
    start_value: float = 0.0,
    schedule: ProResScheduleName = "linear",
    power: float = 1.0,
) -> list[float]:
    return [
        prores_alpha(
            step=step,
            layer_idx=layer_idx,
            num_layers=num_layers,
            warmup_steps=warmup_steps,
            start_value=start_value,
            schedule=schedule,
            power=power,
        )
        for layer_idx in range(num_layers)
    ]
