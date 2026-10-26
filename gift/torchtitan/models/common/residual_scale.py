"""Step-scheduled scalar on a residual branch: ``x + alpha * branch(x)``.

``alpha`` warms up from ``start_value`` to 1; the step comes from ``ProResWarmup``.
"""

import math
from typing import TypeAlias

import torch

from torchtitan.config.schema.models.common.residual_scale import (
    ResidualScaleConfig,
    ResidualScaleSchedule,
)
from torchtitan.protocols.module import Module

__all__ = ["ResidualScale", "ResidualScaleSchedule", "residual_scale_alpha"]


def residual_scale_alpha(
    step: int,
    *,
    warmup_steps: float,
    start_value: float,
    schedule: ResidualScaleSchedule,
    power: float = 1.0,
) -> float:
    """``start_value + (1 - start_value) * f(clamp(step / warmup_steps, 0, 1))``."""
    p = max(step, 0) / warmup_steps
    if p >= 1.0:
        # Exactly 1: cosine at p=1 rounds to 1 - 1e-16.
        return 1.0
    match schedule:
        case ResidualScaleSchedule.linear:
            f = p
        case ResidualScaleSchedule.sqrt:
            f = math.sqrt(p)
        case ResidualScaleSchedule.cosine:
            f = 1.0 - math.cos(math.pi * p / 2.0)
        case ResidualScaleSchedule.power:
            f = p**power
        case _:
            raise ValueError(f"unknown residual scale schedule {schedule!r}")
    return start_value + (1.0 - start_value) * f


class ResidualScale(Module):
    """``x * alpha``, ``alpha`` in a 0-dim non-persistent buffer.

    Not in the state_dict: ``alpha`` is a function of the step. No sharding
    config: a 0-dim tensor broadcasts against a DTensor as a replicated scalar.
    """

    alpha: torch.Tensor

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = ResidualScaleConfig

    def __init__(self, config: Config):
        super().__init__()
        self.warmup_steps = config.warmup_steps
        # An int would make an integer buffer that fill_ truncates.
        self.start_value = float(config.start_value)
        self.schedule = config.schedule
        self.power = config.power
        self._value = self.start_value
        self.register_buffer(
            "alpha",
            torch.tensor(self.start_value, dtype=torch.float32),
            persistent=False,
        )

    def alpha_at(self, step: int) -> float:
        return residual_scale_alpha(
            step,
            warmup_steps=self.warmup_steps,
            start_value=self.start_value,
            schedule=self.schedule,
            power=self.power,
        )

    def set_step(self, step: int) -> float:
        """Set ``alpha`` for ``step`` (0-based) and return it."""
        value = self.alpha_at(step)
        # No launch once saturated.
        if value != self._value:
            self.alpha.fill_(value)
            self._value = value
        return value

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device if buffer_device is not None else self.alpha.device
        self.alpha = torch.tensor(self.start_value, dtype=torch.float32, device=device)
        self._value = self.start_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eager skips the multiply once saturated. Under dynamo is_compiling() is
        # constant: one graph, no guard on _value.
        if not torch.compiler.is_compiling() and self._value == 1.0:
            return x
        return x * self.alpha
