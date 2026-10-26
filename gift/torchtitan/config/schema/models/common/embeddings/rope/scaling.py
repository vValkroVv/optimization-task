"""RoPE frequency-scaling configs; the math is in the runtime ``rope/scaling.py``."""

from __future__ import annotations

import math
from dataclasses import dataclass

from torchtitan.config.configurable import (
    component,
    component_family,
    to_plain,
)
from torchtitan.config.errors import ConfigFieldError


@component_family("rope_scaling", default="none")
@dataclass(frozen=True)
class ScalingConfig:
    """Base of the ``rope_scaling`` family; backends dispatch on the concrete class."""

    def to_dict(self) -> dict:
        """Serialize, naming the type: ``NoScaling`` would otherwise dump as ``{}``."""
        return to_plain(self)


@component("none")
@dataclass(frozen=True)
class NoScaling(ScalingConfig):
    """Frequencies straight from ``theta``; the default."""


@component("llama")
@dataclass(kw_only=True, frozen=True)
class LlamaScaling(ScalingConfig):
    """Llama3/4-style low/high/medium frequency scaling.

    ``high_freq_factor == low_freq_factor`` divides by zero, but the non-finite
    values land in the band ``torch.where`` discards -- which is why
    ``llama4_17bx16e`` builds. Unguarded to stay bit-identical. TODO: guard it.
    """

    scaling_factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192

    def __post_init__(self) -> None:
        for name in ("scaling_factor", "low_freq_factor", "high_freq_factor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigFieldError(name, f"{name} must be finite and positive")
        if self.original_max_position_embeddings <= 0:
            raise ConfigFieldError(
                "original_max_position_embeddings",
                "original_max_position_embeddings must be positive",
            )
        # Equal bounds are an existing Llama4 hard-transition preset.
        if self.high_freq_factor < self.low_freq_factor:
            raise ConfigFieldError(
                "high_freq_factor", "high_freq_factor must be >= low_freq_factor"
            )


@component("yarn")
@dataclass(kw_only=True, frozen=True)
class YarnScaling(ScalingConfig):
    """YaRN context extension (https://arxiv.org/abs/2309.00071).

    Present means active: ``rope_factor <= 1.0`` is rejected, so "requested but
    no-op" cannot happen. ``beta_fast`` is the higher rotation count, hence the
    lower dim index.
    """

    rope_factor: float
    original_seq_len: int
    beta_fast: float = 32.0
    beta_slow: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.rope_factor) or self.rope_factor <= 1.0:
            raise ConfigFieldError(
                "rope_factor",
                f"YarnScaling.rope_factor must be finite and > 1.0, got {self.rope_factor}. "
                "Use NoScaling for a model that does not extend its context.",
            )
        if self.original_seq_len <= 0:
            raise ConfigFieldError(
                "original_seq_len",
                f"YarnScaling.original_seq_len must be positive, got "
                f"{self.original_seq_len}.",
            )
        if not math.isfinite(self.beta_slow) or self.beta_slow <= 0:
            raise ConfigFieldError(
                "beta_slow", "YarnScaling.beta_slow must be finite and positive"
            )
        if not math.isfinite(self.beta_fast) or self.beta_fast <= self.beta_slow:
            raise ConfigFieldError(
                "beta_fast",
                f"YarnScaling needs beta_fast > beta_slow (the higher rotation "
                f"count maps to the lower dim index), got {self.beta_fast} and "
                f"{self.beta_slow}.",
            )


def _cos_sin_yarn_window(
    scaling: YarnScaling, *, dim: int, theta: float
) -> tuple[float, float]:
    """The cos/sin correction window, un-rounded.

    Not ``find_correction_range``: a different float operation order, and these
    numerics must stay bit-for-bit.
    """
    d_half = dim / 2
    low = (
        d_half
        * math.log(scaling.original_seq_len / (scaling.beta_fast * 2 * math.pi))
        / math.log(theta)
    )
    high = (
        d_half
        * math.log(scaling.original_seq_len / (scaling.beta_slow * 2 * math.pi))
        / math.log(theta)
    )
    return low, high


def validate_cos_sin_yarn_window(
    scaling: YarnScaling, *, dim: int, theta: float
) -> None:
    """Raise if the cos/sin YaRN correction window falls outside the head dim."""
    low, high = _cos_sin_yarn_window(scaling, dim=dim, theta=theta)
    d_half = dim / 2
    if not 0 < low < high < d_half - 1:
        raise ValueError(
            f"YaRN correction window must satisfy 0 < {low} < {high} < "
            f"{d_half - 1} for dim={dim}, theta={theta}, "
            f"original_seq_len={scaling.original_seq_len}, "
            f"beta_fast={scaling.beta_fast}, beta_slow={scaling.beta_slow}."
        )
