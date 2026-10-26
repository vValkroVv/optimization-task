"""Init specs shared by the gated-delta linear-attention families (GDN, KDA).

The math lives in the runtime module of the same name.
"""

from dataclasses import dataclass

from torchtitan.config.configurable import component
from torchtitan.config.schema.protocols.module import InitSpec

__all__ = ["GatedDeltaALog", "GatedDeltaDtBias"]


@component("gated_delta_a_log")
@dataclass(frozen=True, slots=True)
class GatedDeltaALog(InitSpec):
    """``A_log ~ Uniform(log 1, log 16)``."""


@component("gated_delta_dt_bias")
@dataclass(frozen=True, slots=True)
class GatedDeltaDtBias(InitSpec):
    """Inverse-softplus of a log-uniform ``(dt_min, dt_max)`` draw clamped at
    ``dt_init_floor``."""
