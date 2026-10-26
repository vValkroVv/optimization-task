"""Parameter-init functions for the gated-delta linear-attention families (GDN, KDA).

The A_log / dt_bias init math, kept next to the GDN/KDA modules that own those
parameters. Pure ``torch`` (no ``fla``), so it is importable without triggering the
fla-dependent module imports in this package. Models wire these into a
``param_init`` dict.
"""

import math

import torch
import torch.nn as nn

from torchtitan.config.schema.models.common.attention.linear.param_init import (
    GatedDeltaALog,
    GatedDeltaDtBias,
)
from torchtitan.protocols.module import register_param_init

__all__ = ["init_gated_delta_a_log", "init_gated_delta_dt_bias"]


@torch.no_grad()
def init_gated_delta_a_log(param: nn.Parameter) -> None:
    # A_log ~ Uniform(log 1, log 16).
    param.uniform_(math.log(1.0), math.log(16.0))


@torch.no_grad()
def init_gated_delta_dt_bias(param: nn.Parameter) -> None:
    # dt_bias = inverse-softplus of a log-uniform(dt_min, dt_max) draw clamped at
    # dt_init_floor; inverse softplus = x + log(-expm1(-x)).
    dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
    param.uniform_(math.log(dt_min), math.log(dt_max)).exp_().clamp_(min=dt_init_floor)
    param.add_(torch.log(-torch.expm1(-param)))


register_param_init(GatedDeltaALog)(lambda spec: init_gated_delta_a_log)
register_param_init(GatedDeltaDtBias)(lambda spec: init_gated_delta_dt_bias)
