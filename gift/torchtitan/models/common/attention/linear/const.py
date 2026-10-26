"""Shared pieces of the linear-attention families (GDN, KDA)."""

from torchtitan.config.schema.models.common.attention.linear.const import (
    GatedNormActivation,
    linear_delta_core_flops,
)

__all__ = ["GatedNormActivation", "linear_delta_core_flops"]
