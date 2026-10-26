from .const import DEFAULT_NORM_EPS
from .gated_norm import GatedRMSNorm
from .groupnorm import GroupNorm
from .layernorm import LayerNorm
from .rmsnorm import RMSNorm

__all__ = [
    "DEFAULT_NORM_EPS",
    "NormConfig",
    "GatedRMSNorm",
    "GroupNorm",
    "LayerNorm",
    "RMSNorm",
]

# Union of the norm Configs accepted wherever a model declares a norm field
NormConfig = RMSNorm.Config | LayerNorm.Config | GroupNorm.Config | GatedRMSNorm.Config
