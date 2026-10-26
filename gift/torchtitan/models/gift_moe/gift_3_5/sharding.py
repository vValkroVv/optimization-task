"""Compatibility entrypoint for the shared module-based GIFT sharding plan."""

from torchtitan.models.gift_moe._base.sharding import (
    set_gift_moe_sharding_config as set_gift_hybrid_sharding_config,
)

__all__ = ["set_gift_hybrid_sharding_config"]
