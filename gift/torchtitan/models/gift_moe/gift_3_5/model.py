"""GIFT 3.5 hybrid model: interleaved linear-attention (GDN/KDA) + MLA layers.

``GiftHybridModel`` keeps the base's ``prepare_config`` and overrides only the
two hooks that differ: which parallelisms the family rejects, and which sharding
config it applies. The generic decoder forward dispatches the mixed MLA /
linear-attention layers uniformly. FSDP (+ MoE EP) only.
"""

from typing import TypeAlias, cast

from torchtitan.config.schema.models.gift_moe.gift_3_5.model import (
    GiftHybridModelConfig,
)
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.models.gift_moe._base.model import GiftMoEModel

__all__ = ["GiftHybridModel"]


class GiftHybridModel(GiftMoEModel):
    """GiftMoE with a mix of linear-attention and MLA layers."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GiftHybridModelConfig

    @classmethod
    def _set_sharding_config(cls, model_config: BaseModelConfig, parallelism) -> None:
        model_config = cast(GiftHybridModelConfig, model_config)
        from torchtitan.models.gift_moe.gift_3_5.sharding import (
            set_gift_hybrid_sharding_config,
        )

        set_gift_hybrid_sharding_config(
            model_config,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
        )
