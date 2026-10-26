from typing import TypeAlias, cast

import torch.nn as nn

from torchtitan.config.schema.models.gift_moe._base.model import (
    GiftMoEModelConfig,
)
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.models.common.decoder_blocks import Decoder
from torchtitan.protocols.model import ModelRuntimeConfig


class GiftMoEModel(Decoder):
    """Base GiftMoE model: MLA attention + routed/shared-expert MoE."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GiftMoEModelConfig

    @classmethod
    def prepare_config(
        cls,
        model_config: BaseModelConfig,
        *,
        config: ModelRuntimeConfig,
        **kwargs,
    ) -> None:
        """Variants override ``validate_parallelism`` and ``_set_sharding_config``, not this."""
        model_config = cast(GiftMoEModelConfig, model_config)
        parallelism = config.parallelism

        Decoder.prepare_config(model_config, config=config, **kwargs)

        # TODO: sequence- and context-parallelism with MTP. FSDP / TP (no SP) / EP
        # / PP are supported.

        cls._set_sharding_config(model_config, parallelism)

    @classmethod
    def _set_sharding_config(cls, model_config: BaseModelConfig, parallelism) -> None:
        model_config = cast(GiftMoEModelConfig, model_config)
        from torchtitan.models.gift_moe._base.sharding import (
            set_gift_moe_sharding_config,
        )

        set_gift_moe_sharding_config(
            model_config,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
        )

    @classmethod
    def get_nparams_and_flops(
        cls, model_config: BaseModelConfig, model: "nn.Module", seq_len: int
    ) -> tuple[int, int]:
        model_config = cast(GiftMoEModelConfig, model_config)
        from torchtitan.models.utils import get_blockwise_nparams_and_flops

        return get_blockwise_nparams_and_flops(model_config, model, seq_len)
