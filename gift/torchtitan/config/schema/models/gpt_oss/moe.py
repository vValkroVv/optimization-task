from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.moe.const import ScoreFunc
from torchtitan.config.schema.models.common.moe.experts.gmm import GroupedExpertsConfig
from torchtitan.config.schema.models.common.moe.moe import MoEConfig
from torchtitan.config.schema.models.common.moe.router import (
    TokenChoiceTopKRouterConfig,
)


@config
class GptOssGroupedExpertsConfig(GroupedExpertsConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.gpt_oss.moe:GptOssGroupedExperts"

    swiglu_limit: float = 7.0
    # mlp1 interleaves gate and up, so no projection takes hidden z-loss.
    hidden_z_loss_projections: ClassVar[dict[str, str]] = {}

    def __post_init__(self) -> None:
        # slots=True breaks super().
        GroupedExpertsConfig.__post_init__(self)
        if self.w1_hidden_z_loss_coef or self.w3_hidden_z_loss_coef:
            raise ValueError(
                "GptOssGroupedExperts does not implement the w1/w3 hidden "
                "z-loss; leave w1_hidden_z_loss_coef / w3_hidden_z_loss_coef "
                "at 0.0."
            )


@config(type="gpt_oss_biased")
class GptOssMoEConfig(MoEConfig):
    """Token-choice MoE with biased experts, a softmax route-normalized router
    whose gate carries a bias, and no shared experts."""

    _owner_spec: ClassVar[str] = MoEConfig._owner_spec

    # pyrefly: ignore [bad-override]
    experts: GptOssGroupedExpertsConfig
    router: TokenChoiceTopKRouterConfig = field(
        default_factory=lambda: TokenChoiceTopKRouterConfig(
            score_func=ScoreFunc.SOFTMAX, route_norm=True
        )
    )

    gate_bias: ClassVar[bool | None] = True
