from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.decoder_blocks.decoder import DecoderConfig


@config(type="gift_moe")
class GiftMoEModelConfig(DecoderConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.gift_moe._base.model:GiftMoEModel"
    family: ClassVar[str] = "gift_moe"
    supports_mtp_model_parallelism: ClassVar[bool] = True
    supported_sharding_backends: ClassVar[tuple[str, ...]] = ("fsdp2", "gfsdp2")
    _runtime_spec: ClassVar[str] = "torchtitan.models.gift_moe.versions:BASE"

    def validate_hf_export(self, *, step: float | None = None):
        from torchtitan.config.schema.models.gift_moe._base.hf_config import (
            build_hf_config,
        )
        from torchtitan.config.validation import Issue

        issues = self._residual_scale_hf_issues(step)
        try:
            build_hf_config(self)
        except ValueError as exc:
            issues.append(Issue(path="model.config", message=str(exc)))
        return issues
