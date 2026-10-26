from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.const import (
    AttentionMaskKind,
    MaskType,
)
from torchtitan.config.schema.protocols.module import ModuleConfig


@config(family="inner_attention")
class InnerAttentionConfig(ModuleConfig):
    """Base config of an inner softmax kernel.

    The first member of ``supported_mask_types`` is the default mask type.
    """

    mask_kind: ClassVar[AttentionMaskKind] = AttentionMaskKind.NONE
    supported_mask_types: ClassVar[tuple[MaskType, ...]] = (MaskType.CAUSAL,)
    supports_context_parallel: ClassVar[bool] = False
    """Whether the runtime's Context Parallel wraps this kernel."""

    @classmethod
    def default_mask_type(cls) -> MaskType:
        return cls.supported_mask_types[0]
