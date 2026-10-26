from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.const import (
    AttentionMaskKind,
    MaskType,
)
from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)

# torch.nn.attention.flex_attention._DEFAULT_SPARSE_BLOCK_SIZE, kept equal by a test.
_DEFAULT_SPARSE_BLOCK_SIZE = 128


@config(type="flex")
class FlexAttentionConfig(InnerAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.kernels.flex:FlexAttention"
    )
    supports_context_parallel: ClassVar[bool] = True

    block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE
    kernel_options: dict[str, bool | int | str] = field(default_factory=dict)

    mask_kind: ClassVar[AttentionMaskKind] = AttentionMaskKind.BLOCK_MASK
    supported_mask_types: ClassVar[tuple[MaskType, ...]] = (
        MaskType.BLOCK_CAUSAL,
        MaskType.CAUSAL,
    )

    def __post_init__(self) -> None:
        backend = self.kernel_options.get("BACKEND")
        if backend is not None and backend not in (
            "AUTO",
            "TRITON",
            "FLASH",
            "TRITON_DECODE",
        ):
            raise ValueError(
                "kernel_options.BACKEND must be AUTO, TRITON, FLASH or TRITON_DECODE"
            )
        if backend is not None and "FORCE_USE_FLEX_ATTENTION" in self.kernel_options:
            raise ValueError(
                "kernel_options.BACKEND cannot be combined with FORCE_USE_FLEX_ATTENTION"
            )

    def validate(self, ctx):
        # The flash lowering of flex_attention is Hopper+ only: a property of
        # the machine, so check it in the container, never pre-submit.
        if ctx.stage != "in_container" or self.kernel_options.get("BACKEND") != "FLASH":
            return []
        from torchtitan.config.validation import Issue
        from torchtitan.tools.utils import has_cuda_capability

        if has_cuda_capability(9, 0):
            return []
        return [
            Issue(
                path="kernel_options.BACKEND",
                message="FlexAttention with BACKEND='FLASH' requires "
                "Hopper or Blackwell (SM >= 9.0)",
            )
        ]
