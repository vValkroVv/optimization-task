"""``FlexAttention`` -- inner kernel over ``flex_attention`` with torch.compile."""

from collections.abc import Callable
from typing import ClassVar, TypeAlias

import torch
from torch.nn.attention.flex_attention import (
    _score_mod_signature,
    AuxRequest,
    BlockMask,
    flex_attention,
)

from torchtitan.config.schema.models.common.attention.softmax.kernels.flex import (
    FlexAttentionConfig,
)
from torchtitan.protocols.module import Module


__all__ = ["FlexAttention"]


class FlexAttention(Module):
    """Inner attention using ``flex_attention`` with torch.compile and CP support.

    Each backend handles its own layout transpose: ``forward()`` transposes from
    ``(bs, seq, heads, dim)`` to ``(bs, heads, seq, dim)`` before calling
    ``flex_attention``, and transposes back before returning.

    Note:
        The forward function must have q, k, v as the first three arguments
        to be compatible with _ContextParallel.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FlexAttentionConfig

    inductor_configs: ClassVar[dict[str, bool]] = {
        "wrap_inductor_compiled_regions": True,
        # Autotune to discover good kernel_options; set them explicitly and disable
        # for faster compilation on later runs.
        "max_autotune": True,
        # After autotune, tune individual params (block sizes, num_warps/stages)
        # one at a time, keeping >0.1% runtime wins.
        "coordinate_descent_tuning": True,
        "triton.cudagraphs": False,
    }

    # pyrefly: ignore[no-matching-overload]
    _compiled_flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention,
        options=inductor_configs,
    )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.kernel_options = config.kernel_options

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_masks: BlockMask | None = None,
        score_mod: _score_mod_signature | None = None,
        scale: float | None = None,
        return_lse: bool = False,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(attention_masks, (BlockMask, type(None))), (
            f"attention_masks must be instance of BlockMask or None, got {type(attention_masks)}"
        )

        # Transpose to (bs, heads, seq, dim) for flex_attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 1. _compiled_flex_attn has to be a class variable, otherwise there will
        #    be multiple compiled flex_attention instances, which can be slow.
        # 2. `self._compiled_flex_attn` is not correct, `self` will be passed in
        #    as the first argument, which will cause an error.
        #    `FlexAttention._compiled_flex_attn` is correct.
        out, aux = FlexAttention._compiled_flex_attn(
            q,
            k,
            v,
            block_mask=attention_masks,
            scale=scale,
            enable_gqa=enable_gqa,
            return_aux=AuxRequest(lse=return_lse),
            kernel_options=self.kernel_options,
        )
        # Transpose back to (bs, seq, heads, dim)
        if return_lse:
            return out.transpose(1, 2), aux.lse.transpose(1, 2)
        return out.transpose(1, 2)
