"""Compute boundaries for a dense FFN nested inside a GFSDP block."""

from collections.abc import Callable

from torch import nn


class _CallableRegion(nn.Module):
    """A callable with a compile slot, without a second parameter registration."""

    def __init__(self, forward: Callable):
        super().__init__()
        self.forward = forward


def enable_ffn_overlap(unit: nn.Module) -> None:
    """Keep the registered module tree while exposing three compute regions.

    The unit's forward orchestrates these calls eagerly. GFSDP therefore sees
    the FFN's input-gradient boundary outside the parent's compiled backward.
    """
    block = getattr(unit, "block", unit)
    if block.moe_enabled or block.feed_forward is None:
        raise ValueError("FFN overlap requires a dense feed_forward block")
    if getattr(unit, "_ffn_overlap_regions", None) is not None:
        raise ValueError("FFN overlap was already configured for this block")
    unit._ffn_overlap_regions = (
        _CallableRegion(unit._forward_ffn_prefix),
        block.feed_forward,
        _CallableRegion(unit._forward_ffn_suffix),
    )
