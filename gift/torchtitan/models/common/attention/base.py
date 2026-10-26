"""The contract every attention family declares to the decoder around it."""

from typing import TYPE_CHECKING, TypeAlias

from torchtitan.config.schema.models.common.attention.base import BaseAttentionConfig
from torchtitan.protocols.module import Module

__all__ = ["BaseAttention"]


if TYPE_CHECKING:
    pass


class BaseAttention(Module):
    """Base of every attention family: softmax, linear, sparse.

    Carries no architecture, only what the decoder must agree with: masking and
    the RoPE contract. The decoder builds one
    ``attention_masks`` object for a whole stack, so each family must declare
    what it needs from it before any layer runs.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = BaseAttentionConfig
