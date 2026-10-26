"""Grouped-Query (softmax) Attention and its Q/K/V projection strategies."""

import torch

from typing import TypeAlias

from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    causal_softmax_core_flops,
    GQAttentionConfig,
    SoftmaxAttentionConfig,
)
from torchtitan.models.common.attention.base import BaseAttention
from torchtitan.models.common.attention.const import AttentionMasksType
from torchtitan.models.common.embeddings.rope import (
    apply_rotary_emb_complex,
    apply_rotary_emb_cos_sin,
)
from torchtitan.models.common.norms import RMSNorm

__all__ = [
    "causal_softmax_core_flops",
    "GQAttention",
    "SoftmaxAttention",
]


class SoftmaxAttention(BaseAttention):
    """Base of the softmax families: an inner kernel over projected Q/K/V.

    Adds the head count, the kernel choice and the optional output gate.
    Masking belongs to the inner kernel: it supplies the default ``mask_type``
    and the set of values an explicit one must be in.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = SoftmaxAttentionConfig


class GQAttention(SoftmaxAttention):
    """Grouped-Query Attention; the Q/K/V projection strategy is set by the
    ``qkv_linear`` config field (:class:`QKVLinear` or :class:`FusedQKVLinear`)."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GQAttentionConfig

    def __init__(self, config: Config):
        super().__init__()
        assert config.dim is not None, "partial GQAttention.Config"
        self.n_heads = config.n_heads
        self.n_kv_heads = (
            config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.dim // config.n_heads
        )
        self.enable_gqa = self.n_heads > self.n_kv_heads
        self.use_rope = config.use_rope

        # Pluggable QKV projection
        assert config.qkv_linear is not None and config.wo is not None
        self.qkv_linear = config.qkv_linear.build()
        self.wo = config.wo.build()
        self.inner_attention = config.inner_attention.build()

        # Optional QK normalization
        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if config.qk_norm is not None:
            self.q_norm = config.qk_norm.build()
            self.k_norm = config.qk_norm.build()

        # Optional sigmoid output gate
        self.gate = config.gate.build() if config.gate is not None else None

        # Scaling factor (needed when head_dim differs from dim // n_heads)
        self.scaling = self.head_dim**-0.5 if config.head_dim is not None else None

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.qkv_linear(x)

        # Optional QK normalization (before RoPE)
        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # The cache says which format it is, so the pairing cannot be wrong.
        # NOTE: two-way only -- a third backend with a float cache lands in the
        # cos/sin branch silently. Give the cache an identity before adding one.
        if self.use_rope:
            if rope_cache.is_complex():
                xq, xk = apply_rotary_emb_complex(
                    xq, xk, freqs_cis=rope_cache, positions=positions
                )
            else:
                xq, xk = apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)

        # Handle iRoPE dict masks (Llama4)
        if isinstance(attention_masks, dict):
            mask_key = "rope" if self.use_rope else "nope"
            attention_masks = attention_masks[mask_key]

        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        output = output.view(bs, seqlen, -1)
        if self.gate is not None:
            output = self.gate(x, output)
        return self.wo(output)
