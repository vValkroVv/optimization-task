"""Q/K/V projection strategies (separate or fused) shared across attention.

These produce the per-head ``(xq, xk, xv)`` tensors an attention core consumes.
:class:`LinearAttnQKV` is the front-end for the linear-attention families
(``GatedDeltaNet`` / ``KimiDeltaAttention``); it holds the short-conv weight in a
plain :class:`Conv1d` (no ``fla`` at construction) and needs
``flash-linear-attention`` only at *forward* time, where the fused conv-unpack op runs.
"""

import torch

from typing import TypeAlias

from torchtitan.config.schema.models.common.attention.qkv import (
    BaseQKVLinearConfig,
    FusedQKVLinearConfig,
    LinearAttnQKVConfig,
    QKVLinearConfig,
)
from torchtitan.protocols.module import Module

__all__ = [
    "BaseQKVLinear",
    "FusedQKVLinear",
    "LinearAttnQKV",
    "QKVLinear",
]


class BaseQKVLinear(Module):
    """Base class for Q/K/V projection strategies.

    Subclasses implement different projection approaches (separate or fused)
    while providing a uniform interface to :class:`GQAttention`.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = BaseQKVLinearConfig

    def __init__(self, config: Config):
        super().__init__()
        assert config.head_dim is not None, "partial QKV config"
        self.head_dim = config.head_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input into Q, K, V tensors.

        Returns:
            (xq, xk, xv) each with shape [B, L, local_heads, head_dim].
        """
        raise NotImplementedError


class QKVLinear(BaseQKVLinear):
    """Three separate linear projections for Q, K, V."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = QKVLinearConfig

    def __init__(self, config: Config):
        super().__init__(config)
        assert config.wq is not None and config.wkv is not None, (
            "partial QKVLinear.Config"
        )
        self.wq = config.wq.build()
        self.wk = config.wkv.build()
        self.wv = config.wkv.build()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # Use -1 instead of n_heads (or n_kv_heads) to infer the
        # actual local heads from sizes as TP may have sharded them.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        return xq, xk, xv


class FusedQKVLinear(BaseQKVLinear):
    """Single fused linear projection, split along the R dimension
    (R = n_heads // n_kv_heads + 2 = Q-heads-per-KV-group + K + V). Fewer kernel
    launches than three separate projections; ColwiseParallel-compatible on ``wqkv``.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FusedQKVLinearConfig

    def __init__(self, config: Config):
        super().__init__(config)
        assert (
            config.wqkv is not None
            and config.n_heads is not None
            and config.n_kv_heads is not None
        ), "partial FusedQKVLinear.Config"
        self.wqkv = config.wqkv.build()
        self.heads_per_kv = config.n_heads // config.n_kv_heads
        self.r_dim = self.heads_per_kv + 2

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, seqlen, _ = x.shape
        # Fused QKV: single matmul, then reshape and split along R dim.
        # [B, L, n_kv_heads * R * head_dim] -> [B, L, n_kv_heads, R, head_dim]
        # Use -1 for n_kv_heads so TP sharding is handled automatically.
        qkv = self.wqkv(x)
        qkv = qkv.view(bs, seqlen, -1, self.r_dim, self.head_dim)
        # torch.split returns contiguous views for size-1 splits (xk, xv).
        # xq (size heads_per_kv) is non-contiguous; reshape triggers a copy.
        xq, xk, xv = torch.split(qkv, [self.heads_per_kv, 1, 1], dim=-2)
        xq = xq.reshape(bs, seqlen, -1, self.head_dim)
        xk = xk.reshape(bs, seqlen, -1, self.head_dim)
        xv = xv.reshape(bs, seqlen, -1, self.head_dim)
        return xq, xk, xv


class LinearAttnQKV(Module):
    """Q/K/V front-end shared by the linear-attention families (GDN / KDA).

    Deliberately not a :class:`BaseQKVLinear`: that base is the uniform view a
    softmax block checks its ``qkv_linear`` against, and counts heads one way.

    A single projection produces packed ``[2·key_dim + value_dim]`` features that
    ``fla``'s ``fused_qkv_conv1d_unpack`` (depthwise causal short conv + SiLU, also
    L2-normalizing q/k and resetting at document boundaries given ``cu_seqlens``)
    unpacks into per-head ``q, k, v``; ``num_v_heads`` is a multiple of
    ``num_k_heads``. The conv *weight* lives in a plain :class:`Conv1d` (forward
    unused — the fused op applies it), so ``fla`` is needed only at forward.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = LinearAttnQKVConfig

    def __init__(self, config: Config):
        super().__init__()
        self.head_k_dim = config.head_k_dim
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_v_dim = config.head_v_dim
        self.normalize_qk = config.normalize_qk
        self.eps = config.eps

        self.qkv_proj = config.qkv_proj.build()
        self.qkv_conv1d = config.conv.build()

    def forward(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torchtitan.models.common.attention.linear.fla_custom_ops import (
            qkv_conv1d_unpack_op,
        )

        qkv = self.qkv_proj(x)
        # q, k: [.., L, num_k_heads, head_k_dim]; v: [.., L, num_v_heads, head_v_dim].
        q, k, v = qkv_conv1d_unpack_op(
            qkv,
            self.qkv_conv1d.weight,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            cu_seqlens,
            self.normalize_qk,
            self.eps,
        )
        return q, k, v
