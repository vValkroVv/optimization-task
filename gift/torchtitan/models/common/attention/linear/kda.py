"""Kimi Delta Attention (KDA) linear attention.

A gated delta-rule family (see :class:`GatedDeltaAttention`). Differs from GDN in
its gate features (per-channel ``g`` from a low-rank ``f_proj``), a
per-(value-head × key-dim) ``dt_bias`` / per-value-head ``A_log``, and the KDA
scan kernel (``chunk_kda``). Kernel and gated norm come from ``fla``, imported
lazily. RoPE unused.
TODO: training path only (chunk mode; no SP/CP, no cache).
"""

import torch
from torch import nn

from typing import TypeAlias

from torchtitan.config.schema.models.common.attention.linear.kda import (
    KimiDeltaAttentionConfig,
)
from torchtitan.models.common.attention.const import AttentionMasksType
from torchtitan.models.common.attention.linear.fla_custom_ops import kda_chunk_op
from torchtitan.models.common.attention.linear.gated_delta import GatedDeltaAttention
from torchtitan.models.common.attention.varlen_meta import (
    flatten_varlen_batch,
    unflatten_varlen_batch,
    VarlenMetadata,
)

__all__ = ["KimiDeltaAttention"]


class KimiDeltaAttention(GatedDeltaAttention):
    """Gating is a per-head-channel ``g`` and the output gate, both low-rank
    two-layer projections through ``head_v_dim``, plus a per-value-head ``beta``."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = KimiDeltaAttentionConfig

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate_dim = config.gate_dim

        b_proj, f_proj_in, f_proj_out, g_proj_in, g_proj_out = (
            config._gating_projections()
        )
        self.b_proj = b_proj.build()
        # f_proj / g_proj are 2-layer low-rank MLPs (no activation between).
        self.f_proj_in = f_proj_in.build()
        self.f_proj_out = f_proj_out.build()
        self.g_proj_in = g_proj_in.build()
        self.g_proj_out = g_proj_out.build()

        # A_log is per value-head (shape (1,1,num_v_heads,1)); dt_bias is per
        # (value-head × key-dim) = gate_dim. Both fp32. Initialized via the inherited
        # ``param_init`` (``GATED_DELTA_DECAY_INIT``) during ``init_states``.
        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.num_v_heads, 1, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(torch.empty(self.gate_dim, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # Varlen: fla's cu_seqlens kernels need a batch-1 packed stream, so flatten
        # [B, S, dim] -> [1, B*S, dim] (free reshape). cu_seqlens feeds both the QKV
        # short conv and the scan so each resets at document boundaries.
        if self.varlen:
            if attention_masks is None:
                raise ValueError(
                    "KimiDeltaAttention is configured for packed documents "
                    "(mask_type=block_causal) but got no attention_masks. Without "
                    "cu_seqlens the short conv and the scan carry state across "
                    "document boundaries, which degrades training silently "
                    "instead of failing."
                )
            assert isinstance(attention_masks, VarlenMetadata)
            cu_seqlens = attention_masks.cu_seq_q
            x = flatten_varlen_batch(x)
        else:
            cu_seqlens = None
        b_, s_, _ = x.shape

        # q, k: [.., L, num_k_heads, head_k_dim]; v: [.., L, num_v_heads, head_v_dim].
        q, k, v = self.qkv(x, cu_seqlens=cu_seqlens)

        g = self.f_proj_out(self.f_proj_in(x))
        g = g.view(b_, s_, self.num_v_heads, self.head_k_dim)
        beta = self.b_proj(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        o = kda_chunk_op(q, k, v, g, beta, self.A_log, self.dt_bias, cu_seqlens)

        o_gate = self.g_proj_out(self.g_proj_in(x))
        o_gate = o_gate.view(b_, s_, self.num_v_heads, self.head_v_dim)
        o = self.o_norm(o, o_gate) * self.o_norm_output_multiplier
        o = o.reshape(b_, s_, self.value_dim)
        o = self.o_proj(o)
        if cu_seqlens is not None:
            o = unflatten_varlen_batch(o, bsz, seqlen)
        return o
