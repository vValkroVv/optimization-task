"""Gated DeltaNet (GDN) linear attention.

A gated delta-rule family (see :class:`GatedDeltaAttention`) and so a sibling of
``GQAttention`` / ``MLA``, not a softmax inner kernel: it takes no
``inner_attention``. Scan kernel and gated norm come from ``fla``, imported
lazily. RoPE unused.
TODO: training path only (chunk mode; no SP/CP, no cache).
"""

import torch
from torch import nn

from typing import TypeAlias

from torchtitan.config.schema.models.common.attention.linear.gdn import (
    GatedDeltaNetConfig,
)
from torchtitan.models.common.attention.const import AttentionMasksType
from torchtitan.models.common.attention.linear.fla_custom_ops import gdn_chunk_op
from torchtitan.models.common.attention.linear.gated_delta import GatedDeltaAttention
from torchtitan.models.common.attention.varlen_meta import (
    flatten_varlen_batch,
    unflatten_varlen_batch,
    VarlenMetadata,
)

__all__ = ["GatedDeltaNet"]


class GatedDeltaNet(GatedDeltaAttention):
    """Gating is one packed ``ba_proj`` (beta and the decay input) plus a
    ``z_proj`` gate for the o_norm."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GatedDeltaNetConfig

    def __init__(self, config: Config):
        super().__init__(config)
        ba_proj, z_proj = config._gating_projections()
        self.ba_proj = ba_proj.build()
        self.z_proj = z_proj.build()

        # Per value-head decay parameters (fp32). Initialized via the inherited
        # ``param_init`` (``GATED_DELTA_DECAY_INIT``) during ``init_states``.
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))

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
                    "GatedDeltaNet is configured for packed documents "
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

        # Decay/gating: ba_proj -> (beta b, decay-input a), each [.., L, num_v_heads].
        ba = self.ba_proj(x).view(
            b_, s_, self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads
        )
        b, a = torch.split(
            ba,
            [
                self.num_v_heads // self.num_k_heads,
                self.num_v_heads // self.num_k_heads,
            ],
            dim=3,
        )
        b = b.reshape(b_, s_, self.num_v_heads)
        a = a.reshape(b_, s_, self.num_v_heads)

        beta = b.sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Output gate, reshaped per value head for the gated RMSNorm.
        z = self.z_proj(x).view(b_, s_, self.num_v_heads, self.head_v_dim)

        o = gdn_chunk_op(
            q,
            k,
            v,
            a.contiguous(),
            beta,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
        )

        o = self.o_norm(o, z) * self.o_norm_output_multiplier
        o = o.reshape(b_, s_, self.value_dim)
        o = self.o_proj(o)
        if cu_seqlens is not None:
            o = unflatten_varlen_batch(o, bsz, seqlen)
        return o
