"""Multi-head Latent Attention (MLA).

DeepSeek-V3 style low-rank Q/KV compression with a decoupled RoPE head. Shared by
``deepseek_v3`` and ``gift_moe``.
"""

import math
import os
from typing import TypeAlias

import torch
from torch.distributed.tensor import DTensor

from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig
from torchtitan.models.common.attention.const import AttentionMasksType
from torchtitan.models.common.attention.softmax.gqa import SoftmaxAttention
from torchtitan.models.common.embeddings.rope import (
    apply_rotary_emb_complex,
)
from torchtitan.models.common.regularization import (
    AddAuxiliaryLoss,
    aux_loss_args,
    declare_aux_loss,
    hidden_z_loss,
)
from torchtitan.tools.logging import logger

__all__ = ["MLA"]


class MLA(SoftmaxAttention):
    """MLA module. The inner softmax kernel (SDPA/Flex/Varlen) is pluggable via
    ``inner_attention``."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = MLAConfig

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self._mla_q_latent_z_loss_coef = config.q_latent_hidden_z_loss_coef
        self._mla_kv_latent_z_loss_coef = config.kv_latent_hidden_z_loss_coef
        if self.q_lora_rank == 0 and self._mla_q_latent_z_loss_coef != 0.0:
            logger.warning(
                "MLA q_latent_hidden_z_loss_coef is ignored when q_lora_rank=0: "
                "full-rank Q has no latent."
            )
            self._mla_q_latent_z_loss_coef = 0.0
        if self._mla_q_latent_z_loss_coef != 0.0:
            declare_aux_loss(self, "hidden_z_loss.q_latent")
        if self._mla_kv_latent_z_loss_coef != 0.0:
            declare_aux_loss(self, "hidden_z_loss.kv_latent")

        # Which path is populated is checked in Config.__post_init__.
        if self.q_lora_rank == 0:
            assert config.wq is not None
            self.wq = config.wq.build()
        else:
            assert config.wq_a is not None and config.wq_b is not None
            assert config.q_norm is not None
            self.wq_a = config.wq_a.build()
            self.q_norm = config.q_norm.build()
            self.wq_b = config.wq_b.build()

        # TODO(fegin): revisit
        # https://github.com/pytorch/torchtitan/pull/2785#discussion_r3034078575
        wkv_a, wkv_b, wo, kv_norm = config._projections()
        self.wkv_a = wkv_a.build()
        self.kv_norm = kv_norm.build()
        self.wkv_b = wkv_b.build()
        self.wo = wo.build()
        # Optional sigmoid output gate (gate_proj: dim -> n_heads*v_head_dim)
        self.gate = config.gate.build() if config.gate is not None else None

        # Latent up-projection scaling. q_hidden_dim is dim on the full-rank Q
        # path, so alpha_q == 1 there and only the q_lora_rank > 0 path is scaled.
        self.alpha_q = 1.0
        self.alpha_kv = 1.0
        if config.use_mla_scaling_factor:
            assert config.dim is not None, "partial MLA.Config"
            q_hidden_dim = config.dim if self.q_lora_rank == 0 else self.q_lora_rank
            self.alpha_q = math.sqrt(config.dim / q_hidden_dim)
            self.alpha_kv = math.sqrt(config.dim / self.kv_lora_rank)

        self.softmax_scale = self.qk_head_dim**-0.5

        # Into softmax_scale, not the cache: only part of the head is rotated but
        # the compensation must reach the whole logit. Squared: q and k both.
        if config.yarn_factor is not None:
            mscale = 0.1 * config.mscale * math.log(config.yarn_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = config.inner_attention.build()

        fused_rope = os.getenv("GIFT_MLA_FUSED_ROPE", "")
        if fused_rope not in ("", "0", "1"):
            raise ValueError(f"GIFT_MLA_FUSED_ROPE must be 0 or 1, got {fused_rope!r}")
        self._fused_rope = fused_rope == "1"

    def _rotate_query(self, q, freqs_cis, positions):
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        (q_pe,) = apply_rotary_emb_complex(
            q_pe, freqs_cis=freqs_cis, positions=positions
        )
        return torch.cat([q_nope, q_pe], dim=-1)

    def _rotate_shared_key(self, k_pe, freqs_cis, positions):
        (k_pe,) = apply_rotary_emb_complex(
            k_pe.unsqueeze(2), freqs_cis=freqs_cis, positions=positions
        )
        return k_pe

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.alpha_q * self.wq(x)
        else:
            q_latent = self.wq_a(x)
            q_latent = self.alpha_q * self.q_norm(q_latent)
            if self.training and self._mla_q_latent_z_loss_coef != 0.0:
                q_latent = AddAuxiliaryLoss.apply(
                    q_latent,
                    hidden_z_loss(q_latent, coef=self._mla_q_latent_z_loss_coef),
                    *aux_loss_args(self, "hidden_z_loss.q_latent"),
                )
            q = self.wq_b(q_latent)
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)

        use_fused_rope = (
            self._fused_rope
            and torch.compiler.is_compiling()
            and q.is_cuda
            and q.numel() > 0
            and q.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and self.qk_rope_head_dim > 0
            and self.qk_rope_head_dim % 2 == 0
            and freqs_cis.device == q.device
            and freqs_cis.dtype == torch.complex64
            and freqs_cis.ndim == 2
            and freqs_cis.shape[1] == self.qk_rope_head_dim // 2
            and not freqs_cis.requires_grad
            and not isinstance(q, DTensor)
            and not isinstance(freqs_cis, DTensor)
            and (
                freqs_cis.shape[0] >= seqlen
                if positions is None
                else (
                    positions.device == q.device
                    and positions.dtype in (torch.int32, torch.int64)
                    and positions.ndim == 2
                    and positions.shape[0] in (1, bsz)
                    and positions.shape[1] == seqlen
                    and not isinstance(positions, DTensor)
                )
            )
        )
        if not use_fused_rope:
            q = self._rotate_query(q, freqs_cis, positions)

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        if not use_fused_rope:
            k_pe = self._rotate_shared_key(k_pe, freqs_cis, positions)

        if self.training and self._mla_kv_latent_z_loss_coef != 0.0:
            kv = AddAuxiliaryLoss.apply(
                kv,
                hidden_z_loss(kv, coef=self._mla_kv_latent_z_loss_coef),
                *aux_loss_args(self, "hidden_z_loss.kv_latent"),
            )
        kv = self.alpha_kv * self.kv_norm(kv)
        kv = self.wkv_b(kv)
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        if (
            use_fused_rope
            and k_nope.device == k_pe.device == q.device
            and k_nope.dtype == k_pe.dtype == q.dtype
            and k_nope.shape[:3] == q.shape[:3]
            and not isinstance(k_nope, DTensor)
            and not isinstance(k_pe, DTensor)
        ):
            from torchtitan.models.common.attention.softmax.mla_rope_pack import (
                MLARoPEPack,
            )

            q, k = MLARoPEPack.apply(q, k_nope, k_pe, freqs_cis, positions)
        else:
            if use_fused_rope:
                q = self._rotate_query(q, freqs_cis, positions)
                k_pe = self._rotate_shared_key(k_pe, freqs_cis, positions)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        output = self.inner_attention(
            q, k, v, attention_masks=attention_masks, scale=self.softmax_scale
        ).contiguous()
        output = output.view(bsz, seqlen, -1)
        if self.gate is not None:
            output = self.gate(x, output)
        return self.wo(output)
