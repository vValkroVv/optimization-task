"""``VarlenAttention`` -- inner kernel over FlashAttention ``varlen_attn``.

Consumes the family-agnostic varlen contract from ``attention/varlen_meta.py``
(packed documents + cu_seqlens metadata).
"""

from typing import TypeAlias

import torch
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
)
from torch.nn.attention.varlen import varlen_attn

from torchtitan.config.schema.models.common.attention.softmax.kernels.varlen import (
    VarlenAttentionConfig,
)
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.attention.varlen_meta import (
    pack_varlen_qkv,
    unpack_varlen_out,
    VarlenMetadata,
)
from torchtitan.protocols.module import Module


__all__ = ["VarlenAttention"]


class VarlenAttention(Module):
    # pyrefly: ignore [bad-override]
    Config: TypeAlias = VarlenAttentionConfig

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size

        from torchtitan.tools.utils import has_cuda_capability

        # Hopper (SM 9.0) uses FA3
        if has_cuda_capability(9, 0):
            if current_flash_attention_impl() != "FA3":
                activate_flash_attention_impl("FA3")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(attention_masks, VarlenMetadata), (
            f"attention_masks must be instance of VarlenMetadata but got {type(attention_masks)}"
        )

        cu_seq_q = attention_masks.cu_seq_q
        cu_seq_k = attention_masks.cu_seq_k
        max_q = attention_masks.max_q
        max_k = attention_masks.max_k

        batch_size, seq_len = q.shape[0], q.shape[1]

        # varlen attention expects (bs*seqlen, n_heads, head_dim)
        xq_packed, xk_packed, xv_packed = pack_varlen_qkv(q, k, v)

        # Some operators can upcast under AMP, but varlen attention currently only
        # supports bf16/fp16 inputs. If this changes, or fp16 training support
        # is added, this may need to be revisited.
        xq_packed = xq_packed.to(torch.bfloat16)
        xk_packed = xk_packed.to(torch.bfloat16)
        xv_packed = xv_packed.to(torch.bfloat16)

        varlen_kwargs = dict()

        # TODO(pytorch/pytorch#179760): FA2's auto num_splits heuristic
        # produces NaN intermittently with paged KV (block_table). Force
        # num_splits=1 as a workaround. current_flash_attention_impl()
        # returns None when FA2 is the implicit default (SM < 9.0).
        # For FA3, only force num_splits=1 in batch-invariant mode
        # to prevent non-deterministic split-k reductions.
        fa_impl = current_flash_attention_impl()
        if fa_impl in (None, "FA2") or is_in_batch_invariant_mode():
            varlen_kwargs["num_splits"] = 1

        # Forward enable_gqa from GQAttention when Q and KV head counts differ
        if kwargs.get("enable_gqa", False):
            varlen_kwargs["enable_gqa"] = True

        out_packed = varlen_attn(
            xq_packed,
            xk_packed,
            xv_packed,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            scale=scale,
            window_size=self.window_size,
            **varlen_kwargs,  # pyrefly: ignore [bad-argument-type]
        )
        assert isinstance(out_packed, torch.Tensor)
        # Reshape back to the format expected by GQAttention.forward()
        out = unpack_varlen_out(out_packed, batch_size, seq_len)

        return out.to(q.dtype)
