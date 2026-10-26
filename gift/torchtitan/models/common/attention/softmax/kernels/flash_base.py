"""FlashAttention inner kernels backed by the ``flash_attn`` library.

Per-version concrete kernels live in ``flash_fa2/3/4.py``, expect
``(bs, seq, heads, dim)`` (the layout GQA/MLA already use, so no transpose) and
fp16/bf16 inputs; GQA is inferred from head counts. The packages are optional
and imported lazily, so a missing/ABI-broken build fails at ``.build()``, not at
import. TODO: Context parallelism is not supported with a flash backend.
"""

import importlib
from typing import ClassVar, TypeAlias

import torch

from torchtitan.config.schema.models.common.attention.softmax.kernels.flash_base import (
    FlashAttentionConfig,
    FlashVarlenAttentionConfig,
)
from torchtitan.models.common.attention.varlen_meta import (
    pack_varlen_qkv,
    unpack_varlen_out,
    VarlenMetadata,
)
from torchtitan.protocols.module import Module


__all__ = ["FlashAttention", "FlashVarlenAttention"]


class _FlashAttentionBase(Module):
    """Lazy-import plumbing; subclasses set ``_PACKAGE`` / ``_LABEL`` / ``_FN_NAME``.

    Two knobs absorb the FA2/FA3 vs FA4 (``flash_attn.cute``) API drift so the
    forwards below stay shared:

    - ``_LSE_KWARG``: FA2/FA3 request softmax stats via ``return_attn_probs``;
      FA4 via ``return_lse``.
    - ``_WINDOW_UNLIMITED_IS_NONE``: FA2/FA3 spell an unbounded side as ``-1``;
      FA4 uses ``None``. When set, ``-1`` entries are mapped to ``None``.
    """

    _PACKAGE: ClassVar[str] = ""
    _LABEL: ClassVar[str] = ""
    _FN_NAME: ClassVar[str] = ""
    _LSE_KWARG: ClassVar[str] = "return_attn_probs"
    _WINDOW_UNLIMITED_IS_NONE: ClassVar[bool] = False

    def __init__(
        self, config: FlashAttentionConfig | FlashVarlenAttentionConfig
    ) -> None:
        super().__init__()
        self.window_size = config.window_size
        self._flash_fn = self._load_flash_fn()

    @classmethod
    def _load_flash_fn(cls):
        try:
            mod = importlib.import_module(cls._PACKAGE)
        except ImportError as e:
            raise ValueError(
                f"{cls.__name__} requires the '{cls._PACKAGE}' "
                f"package, which failed to import ({type(e).__name__}: {e}). "
                "Install a flash-attn build matching your torch version, or name "
                "a torch-native kernel config (ScaledDotProductAttention / "
                "FlexAttention / VarlenAttention)."
            ) from e
        return getattr(mod, cls._FN_NAME)

    def _window(self) -> tuple:
        if self._WINDOW_UNLIMITED_IS_NONE:
            return tuple(None if w == -1 else w for w in self.window_size)
        return self.window_size


class FlashAttention(_FlashAttentionBase):
    """Dense flash attention over ``flash_attn_func``."""

    _FN_NAME = "flash_attn_func"

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FlashAttentionConfig

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        is_causal: bool = True,
        return_lse: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        in_dtype = q.dtype
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        out = self._flash_fn(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=is_causal,
            window_size=self._window(),
            # flash_attn doesn't honor torch's global deterministic flag, so
            # forward it explicitly (controls the backward dQ/dK/dV accumulation).
            # Set by --debug.deterministic via torch.use_deterministic_algorithms.
            deterministic=torch.are_deterministic_algorithms_enabled(),
            **{self._LSE_KWARG: return_lse},
        )
        if return_lse:
            # FA2 (out, lse, ...), FA3/FA4 (out, lse); lse (bs,heads,seq) ->
            # (bs,seq,heads) to match FlexAttention.
            return out[0].to(in_dtype), out[1].transpose(1, 2)
        # Some wrappers return a (out, ...) tuple even without LSE requested.
        out = out[0] if isinstance(out, tuple) else out
        return out.to(in_dtype)


class FlashVarlenAttention(_FlashAttentionBase):
    """Varlen flash attention over ``flash_attn_varlen_func`` (packed documents)."""

    _FN_NAME = "flash_attn_varlen_func"

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FlashVarlenAttentionConfig

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(attention_masks, VarlenMetadata), (
            f"attention_masks must be instance of VarlenMetadata but got {type(attention_masks)}"
        )

        batch_size, seq_len = q.shape[0], q.shape[1]
        in_dtype = q.dtype

        xq, xk, xv = pack_varlen_qkv(q, k, v)
        xq, xk, xv = xq.to(torch.bfloat16), xk.to(torch.bfloat16), xv.to(torch.bfloat16)

        out_packed = self._flash_fn(
            xq,
            xk,
            xv,
            attention_masks.cu_seq_q,
            attention_masks.cu_seq_k,
            attention_masks.max_q,
            attention_masks.max_k,
            softmax_scale=scale,
            causal=True,
            window_size=self._window(),
            # Honor --debug.deterministic (see dense forward).
            deterministic=torch.are_deterministic_algorithms_enabled(),
        )
        return unpack_varlen_out(out_packed, batch_size, seq_len).to(in_dtype)
