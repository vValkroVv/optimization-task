import math
from typing import TypeAlias

import torch
import torch.nn.functional as F


from torchtitan.config.schema.models.common.ffn import (
    compute_ffn_hidden_dim,
    FeedForwardConfig,
)
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

__all__ = ["FeedForward", "compute_ffn_hidden_dim"]


def _clipped_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: float,
    counts: torch.Tensor | None,
) -> torch.Tensor:
    """bf16 SwiGLU, clipped and counted as the fused fp8 kernel does."""
    if limit > 0.0:
        if counts is not None:
            counts[0].add_(gate.ge(limit).sum(dtype=torch.int64))
            up_clipped = up.le(-limit).logical_or(up.ge(limit))
            counts[1].add_(up_clipped.sum(dtype=torch.int64))
            counts[2].add_(gate.numel())
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return F.silu(gate) * up


class FeedForward(Module):
    """SwiGLU feed-forward module shared across models.

    Config takes the **final** hidden_dim (no internal 2/3 scaling).
    Use compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FeedForwardConfig

    def __init__(self, config: Config):
        super().__init__()
        w1, w2, w3 = config._projections()
        self.w1 = w1.build()
        self.w2 = w2.build()
        self.w3 = w3.build()
        from torchtitan.ops.float8.cuda_gemm import FP8Linear

        fp8_linears = [isinstance(w, FP8Linear) for w in (self.w1, self.w2, self.w3)]
        self.fp8_enabled = all(fp8_linears)
        if any(fp8_linears) and not self.fp8_enabled:
            raise ValueError(
                "FeedForward got a mix of FP8Linear and Linear projections "
                f"(w1/w2/w3 fp8 = {fp8_linears}); the fused fp8 SwiGLU path needs all "
                "three. Widen fp8.dense_include_linear_layers to cover w1|w2|w3."
            )
        if config.use_shared_quantizer and not self.fp8_enabled:
            logger.warning(
                "use_shared_quantizer is set but this FeedForward is not fp8; ignoring"
            )

        self.use_shared_quantizer = bool(
            config.use_shared_quantizer and self.fp8_enabled
        )
        self.dense_fused_swiglu_quant = config.dense_fused_swiglu_quant
        self.swiglu_limit = config.swiglu_limit
        self.swiglu_clip_counts: torch.Tensor | None = None
        z_loss_on_gate_up = any(
            w.hidden_z_loss is not None and w.hidden_z_loss.coef != 0.0
            for w in (w1, w3)
        )
        if self.fp8_enabled and self.dense_fused_swiglu_quant and z_loss_on_gate_up:
            raise ValueError(
                "FeedForward w1/w3 hidden z-loss needs the bf16 SwiGLU on the fp8 "
                "path: the fused SwiGLU+quant backward hands w1/w3 an fp8 gradient "
                "the z-loss gradient cannot be added to. Set "
                "fp8.dense_fused_swiglu_quant=False."
            )

    def fp8_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused blockwise fp8 SwiGLU FFN."""
        from torchtitan.ops.float8._cudacherry import swiglu_quantize
        from torchtitan.ops.float8.fp8_tensor import is_quantized_tensor
        from torchtitan.ops.float8.fp8_utils import SharedActivationQuantize

        if self.use_shared_quantizer and not is_quantized_tensor(x):
            input_shape = x.shape
            x = SharedActivationQuantize.apply(x)
            # Set again here, in traced code: the ``orig_shape`` the Function
            # assigns is invisible to Dynamo (``allow_in_graph``), which then
            # cannot rebuild its SymInts once the sequence length is dynamic.
            x.orig_shape = input_shape

        if not self.dense_fused_swiglu_quant:
            # bf16 SwiGLU on the unpadded projection outputs; w2 quantizes.
            return self.w2(self._swiglu(self.w1(x), self.w3(x)))

        base_shape = getattr(x, "orig_shape", x.shape)
        gate_out = self.w1(x, do_unpad_out=False)
        up_out = self.w3(x, do_unpad_out=False)
        counts = self.swiglu_clip_counts if self.swiglu_limit > 0.0 else None
        prod_silu_out = swiglu_quantize(
            gate_out,
            up_out,
            dtype=torch.float8_e4m3fn,
            fp8_input_store=True,
            power_two_max_round=True,
            orig_shape=(*base_shape[:-1], self.w1.out_features),
            swiglu_limit=self.swiglu_limit,
            clip_counts=counts,
        )
        if counts is not None:
            # gate_out is padded to 128 in both dims; count the real elements.
            rows = math.prod(base_shape[:-1])
            counts[2].add_(rows * self.w1.out_features)
        return self.w2(prod_silu_out)

    def _swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return _clipped_swiglu(gate, up, self.swiglu_limit, self.swiglu_clip_counts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp8_enabled:
            return self.fp8_forward(x)
        return self.w2(self._swiglu(self.w1(x), self.w3(x)))
