import math
from typing import TypeAlias

import torch
import torch.nn.functional as F


from torchtitan.config.schema.models.common.ffn import (
    compute_ffn_hidden_dim,
    FeedForwardConfig,
)
from torchtitan.models.common.regularization import (
    attach_hidden_z_loss,
    declare_fused_aux_loss,
    fused_aux_loss_state,
    HiddenZLoss,
)
from torchtitan.ops.float8 import make_zloss_state
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
        # The w1/w3 z-loss specs. Their hooks attach after the fp8 decision
        # below: the fused fp8 SwiGLU applies an output z-loss itself.
        z_loss_specs = (w1.hidden_z_loss, w3.hidden_z_loss)
        w1, w3 = (
            w if spec is None else w._copy(hidden_z_loss=None)
            for w, spec in zip((w1, w3), z_loss_specs)
        )
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
        self._check_input_z_loss(
            {"w1": z_loss_specs[0], "w2": w2.hidden_z_loss, "w3": z_loss_specs[1]}
        )
        self._hidden_z_loss_coefs = self._attach_hidden_z_loss(z_loss_specs)

    def _check_input_z_loss(self, specs: dict[str, HiddenZLoss | None]) -> None:
        """Refuse an input z-loss on a projection whose input is a packed fp8 operand.

        The z-loss hook reads a bf16 tensor. The shared quantizer packs the input
        of w1 and w3, and the fused SwiGLU packs the input of w2.

        Raises:
            ValueError: If such a spec has a coefficient that is not zero
        """
        fused = self.fp8_enabled and self.dense_fused_swiglu_quant
        packed = {
            "w1": self.use_shared_quantizer,
            "w2": fused,
            "w3": self.use_shared_quantizer,
        }
        for name, spec in specs.items():
            if (
                packed[name]
                and spec is not None
                and spec.on == "input"
                and spec.coef != 0.0
            ):
                raise ValueError(
                    f"FeedForward {name} has a hidden z-loss with on='input', but "
                    f"{name} gets a packed fp8 input on this path, and the z-loss "
                    "reads bf16. Use on='output', or put the spec on the FeedForward "
                    "itself, or set fp8.dense_use_shared_quantizer=false (w1, w3) or "
                    "fp8.dense_fused_swiglu_quant=false (w2)."
                )

    def _attach_hidden_z_loss(
        self, specs: tuple[HiddenZLoss | None, HiddenZLoss | None]
    ) -> tuple[float, float]:
        """Attach the w1/w3 hidden z-loss. Give the (w1, w3) coefficients of the kernel.

        The fused fp8 SwiGLU (``swiglu_quantize_zloss``) adds the z-loss gradient
        of gate and up before it rounds them to fp8. Thus on that path an output
        z-loss goes to the kernel, and the projection declares its splice for the
        names and the totals of ``wire_aux_loss``. Each other spec gets the hook
        of ``attach_hidden_z_loss``, as ``ModuleConfig.build`` attaches it.
        """
        fused = self.fp8_enabled and self.dense_fused_swiglu_quant
        coefs = []
        splices = []
        for module, spec in zip((self.w1, self.w3), specs):
            if fused and spec is not None and spec.on == "output":
                coefs.append(spec.coef)
                splices.append((module, "hidden_z_loss") if spec.coef else None)
                continue
            coefs.append(0.0)
            splices.append(None)
            if spec is not None:
                attach_hidden_z_loss(spec, module)
        if any(coefs):
            declare_fused_aux_loss(self, tuple(splices), make_zloss_state)
        return coefs[0], coefs[1]

    def fp8_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused blockwise fp8 SwiGLU FFN."""
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
        return self.w2(self._fused_swiglu(gate_out, up_out, base_shape))

    def _fused_swiglu(
        self,
        gate_out: torch.Tensor,
        up_out: torch.Tensor,
        base_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Give the packed fp8 SwiGLU of the padded w1/w3 outputs, for w2.

        ``swiglu_quantize_zloss`` also applies the w1/w3 output z-loss in
        training. Its backward gives gate and up their packed fp8 gradients.
        ``base_shape`` is the real (unpadded) shape of the FFN input. O(M * K).
        """
        from torchtitan.ops.float8._cudacherry import swiglu_quantize_zloss

        counts = self.swiglu_clip_counts if self.swiglu_limit > 0.0 else None
        prod_silu_out = swiglu_quantize_zloss(
            gate_out,
            up_out,
            dtype=torch.float8_e4m3fn,
            fp8_input_store=True,
            power_two_max_round=True,
            orig_shape=(*base_shape[:-1], self.w1.out_features),
            swiglu_limit=self.swiglu_limit,
            clip_counts=counts,
            coefs=self._hidden_z_loss_coefs if self.training else (0.0, 0.0),
            zloss_state=fused_aux_loss_state(self),
        )
        if counts is not None:
            # gate_out is padded to 128 in both dims; count the real elements.
            rows = math.prod(base_shape[:-1])
            counts[2].add_(rows * self.w1.out_features)
        return prod_silu_out

    def _swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return _clipped_swiglu(gate, up, self.swiglu_limit, self.swiglu_clip_counts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp8_enabled:
            return self.fp8_forward(x)
        return self.w2(self._swiglu(self.w1(x), self.w3(x)))
