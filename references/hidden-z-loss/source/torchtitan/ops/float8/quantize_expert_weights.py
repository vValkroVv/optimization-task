import torch

from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor, InputLayoutType
from torchtitan.ops.float8.triton_kernels import block_quant_fn


@torch.library.custom_op("torchtitan::fp8_blockwise_quantize", mutates_args=())
def fp8_blockwise_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return block_quant_fn(w.contiguous())


@fp8_blockwise_quantize.register_fake
def _(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_groups, rows, cols = w.shape
    return (
        torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device),
        torch.empty(
            (num_groups, rows // 128, cols // 128), dtype=torch.float32, device=w.device
        ),
    )


class _QuantizeExpertWeight(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        data: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> Float8BlockwiseQTensor:
        if data is None:
            data, scale = torch.ops.torchtitan.fp8_blockwise_quantize(w)
        return Float8BlockwiseQTensor(
            data=data,
            scale=scale,
            orig_dtype=w.dtype,
            input_layout_type=InputLayoutType.ROWWISE,
            block_scaling_dim=2,
            scale_block_first=False,
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        # The wgrad belongs to the parameter; the planes are values, not edges.
        return grad_output, None, None


def quantize_expert_weight(
    w: torch.Tensor,
    quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Float8BlockwiseQTensor:
    """The fp8 view of an expert weight, carrying the parameter's gradient."""
    return _QuantizeExpertWeight.apply(w, *(quantized or (None, None)))
