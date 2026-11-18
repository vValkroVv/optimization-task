"""Gated RMSNorm with a rowwise E4M3 epilogue for GDN output projections."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from torchtitan.models.common.attention.linear.fla_custom_ops import fla_vjp
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor, InputLayoutType


@gluon.jit(do_not_specialize=["HEAD_ROWS"])
def _gdn_output_norm_quantize_kernel(
    X,
    Gate,
    Weight,
    Quantized,
    Scales,
    HEAD_ROWS,
    NUM_HEADS: gl.constexpr,
    EPS: gl.constexpr,
    SWISH: gl.constexpr,
    ZERO_CENTERED: gl.constexpr,
    BT: gl.constexpr,
):
    head_dim: gl.constexpr = 128
    # FLA uses eight scalars per lane and sixteen lanes per head. Pin the
    # layout so the FP8 store cannot change the FP32 norm reduction order.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [8, 1], [1, 0])
    dims = gl.arange(0, head_dim, layout=gl.SliceLayout(0, layout))
    rows = gl.program_id(0) * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
    offsets = rows[:, None].to(gl.int64) * head_dim + dims[None, :]
    mask = rows < HEAD_ROWS
    x = gl.load(X + offsets, mask[:, None], 0).to(gl.float32)
    gate = gl.load(Gate + offsets, mask[:, None], 0).to(gl.float32)
    weight = gl.load(Weight + dims).to(gl.float32)
    if ZERO_CENTERED:
        weight += 1.0
    rstd = 1 / gl.sqrt(gl.sum(x * x, axis=1) / head_dim + EPS)
    value = (x * rstd[:, None]) * weight[None, :]
    if SWISH:
        value *= gate
    value *= 1.0 / (1.0 + gl.exp(-gate))

    # FLA writes BF16 before GDN's separate BF16 multiply by two.
    normed = value.to(gl.bfloat16).to(gl.float32)
    scaled = (normed * 2.0).to(gl.bfloat16).to(gl.float32)

    # Match CudaCherry's vec128 scale and its [head, flattened token] layout.
    scale_unrounded = gl.maximum(
        gl.div_rn(gl.max(gl.abs(scaled), axis=1), 448.0), 1e-30
    )
    bits = scale_unrounded.to(gl.int32, bitcast=True)
    scale = ((bits + 0x007FFFFF) & 0x7F800000).to(gl.float32, bitcast=True)
    codes = scaled * (1.0 / scale[:, None])
    gl.store(Quantized + offsets, codes, mask[:, None])
    scale_offsets = (rows % NUM_HEADS).to(gl.int64) * (
        HEAD_ROWS // NUM_HEADS
    ) + rows // NUM_HEADS
    gl.store(Scales + scale_offsets, scale, mask)


def _allocate_payload(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, head_dim = x.shape
    return (
        x.new_empty((batch, seq_len, heads * head_dim), dtype=torch.float8_e4m3fn),
        x.new_empty((heads, batch * seq_len), dtype=torch.float32),
    )


@torch.library.custom_op("torchtitan::gdn_output_norm_quantize", mutates_args=())
def gdn_output_norm_quantize(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str,
    zero_centered: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    data, scales = _allocate_payload(x)
    batch, seq_len, heads, _ = x.shape
    head_rows = batch * seq_len * heads
    tile_rows = 64 if zero_centered else 32
    _gdn_output_norm_quantize_kernel[(triton.cdiv(head_rows, tile_rows),)](
        x,
        gate,
        weight,
        data,
        scales,
        head_rows,
        heads,
        eps,
        activation in ("silu", "swish"),
        zero_centered,
        tile_rows,
        num_warps=8,
        num_stages=2,
    )
    return data, scales


@gdn_output_norm_quantize.register_fake
def _(x, gate, weight, eps, activation, zero_centered):
    return _allocate_payload(x)


@torch._dynamo.allow_in_graph
class GDNOutputNormFP8(torch.autograd.Function):
    """Keep a BF16 gradient edge while passing the FP8 payload to FP8Linear."""

    @staticmethod
    def forward(ctx, x, gate, weight, eps, activation, zero_centered):
        ctx.save_for_backward(x, gate, weight)
        ctx.norm_args = (eps, activation, zero_centered)
        data, scale = gdn_output_norm_quantize(
            x, gate, weight, eps, activation, zero_centered
        )
        return Float8BlockwiseQTensor(
            data,
            scale,
            x.dtype,
            InputLayoutType.ROWWISE,
            1,
            scale_block_first=True,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, gate, weight = ctx.saved_tensors
        eps, activation, zero_centered = ctx.norm_args
        grad_norm = (grad_output * 2.0).reshape(x.shape)
        grads = fla_vjp(
            "gated_rmsnorm",
            [x, gate, weight],
            [grad_norm],
            [],
            [eps, zero_centered],
            activation,
        )
        return *grads, None, None, None
