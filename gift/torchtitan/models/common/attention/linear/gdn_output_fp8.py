"""Gated RMSNorm with a rowwise E4M3 epilogue for GDN output projections."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from torchtitan.models.common.attention.linear.fla_custom_ops import (
    fla_vjp,
    gated_rmsnorm_op,
)
from torchtitan.models.common.attention.linear.gdn_output_norm_backward import (
    gdn_output_norm_backward,
)
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor, InputLayoutType
from torchtitan.ops.float8.fp8_utils import pad_x_to_y_divisible


@gluon.jit
def _norm_gate(x, gate, weight, rstd, SWISH: gl.constexpr):
    value = (x * rstd) * weight
    if SWISH:
        value *= gate
    return value * (1.0 / (1.0 + gl.exp(-gate)))


@gluon.jit
def _store_fp8(values, offsets, scale_offsets, mask, Quantized, Scales):
    # CudaCherry arranges the vec128 scale plane as [feature block, token].
    scale_unrounded = gl.maximum(
        gl.div_rn(gl.max(gl.abs(values), axis=1), 448.0), 1e-30
    )
    bits = scale_unrounded.to(gl.int32, bitcast=True)
    scale = ((bits + 0x007FFFFF) & 0x7F800000).to(gl.float32, bitcast=True)
    gl.store(Quantized + offsets, values * (1.0 / scale[:, None]), mask[:, None])
    gl.store(Scales + scale_offsets, scale, mask)


@gluon.jit(do_not_specialize=["HEAD_ROWS"])
def _gdn_output_norm_quantize_128_kernel(
    X,
    Gate,
    Weight,
    Quantized,
    Scales,
    HEAD_ROWS,
    NUM_HEADS: gl.constexpr,
    EPS: gl.constexpr,
    MULTIPLIER: gl.constexpr,
    SWISH: gl.constexpr,
    ZERO_CENTERED: gl.constexpr,
    BT: gl.constexpr,
):
    head_dim: gl.constexpr = 128
    # FLA uses eight scalars per lane and sixteen lanes per head. Pin the
    # layout so the FP8 store cannot change the FP32 norm reduction order.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [8, 1], [1, 0])
    dims = gl.arange(0, head_dim, layout=gl.SliceLayout(0, layout))
    rows = gl.program_id(0).to(gl.int64) * BT + gl.arange(
        0, BT, layout=gl.SliceLayout(1, layout)
    )
    offsets = rows[:, None] * head_dim + dims[None, :]
    mask = rows < HEAD_ROWS
    x = gl.load(X + offsets, mask[:, None], 0).to(gl.float32)
    gate = gl.load(Gate + offsets, mask[:, None], 0).to(gl.float32)
    weight = gl.load(Weight + dims).to(gl.float32)
    if ZERO_CENTERED:
        weight += 1.0
    rstd = 1 / gl.sqrt(gl.sum(x * x, axis=1) / head_dim + EPS)
    value = _norm_gate(x, gate, weight[None, :], rstd[:, None], SWISH)
    # FLA writes BF16 before GDN applies its multiplier in BF16.
    normed = value.to(gl.bfloat16).to(gl.float32)
    scaled = (normed * MULTIPLIER).to(gl.bfloat16).to(gl.float32)
    scale_offsets = (rows % NUM_HEADS) * (HEAD_ROWS // NUM_HEADS) + rows // NUM_HEADS
    _store_fp8(scaled, offsets, scale_offsets, mask, Quantized, Scales)


@triton.jit(do_not_specialize=["HEAD_ROWS"])
def _gdn_output_rstd_kernel(
    X,
    Rstd,
    eps,
    HEAD_ROWS,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    # Match FLA's tiled gated RMS forward, including its masked feature tail.
    first_row = tl.program_id(0).to(tl.int64) * BLOCK_HEADS
    x_ptr = tl.make_block_ptr(
        X,
        (HEAD_ROWS, HEAD_DIM),
        (HEAD_DIM, 1),
        (first_row, 0),
        (BLOCK_HEADS, BLOCK_DIM),
        (1, 0),
    )
    x = tl.load(x_ptr, boundary_check=(0, 1)).to(tl.float32)
    x = tl.where(tl.arange(0, BLOCK_DIM)[None, :] < HEAD_DIM, x, 0.0)
    rstd = 1 / tl.sqrt(tl.sum(x * x, axis=1) / HEAD_DIM + eps)
    rstd_ptr = tl.make_block_ptr(
        Rstd, (HEAD_ROWS,), (1,), (first_row,), (BLOCK_HEADS,), (0,)
    )
    tl.store(rstd_ptr, rstd, boundary_check=(0,))


@gluon.jit(do_not_specialize=["TOKENS", "LOGICAL_SEQ", "PADDED_SEQ"])
def _gdn_output_norm_quantize_blocks_kernel(
    X,
    Gate,
    Weight,
    Rstd,
    Quantized,
    Scales,
    TOKENS,
    LOGICAL_SEQ,
    PADDED_SEQ,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    NUM_BLOCKS: gl.constexpr,
    MULTIPLIER: gl.constexpr,
    SWISH: gl.constexpr,
    ZERO_CENTERED: gl.constexpr,
    NORMALIZED_INPUT: gl.constexpr,
    TOKEN_PADDING: gl.constexpr,
    TILE_BLOCKS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    rows = gl.program_id(0).to(gl.int64) * TILE_BLOCKS + gl.arange(
        0, TILE_BLOCKS, layout=gl.SliceLayout(1, layout)
    )
    tile_dims = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    tokens, blocks = rows // NUM_BLOCKS, rows % NUM_BLOCKS
    features = blocks[:, None] * 128 + tile_dims[None, :]
    valid_tokens = tokens < TOKENS
    logical_tokens = tokens
    if TOKEN_PADDING:
        sequence_offsets = tokens % PADDED_SEQ
        valid_tokens = valid_tokens & (sequence_offsets < LOGICAL_SEQ)
        logical_tokens = (tokens // PADDED_SEQ) * LOGICAL_SEQ + sequence_offsets

    value_dim: gl.constexpr = NUM_HEADS * HEAD_DIM
    valid = valid_tokens[:, None] & (features < value_dim)
    offsets = logical_tokens[:, None] * value_dim + features
    normed = gl.load(X + offsets, valid, 0).to(gl.float32)
    if not NORMALIZED_INPUT:
        gate = gl.load(Gate + offsets, valid, 0).to(gl.float32)
        weight = gl.load(Weight + features % HEAD_DIM, valid, 0).to(gl.float32)
        if ZERO_CENTERED:
            weight += 1.0
        head_rows = logical_tokens[:, None] * NUM_HEADS + features // HEAD_DIM
        rstd = gl.load(Rstd + head_rows, valid, 0).to(gl.float32)
        value = _norm_gate(normed, gate, weight, rstd, SWISH)
        normed = value.to(gl.bfloat16).to(gl.float32)

    scaled = (normed * MULTIPLIER).to(gl.bfloat16).to(gl.float32)
    scaled = gl.where(valid, scaled, 0.0)
    output_offsets = rows[:, None] * 128 + tile_dims[None, :]
    scale_offsets = blocks * TOKENS + tokens
    _store_fp8(
        scaled, output_offsets, scale_offsets, tokens < TOKENS, Quantized, Scales
    )


def _allocate_payload(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, head_dim = x.shape
    value_dim = heads * head_dim
    padded_dim = value_dim + pad_x_to_y_divisible(value_dim, 128)
    # Match FP8Linear's 3-D padding, where the flattened row count decides
    # whether the sequence dimension of each batch item needs padding.
    padded_seq = seq_len
    if (batch * seq_len) % 128 != 0:
        padded_seq += pad_x_to_y_divisible(seq_len, 128)
    return (
        x.new_empty((batch, padded_seq, padded_dim), dtype=torch.float8_e4m3fn),
        x.new_empty((padded_dim // 128, batch * padded_seq), dtype=torch.float32),
    )


@torch.library.custom_op("torchtitan::gdn_output_norm_quantize", mutates_args=())
def gdn_output_norm_quantize(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str,
    zero_centered: bool,
    multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write the padded FP8 payload from contiguous norm inputs."""
    data, scales = _allocate_payload(x)
    batch, seq_len, heads, head_dim = x.shape
    _, padded_seq, padded_dim = data.shape
    head_rows = batch * seq_len * heads
    swish = activation in ("silu", "swish")
    if head_dim == 128 and padded_seq == seq_len:
        tile_rows = 64 if zero_centered else 32
        _gdn_output_norm_quantize_128_kernel[(triton.cdiv(head_rows, tile_rows),)](
            x,
            gate,
            weight,
            data,
            scales,
            head_rows,
            heads,
            eps,
            multiplier,
            swish,
            zero_centered,
            tile_rows,
            num_warps=8,
            num_stages=2,
        )
        return data, scales

    # FLA selects a different, autotuned reduction for heads wider than 512.
    # Reuse its rounded output there and fuse the multiplier with quantization.
    normalized_input = head_dim > 512
    rstd = None
    if normalized_input:
        x = gated_rmsnorm_op(x, gate, weight, eps, activation, zero_centered)
    else:
        rstd = x.new_empty((head_rows,), dtype=torch.float32)
        tile_rows = 64 if head_dim == 128 and zero_centered else 32
        _gdn_output_rstd_kernel[(triton.cdiv(head_rows, tile_rows),)](
            x,
            rstd,
            eps,
            head_rows,
            head_dim,
            tile_rows,
            triton.next_power_of_2(head_dim),
            num_warps=8 if head_dim == 128 else 4,
            num_stages=2,
        )

    padded_tokens = batch * padded_seq
    num_blocks = padded_dim // 128
    tile_blocks = 32
    _gdn_output_norm_quantize_blocks_kernel[
        (triton.cdiv(padded_tokens * num_blocks, tile_blocks),)
    ](
        x,
        gate,
        weight,
        rstd,
        data,
        scales,
        padded_tokens,
        seq_len,
        padded_seq,
        heads,
        head_dim,
        num_blocks,
        multiplier,
        swish,
        zero_centered,
        normalized_input,
        padded_seq != seq_len,
        tile_blocks,
        num_warps=4,
    )
    return data, scales


@gdn_output_norm_quantize.register_fake
def _(x, gate, weight, eps, activation, zero_centered, multiplier):
    return _allocate_payload(x)


@torch._dynamo.allow_in_graph
class GDNOutputNormFP8(torch.autograd.Function):
    """Keep a BF16 gradient edge while passing the FP8 payload to FP8Linear."""

    @staticmethod
    def forward(
        ctx,
        x,
        gate,
        weight,
        eps,
        activation,
        zero_centered,
        multiplier,
        direct_backward,
    ):
        # FLA's input_guard uses contiguous norm inputs. Save that layout so
        # fla_vjp's fake outputs also match the native gradient strides.
        x, gate, weight = x.contiguous(), gate.contiguous(), weight.contiguous()
        ctx.save_for_backward(x, gate, weight)
        ctx.norm_args = (eps, activation, zero_centered, multiplier)
        ctx.direct_backward = direct_backward
        data, scale = gdn_output_norm_quantize(
            x, gate, weight, eps, activation, zero_centered, multiplier
        )
        output = Float8BlockwiseQTensor(
            data,
            scale,
            x.dtype,
            InputLayoutType.ROWWISE,
            1,
            scale_block_first=True,
        )
        batch, seq_len, heads, head_dim = x.shape
        output.orig_shape = (batch, seq_len, heads * head_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, gate, weight = ctx.saved_tensors
        eps, activation, zero_centered, multiplier = ctx.norm_args
        if ctx.direct_backward and grad_output.dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            grads = gdn_output_norm_backward(
                x, gate, weight, grad_output, eps, activation, zero_centered, multiplier
            )
            return *grads, None, None, None, None, None

        _, seq_len, heads, head_dim = x.shape
        grad_norm = (grad_output[:, :seq_len, : heads * head_dim] * multiplier).reshape(
            x.shape
        )
        grads = fla_vjp(
            "gated_rmsnorm",
            [x, gate, weight],
            [grad_norm],
            [],
            [eps, zero_centered],
            activation,
        )
        return *grads, None, None, None, None, None


def gdn_output_norm_fp8(
    x,
    gate,
    weight,
    eps,
    activation,
    zero_centered,
    multiplier,
    *,
    direct_backward=False,
):
    output = GDNOutputNormFP8.apply(
        x, gate, weight, eps, activation, zero_centered, multiplier, direct_backward
    )
    batch, seq_len, heads, head_dim = x.shape
    # Dynamo needs the logical shape in traced code for symbolic sequence sizes.
    output.orig_shape = (batch, seq_len, heads * head_dim)
    return output
