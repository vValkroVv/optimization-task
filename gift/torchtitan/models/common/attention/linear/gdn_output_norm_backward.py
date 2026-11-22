"""Direct backward for the FP8 GDN output epilogue's gated RMSNorm."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit(
    do_not_specialize=[
        "head_rows",
        "seq_len",
        "tiles_per_program",
        "grad_batch_stride",
        "grad_seq_stride",
        "grad_feature_stride",
        "eps",
    ]
)
def _gdn_output_norm_backward_kernel(
    X,
    Gate,
    Weight,
    Grad,
    DX,
    DGate,
    Partials,
    head_rows,
    seq_len,
    tiles_per_program,
    grad_batch_stride,
    grad_seq_stride,
    grad_feature_stride,
    eps,
    MULTIPLIER: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    WIDTH: gl.constexpr,
    SWISH: gl.constexpr,
    ZERO_CENTERED: gl.constexpr,
    FLAT_GRAD: gl.constexpr,
    TILE_ROWS: gl.constexpr,
    TILE_WIDTH: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    partition = gl.program_id(0).to(gl.int64)
    row = gl.arange(0, TILE_ROWS, layout=gl.SliceLayout(1, LAYOUT))
    col = gl.arange(0, TILE_WIDTH, layout=gl.SliceLayout(0, LAYOUT))
    col_valid = col < WIDTH
    weight = gl.load(Weight + col, col_valid, 0).to(gl.float32)
    if ZERO_CENTERED:
        weight += 1.0

    partial = gl.full((TILE_ROWS, TILE_WIDTH), 0, gl.float32, LAYOUT)
    for tile in range(tiles_per_program):
        rows = (partition * tiles_per_program + tile) * TILE_ROWS + row
        row_valid = rows < head_rows
        valid = row_valid[:, None] & col_valid[None, :]
        offset = rows[:, None] * WIDTH + col[None, :]
        x = gl.load(X + offset, valid, 0).to(gl.float32)
        gate = gl.load(Gate + offset, valid, 0).to(gl.float32)
        if FLAT_GRAD:
            grad_offset = offset
        else:
            tokens = rows // NUM_HEADS
            grad_rows = (tokens // seq_len) * grad_batch_stride
            grad_rows += (tokens % seq_len) * grad_seq_stride
            features = (rows % NUM_HEADS)[:, None] * WIDTH + col[None, :]
            grad_offset = grad_rows[:, None] + features * grad_feature_stride
        grad = gl.load(Grad + grad_offset, valid, 0).to(gl.float32)
        # The multiply rounds in Grad's dtype, then autograd casts to the
        # recomputed FLA output's dtype (X's dtype) before its backward.
        grad = (grad * MULTIPLIER).to(Grad.dtype.element_ty)
        grad = grad.to(X.dtype.element_ty).to(gl.float32)

        rstd = 1 / gl.sqrt(gl.sum(x * x, axis=1) / WIDTH + eps)
        xhat = gl.where(col_valid[None, :], x * rstd[:, None], 0.0)
        y = xhat * weight[None, :]
        sigmoid = 1.0 / (1.0 + gl.exp(-gate))
        if SWISH:
            dgate = grad * y * (sigmoid + gate * sigmoid * (1 - sigmoid))
            dy = grad * gate * sigmoid
        else:
            dgate = grad * y * sigmoid * (1 - sigmoid)
            dy = grad * sigmoid

        weighted_dy = dy * weight[None, :]
        correction = gl.sum(xhat * weighted_dy, axis=1) / WIDTH
        dx = (weighted_dy - xhat * correction[:, None]) * rstd[:, None]
        partial += gl.where(valid, dy * xhat, 0.0)
        gl.store(DX + offset, dx, valid)
        # FLA rounds dGate to the input dtype before autograd casts to gate.dtype.
        gl.store(DGate + offset, dgate.to(X.dtype.element_ty), valid)

    gl.store(Partials + partition * WIDTH + col, gl.sum(partial, axis=0), col_valid)


@gluon.jit
def _gdn_output_norm_weight_grad_kernel(
    Partials,
    DWeight,
    WIDTH: gl.constexpr,
    NUM_PARTIALS: gl.constexpr,
    TILE_PARTIALS: gl.constexpr,
    TILE_COLS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    partial = gl.arange(0, TILE_PARTIALS, layout=gl.SliceLayout(1, layout))
    col = gl.program_id(0) * TILE_COLS + gl.arange(
        0, TILE_COLS, layout=gl.SliceLayout(0, layout)
    )
    tile = gl.load(
        Partials + partial[:, None].to(gl.int64) * WIDTH + col[None, :],
        (partial[:, None] < NUM_PARTIALS) & (col[None, :] < WIDTH),
        0,
    )
    gl.store(DWeight + col, gl.sum(tile, axis=0), col < WIDTH)


@torch.library.custom_op("torchtitan::gdn_output_norm_backward", mutates_args=())
def gdn_output_norm_backward(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float,
    activation: str,
    zero_centered: bool,
    multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate with the saved norm inputs and the projection's raw dgrad."""
    batch, seq_len, heads, width = x.shape
    head_rows = batch * seq_len * heads
    tile_width = triton.next_power_of_2(width)
    # Size the registers per program from the padded width. Wide rows use
    # more warps along the feature axis, without a width-specific dispatch limit.
    num_warps = min(16, max(4, tile_width // 1024))
    tile_rows = min(32, max(1, 4096 // tile_width))
    vector = 8 if width % 8 == 0 else 1
    feature_lanes = min(16, max(1, tile_width // vector))
    feature_warps = min(num_warps, max(1, tile_width // (feature_lanes * vector * 2)))
    layout = gl.BlockedLayout(
        [1, vector],
        [32 // feature_lanes, feature_lanes],
        [num_warps // feature_warps, feature_warps],
        [1, 0],
    )

    # Several programs per SM keep the streamed dX/dGate writes in flight.
    # Each program accumulates a fixed group of tiles before writing one partial.
    num_tiles = triton.cdiv(head_rows, tile_rows)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    tiles_per_program = triton.cdiv(num_tiles, 4 * sm_count)
    num_partials = triton.cdiv(num_tiles, tiles_per_program)
    dx, dgate, dweight = _allocate_grads(x, gate, weight)
    partials = x.new_empty((num_partials, width), dtype=torch.float32)
    grad_strides = grad_output.stride()
    flat_grad = grad_strides == (seq_len * heads * width, heads * width, 1)
    _gdn_output_norm_backward_kernel[(num_partials,)](
        x,
        gate,
        weight,
        grad_output,
        dx,
        dgate,
        partials,
        head_rows,
        seq_len,
        tiles_per_program,
        *grad_strides,
        eps,
        multiplier,
        heads,
        width,
        activation in ("silu", "swish"),
        zero_centered,
        flat_grad,
        tile_rows,
        tile_width,
        layout,
        num_warps=num_warps,
    )
    reduction_cols = 8
    _gdn_output_norm_weight_grad_kernel[(triton.cdiv(width, reduction_cols),)](
        partials,
        dweight,
        width,
        num_partials,
        triton.next_power_of_2(num_partials),
        reduction_cols,
        num_warps=4,
    )
    return dx, dgate, dweight


def _allocate_grads(x, gate, weight):
    return tuple(t.new_empty(t.shape) for t in (x, gate, weight))


@gdn_output_norm_backward.register_fake
def _(x, gate, weight, grad_output, eps, activation, zero_centered, multiplier):
    return _allocate_grads(x, gate, weight)
