"""GDN's BF16 width-four short-convolution backward."""

import torch
import triton
import triton.language as tl


@triton.jit
def _short_conv_gradient_tile(
    QKV,
    Weight,
    Grad,
    start,
    length,
    columns,
    rows,
    head,
    QKV_DIM: tl.constexpr,
    GRAD_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NORMALIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    acc = tl.full((rows.shape[0], HEAD_DIM), 0, tl.float32)
    for tap in tl.static_range(4):
        source = rows + tap - 3
        x = tl.load(
            QKV + (start + source[:, None]) * QKV_DIM + columns[None, :],
            (source[:, None] >= 0) & (source[:, None] < length),
            0,
        ).to(tl.float32)
        weight = tl.load(Weight + columns * 4 + tap).to(tl.float32)
        acc += x * weight[None, :]

    dims = tl.arange(0, HEAD_DIM)
    grad = tl.load(
        Grad + ((start + rows[:, None]) * GRAD_HEADS + head) * HEAD_DIM + dims[None, :],
        rows[:, None] < length,
        0,
    ).to(tl.float32)
    if NORMALIZE:
        # FLA normalizes the BF16 SiLU output, then saves BF16 normalized Q/K.
        silu = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
        rstd = 1 / tl.sqrt(tl.sum(silu * silu, axis=1) + EPS)
        norm = (silu * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        dot = tl.sum(grad * norm, axis=1)
        grad = (
            ((grad - norm * dot[:, None]) * rstd[:, None])
            .to(tl.bfloat16)
            .to(tl.float32)
        )

    # FLA's activation derivative reads a separately rounded BF16 preactivation.
    pre_silu = acc.to(tl.bfloat16).to(tl.float32)
    sigmoid = tl.sigmoid(pre_silu)
    grad = grad * sigmoid * (1 + pre_silu * (1 - sigmoid))
    return tl.where(rows[:, None] < length, grad, 0)


@triton.jit(do_not_specialize=["T"])
def _gdn_short_conv_backward_kernel(
    QKV,
    Weight,
    GradQ,
    GradK,
    GradV,
    CuSeqLens,
    ChunkIndices,
    GradQKV,
    PartialWeight,
    T,
    QKV_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BT: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
    EPS: tl.constexpr,
    VARLEN: tl.constexpr,
):
    segment, chunk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if VARLEN:
        sequence = tl.load(ChunkIndices + chunk * 2)
        local_chunk = tl.load(ChunkIndices + chunk * 2 + 1).to(tl.int32)
        start = tl.load(CuSeqLens + sequence).to(tl.int64)
        length = tl.load(CuSeqLens + sequence + 1).to(tl.int64) - start
        partial = chunk.to(tl.int64)
    else:
        local_chunk = chunk
        start = batch.to(tl.int64) * T
        length = T
        partial = batch.to(tl.int64) * tl.num_programs(1) + chunk

    VALUE_GROUPS: tl.constexpr = NUM_V_HEADS // NUM_HEADS
    GROUP_SEGMENTS: tl.constexpr = 2 + VALUE_GROUPS
    head = segment // GROUP_SEGMENTS
    role = segment % GROUP_SEGMENTS
    columns = segment * HEAD_DIM + tl.arange(0, HEAD_DIM)
    local_rows = tl.arange(0, BT)
    rows = local_chunk * BT + local_rows
    halo_rows = (local_chunk + 1) * BT + tl.arange(0, 4)

    if role < 2:
        grad = GradQ if role == 0 else GradK
        dy = _short_conv_gradient_tile(
            QKV,
            Weight,
            grad,
            start,
            length,
            columns,
            rows,
            head,
            QKV_DIM,
            NUM_HEADS,
            HEAD_DIM,
            NORMALIZE_QK,
            EPS,
        )
        halo = _short_conv_gradient_tile(
            QKV,
            Weight,
            grad,
            start,
            length,
            columns,
            halo_rows,
            head,
            QKV_DIM,
            NUM_HEADS,
            HEAD_DIM,
            NORMALIZE_QK,
            EPS,
        )
    else:
        value_head = head * VALUE_GROUPS + role - 2
        dy = _short_conv_gradient_tile(
            QKV,
            Weight,
            GradV,
            start,
            length,
            columns,
            rows,
            value_head,
            QKV_DIM,
            NUM_V_HEADS,
            HEAD_DIM,
            False,
            EPS,
        )
        halo = _short_conv_gradient_tile(
            QKV,
            Weight,
            GradV,
            start,
            length,
            columns,
            halo_rows,
            value_head,
            QKV_DIM,
            NUM_V_HEADS,
            HEAD_DIM,
            False,
            EPS,
        )

    offsets = (start + rows[:, None]) * QKV_DIM + columns[None, :]
    x = tl.load(QKV + offsets, rows[:, None] < length, 0).to(tl.float32)
    dx = tl.full((BT, HEAD_DIM), 0, tl.float32)
    for shift in tl.static_range(4):
        if shift == 0:
            shifted = dy
        else:
            # Share the in-tile gradient and take the final rows from the halo.
            source = local_rows + shift
            in_tile = tl.broadcast_to((source % BT)[:, None], (BT, HEAD_DIM))
            in_halo = tl.broadcast_to((source % 4)[:, None], (BT, HEAD_DIM))
            shifted = tl.where(
                source[:, None] < BT,
                tl.gather(dy, in_tile, axis=0),
                tl.gather(halo, in_halo, axis=0),
            )
        tap = 3 - shift
        weight = tl.load(Weight + columns * 4 + tap).to(tl.float32)
        dx += shifted * weight[None, :]
        dw = tl.sum(shifted * x, axis=0)
        tl.store(PartialWeight + partial * (QKV_DIM * 4) + columns * 4 + tap, dw)
    tl.store(GradQKV + offsets, dx, rows[:, None] < length)


@torch.library.custom_op("torchtitan::gdn_short_conv_backward", mutates_args=())
def gdn_short_conv_backward(
    qkv: torch.Tensor,
    weight: torch.Tensor,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    num_heads: int,
    num_v_heads: int,
    normalize_qk: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_qkv, _, grad_weight = _short_conv_backward(
        qkv,
        weight,
        grad_q,
        grad_k,
        grad_v,
        cu_seqlens,
        num_heads,
        num_v_heads,
        normalize_qk,
        eps,
        quantize=False,
    )
    return grad_qkv, grad_weight


def _short_conv_backward(
    qkv,
    weight,
    grad_q,
    grad_k,
    grad_v,
    cu_seqlens,
    num_heads,
    num_v_heads,
    normalize_qk,
    eps,
    *,
    quantize,
):
    from fla.ops.common.fused_conv_qkv import _prepare_varlen_metadata

    batch, seq_len, qkv_dim = qkv.shape
    tile_rows = 64  # Use the same weight-gradient partials as the FLA backward.
    cu_seqlens, _, chunk_indices = _prepare_varlen_metadata(
        qkv, cu_seqlens, None, None, tile_rows
    )
    chunks = (
        len(chunk_indices)
        if chunk_indices is not None
        else triton.cdiv(seq_len, tile_rows)
    )
    x = qkv.contiguous()
    w = weight.reshape(qkv_dim, 4).contiguous()
    weight_partials = w.new_empty((batch * chunks, qkv_dim, 4), dtype=torch.float32)
    grid = (2 * num_heads + num_v_heads, chunks, batch)
    kernel_inputs = (
        x,
        w,
        grad_q.contiguous(),
        grad_k.contiguous(),
        grad_v.contiguous(),
        cu_seqlens,
        chunk_indices,
    )
    if quantize:
        from torchtitan.models.common.attention.linear.gdn_short_conv_fp8 import (
            gdn_short_conv_backward_fp8_kernel,
        )

        grad_qkv, grad_scales = _allocate_fp8(qkv)
        gdn_short_conv_backward_fp8_kernel[grid](
            *kernel_inputs,
            grad_qkv,
            grad_scales,
            weight_partials,
            seq_len,
            batch * seq_len,
            qkv_dim,
            num_heads,
            num_v_heads,
            normalize_qk,
            eps,
            cu_seqlens is not None,
            num_warps=4,
        )
    else:
        grad_qkv, grad_scales = torch.empty_like(x), None
        _gdn_short_conv_backward_kernel[grid](
            *kernel_inputs,
            grad_qkv,
            weight_partials,
            seq_len,
            qkv_dim,
            num_heads,
            num_v_heads,
            128,
            tile_rows,
            normalize_qk,
            eps,
            cu_seqlens is not None,
            num_warps=4,
        )
    grad_weight = weight_partials.sum(0).to(w.dtype).view_as(weight)
    return grad_qkv, grad_scales, grad_weight


@gdn_short_conv_backward.register_fake
def _(
    qkv,
    weight,
    grad_q,
    grad_k,
    grad_v,
    cu_seqlens,
    num_heads,
    num_v_heads,
    normalize_qk,
    eps,
):
    return torch.empty_like(
        qkv, memory_format=torch.contiguous_format
    ), torch.empty_like(weight, memory_format=torch.contiguous_format)


def _allocate_fp8(qkv):
    batch, seq_len, features = qkv.shape
    return (
        qkv.new_empty(qkv.shape, dtype=torch.float8_e4m3fn),
        qkv.new_empty((features // 128, batch * seq_len), dtype=torch.float32),
    )


@torch.library.custom_op("torchtitan::gdn_short_conv_backward_fp8", mutates_args=())
def gdn_short_conv_backward_fp8(
    qkv: torch.Tensor,
    weight: torch.Tensor,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    num_heads: int,
    num_v_heads: int,
    normalize_qk: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _short_conv_backward(
        qkv,
        weight,
        grad_q,
        grad_k,
        grad_v,
        cu_seqlens,
        num_heads,
        num_v_heads,
        normalize_qk,
        eps,
        quantize=True,
    )


@gdn_short_conv_backward_fp8.register_fake
def _(
    qkv,
    weight,
    grad_q,
    grad_k,
    grad_v,
    cu_seqlens,
    num_heads,
    num_v_heads,
    normalize_qk,
    eps,
):
    data, scale = _allocate_fp8(qkv)
    return data, scale, torch.empty_like(weight, memory_format=torch.contiguous_format)
