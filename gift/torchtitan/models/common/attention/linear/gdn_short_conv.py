"""Fused backward for GDN's packed depthwise causal convolution and Q/K norm."""

import torch
import triton
import triton.language as tl


@triton.jit
def _short_conv_gradient_tile(
    QKV,
    Weight,
    Grad,
    start,
    rows,
    row_end,
    columns,
    dims,
    width,
    head,
    grad_heads,
    is_value,
    QKV_DIM: tl.constexpr,
    TAPS: tl.constexpr,
    PACKED_GRAD_DTYPE: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
    EPS: tl.constexpr,
):
    valid = (rows[:, None] < row_end) & (dims[None, :] < width)
    weight_offsets = columns.to(tl.int64) * TAPS
    acc = tl.full((rows.shape[0], dims.shape[0]), 0, tl.float32)
    for tap in tl.static_range(TAPS):
        source = rows + tap - (TAPS - 1)
        x = tl.load(
            QKV + (start + source[:, None]) * QKV_DIM + columns[None, :],
            valid & (source[:, None] >= 0),
            0,
        ).to(tl.float32)
        weight = tl.load(Weight + weight_offsets + tap, dims < width, 0).to(tl.float32)
        acc += x * weight[None, :]

    grad = tl.load(
        Grad + ((start + rows[:, None]) * grad_heads + head) * width + dims[None, :],
        valid,
        0,
    ).to(tl.float32)
    if NORMALIZE_QK:
        if not is_value:
            # FLA rounds SiLU and the norm gradient to BF16 for every input dtype.
            silu = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
            rstd = 1 / tl.sqrt(tl.sum(silu * silu, axis=1) + EPS)
            norm = (silu * rstd[:, None]).to(QKV.dtype.element_ty).to(tl.float32)
            dot = tl.sum(grad * norm, axis=1)
            grad = (
                ((grad - norm * dot[:, None]) * rstd[:, None])
                .to(tl.bfloat16)
                .to(tl.float32)
            )

    # FLA stores the packed gradient and separately recomputed preactivation
    # before the SiLU derivative. The convolution gradients accumulate in FP32.
    grad = grad.to(PACKED_GRAD_DTYPE).to(tl.float32)
    pre_silu = acc.to(QKV.dtype.element_ty).to(tl.float32)
    sigmoid = tl.sigmoid(pre_silu)
    grad = grad * sigmoid * (1 + pre_silu * (1 - sigmoid))
    return tl.where(valid, grad, 0)


@triton.jit(do_not_specialize=["T"])
def _gdn_short_conv_backward_kernel(
    QKV,
    Weight,
    GradQ,
    GradK,
    GradV,
    CuSeqLens,
    ChunkIndices,
    Output,
    PartialWeight,
    T,
    QKV_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_V_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    TAPS: tl.constexpr,
    BT: tl.constexpr,
    HALO: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
    EPS: tl.constexpr,
    VARLEN: tl.constexpr,
    MATERIALIZE_GRADIENT: tl.constexpr,
):
    segment, chunk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if VARLEN:
        sequence = tl.load(ChunkIndices + chunk * 2)
        local_chunk = tl.load(ChunkIndices + chunk * 2 + 1).to(tl.int64)
        start = tl.load(CuSeqLens + sequence).to(tl.int64)
        length = tl.load(CuSeqLens + sequence + 1).to(tl.int64) - start
        partial = chunk.to(tl.int64)
    else:
        local_chunk = chunk.to(tl.int64)
        start = batch.to(tl.int64) * T
        length = T
        partial = batch.to(tl.int64) * tl.num_programs(1) + chunk

    VALUE_GROUPS: tl.constexpr = NUM_V_HEADS // NUM_HEADS
    GROUP_SEGMENTS: tl.constexpr = 2 + VALUE_GROUPS
    GROUP_WIDTH: tl.constexpr = 2 * HEAD_DIM + VALUE_GROUPS * HEAD_V_DIM
    group = segment // GROUP_SEGMENTS
    role = segment % GROUP_SEGMENTS
    is_value = role >= 2
    width = tl.where(is_value, HEAD_V_DIM, HEAD_DIM)
    grad_heads = tl.where(is_value, NUM_V_HEADS, NUM_HEADS)
    head = tl.where(is_value, group * VALUE_GROUPS + role - 2, group)
    segment_start = tl.where(
        is_value, 2 * HEAD_DIM + (role - 2) * HEAD_V_DIM, role * HEAD_DIM
    )
    dims = tl.arange(0, BLOCK_DIM)
    columns = group * GROUP_WIDTH + segment_start + dims
    grad = GradQ if role == 0 else (GradK if role == 1 else GradV)

    local_rows = tl.arange(0, BT)
    rows = local_chunk * BT + local_rows
    dy = _short_conv_gradient_tile(
        QKV,
        Weight,
        grad,
        start,
        rows,
        length,
        columns,
        dims,
        width,
        head,
        grad_heads,
        is_value,
        QKV_DIM,
        TAPS,
        GradQ.dtype.element_ty,
        NORMALIZE_QK,
        EPS,
    )
    offsets = (start + rows[:, None]) * QKV_DIM + columns[None, :]
    valid = (rows[:, None] < length) & (dims[None, :] < width)
    if MATERIALIZE_GRADIENT:
        tl.store(Output + offsets, dy, valid)
    else:
        if TAPS > 1:
            halo_start = (local_chunk + 1) * BT
            halo_rows = halo_start + tl.arange(0, HALO)
            halo_end = tl.minimum(length, halo_start + TAPS - 1)
            halo = _short_conv_gradient_tile(
                QKV,
                Weight,
                grad,
                start,
                halo_rows,
                halo_end,
                columns,
                dims,
                width,
                head,
                grad_heads,
                is_value,
                QKV_DIM,
                TAPS,
                GradQ.dtype.element_ty,
                NORMALIZE_QK,
                EPS,
            )
        x = tl.load(QKV + offsets, valid, 0).to(tl.float32)
        dx = tl.full((BT, BLOCK_DIM), 0, tl.float32)
        weight_offsets = columns.to(tl.int64) * TAPS
        for shift in tl.static_range(TAPS):
            if shift == 0:
                shifted = dy
            else:
                # Wrap both gathers into bounds. Only the selected tile contributes.
                source = local_rows + shift
                in_tile = tl.broadcast_to((source % BT)[:, None], (BT, BLOCK_DIM))
                in_halo = tl.broadcast_to(
                    ((source - BT) & (HALO - 1))[:, None], (BT, BLOCK_DIM)
                )
                shifted = tl.where(
                    source[:, None] < BT,
                    tl.gather(dy, in_tile, axis=0),
                    tl.gather(halo, in_halo, axis=0),
                )
            tap = TAPS - 1 - shift
            weight = tl.load(Weight + weight_offsets + tap, dims < width, 0).to(
                tl.float32
            )
            dx += shifted * weight[None, :]
            dw = tl.sum(shifted * x, axis=0)
            tl.store(
                PartialWeight + partial * QKV_DIM * TAPS + weight_offsets + tap,
                dw,
                dims < width,
            )
        tl.store(Output + offsets, dx, valid)


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
    head_dim: int,
    head_v_dim: int,
    normalize_qk: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.common.fused_conv_qkv import _prepare_varlen_metadata

    batch, seq_len, qkv_dim = qkv.shape
    taps = weight.shape[-1]
    block_dim = triton.next_power_of_2(max(head_dim, head_v_dim))
    tile_rows = 64  # Match FLA's chunk indices and weight-gradient partials.
    cu_seqlens, _, chunk_indices = _prepare_varlen_metadata(
        qkv, cu_seqlens, None, None, tile_rows
    )
    chunks = (
        len(chunk_indices)
        if chunk_indices is not None
        else triton.cdiv(seq_len, tile_rows)
    )
    x = qkv.contiguous()
    w = (weight[:, 0, :] if weight.ndim == 3 else weight).contiguous()

    # Large tap counts use FLA's accumulation on one FP32 derivative per token.
    # Recomputing a halo as wide as the main tile or wider would duplicate that work.
    wide = taps > tile_rows
    if wide:
        output = torch.empty_like(x, dtype=torch.float32)
        weight_partials = None
    else:
        output = torch.empty_like(x)
        weight_partials = w.new_empty(
            (batch * chunks, qkv_dim, taps), dtype=torch.float32
        )

    _gdn_short_conv_backward_kernel[(2 * num_heads + num_v_heads, chunks, batch)](
        x,
        w,
        grad_q.contiguous(),
        grad_k.contiguous(),
        grad_v.contiguous(),
        cu_seqlens,
        chunk_indices,
        output,
        weight_partials,
        seq_len,
        qkv_dim,
        num_heads,
        num_v_heads,
        head_dim,
        head_v_dim,
        block_dim,
        taps,
        tile_rows,
        triton.next_power_of_2(taps - 1) if 1 < taps <= tile_rows else 0,
        normalize_qk,
        eps,
        cu_seqlens is not None,
        wide,
        num_warps=4,
    )
    if wide:
        from fla.modules.conv.triton.ops import causal_conv1d_bwd

        grad_qkv, grad_weight, _, _, _ = causal_conv1d_bwd(
            x=x,
            dy=output,
            dht=None,
            weight=w,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            BT=tile_rows,
        )
    else:
        grad_qkv = output
        grad_weight = weight_partials.sum(0).to(w.dtype)
    return grad_qkv, grad_weight.view_as(weight)


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
    head_dim,
    head_v_dim,
    normalize_qk,
    eps,
):
    return torch.empty_like(
        qkv, memory_format=torch.contiguous_format
    ), torch.empty_like(weight, memory_format=torch.contiguous_format)
