from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _qkv_conv1d_unpack_kernel(
    x,
    weight,
    initial_state,
    q,
    k,
    v,
    rstd_q,
    rstd_k,
    B: tl.constexpr,
    T,
    BT_: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    HD: tl.constexpr,
    HVD: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    GROUP_WIDTH: tl.constexpr,
    VALUE_GROUPS: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BT: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
    SAVE_RSTD: tl.constexpr,
    EPS: tl.constexpr,
):
    hs = tl.program_id(0)
    t_block = tl.program_id(1)
    b = tl.program_id(2)

    h = hs // NUM_SEGMENTS
    segment = hs - h * NUM_SEGMENTS

    is_q = segment == 0
    is_k = segment == 1
    is_v = segment >= 2

    seg_width = tl.where(is_v, HVD, HD)
    v_local = segment - 2
    d_base = h * GROUP_WIDTH + tl.where(
        is_q,
        0,
        tl.where(is_k, HD, 2 * HD + v_local * HVD),
    )

    offs_t = t_block * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < seg_width
    packed_d = d_base + offs_d

    acc = tl.zeros((BT, BLOCK_D), dtype=tl.float32)
    for i_w in tl.static_range(-W + 1, 1):
        src_t = offs_t + i_w
        w = tl.load(
            weight + packed_d * W + (i_w + W - 1),
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        x_val = tl.load(
            x + b * T * D + src_t[:, None] * D + packed_d[None, :],
            mask=(mask_t[:, None] & (src_t[:, None] >= 0) & mask_d[None, :]),
            other=0.0,
        ).to(tl.float32)

        if HAS_INITIAL_STATE and t_block == 0:
            state_val = tl.load(
                initial_state + b * D * W + packed_d[None, :] * W + (src_t + W)[:, None],
                mask=((src_t[:, None] < 0) & ((src_t + W)[:, None] >= 0) & mask_d[None, :]),
                other=0.0,
            ).to(tl.float32)
            x_val += state_val

        acc += x_val * w[None, :]

    acc = acc * tl.sigmoid(acc)

    if NORMALIZE_QK:
        acc_for_norm = acc.to(tl.bfloat16).to(tl.float32)
        sq = tl.sum(tl.where(mask_d[None, :], acc_for_norm * acc_for_norm, 0.0), axis=1)
        rstd = 1 / tl.sqrt(sq + EPS)
        acc = tl.where(is_v, acc, acc_for_norm * rstd[:, None])
        if SAVE_RSTD:
            if is_q:
                tl.store(rstd_q + (b * T + offs_t) * H + h, rstd, mask=mask_t)
            elif is_k:
                tl.store(rstd_k + (b * T + offs_t) * H + h, rstd, mask=mask_t)

    if is_q:
        out = q + ((b * T + offs_t[:, None]) * H + h) * HD + offs_d[None, :]
        tl.store(out, acc, mask=mask_t[:, None] & mask_d[None, :])
    elif is_k:
        out = k + ((b * T + offs_t[:, None]) * H + h) * HD + offs_d[None, :]
        tl.store(out, acc, mask=mask_t[:, None] & mask_d[None, :])
    else:
        vh = h * VALUE_GROUPS + v_local
        out = v + ((b * T + offs_t[:, None]) * HV + vh) * HVD + offs_d[None, :]
        tl.store(out, acc, mask=mask_t[:, None] & mask_d[None, :])


@triton.jit(do_not_specialize=["T"])
def _qkv_conv1d_unpack_varlen_kernel(
    x,
    weight,
    initial_state,
    cu_seqlens,
    chunk_indices,
    q,
    k,
    v,
    rstd_q,
    rstd_k,
    B: tl.constexpr,
    T,
    BT_: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    HD: tl.constexpr,
    HVD: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    GROUP_WIDTH: tl.constexpr,
    VALUE_GROUPS: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BT: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
    SAVE_RSTD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    EPS: tl.constexpr,
):
    hs = tl.program_id(0)
    t_block = tl.program_id(1)
    b = tl.program_id(2)

    h = hs // NUM_SEGMENTS
    segment = hs - h * NUM_SEGMENTS

    is_q = segment == 0
    is_k = segment == 1
    is_v = segment >= 2

    seg_width = tl.where(is_v, HVD, HD)
    v_local = segment - 2
    d_base = h * GROUP_WIDTH + tl.where(
        is_q,
        0,
        tl.where(is_k, HD, 2 * HD + v_local * HVD),
    )

    if IS_VARLEN:
        n = tl.load(chunk_indices + t_block * 2).to(tl.int32)
        t_block = tl.load(chunk_indices + t_block * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + n).to(tl.int64)
        eos = tl.load(cu_seqlens + n + 1).to(tl.int64)
        seq_len = eos - bos
    else:
        n = b
        bos = (b * T).to(tl.int64)
        seq_len = T

    offs_t = t_block * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, BLOCK_D)
    mask_t = offs_t < seq_len
    mask_d = offs_d < seg_width
    packed_d = d_base + offs_d

    acc = tl.zeros((BT, BLOCK_D), dtype=tl.float32)
    for i_w in tl.static_range(-W + 1, 1):
        src_t = offs_t + i_w
        src_abs_t = bos + src_t
        w = tl.load(
            weight + packed_d * W + (i_w + W - 1),
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        x_val = tl.load(
            x + src_abs_t[:, None] * D + packed_d[None, :],
            mask=(
                mask_t[:, None]
                & (src_t[:, None] >= 0)
                & (src_t[:, None] < seq_len)
                & mask_d[None, :]
            ),
            other=0.0,
        ).to(tl.float32)

        if HAS_INITIAL_STATE and t_block == 0:
            state_val = tl.load(
                initial_state + n * D * W + packed_d[None, :] * W + (src_t + W)[:, None],
                mask=((src_t[:, None] < 0) & ((src_t + W)[:, None] >= 0) & mask_d[None, :]),
                other=0.0,
            ).to(tl.float32)
            x_val += state_val

        acc += x_val * w[None, :]

    acc = acc * tl.sigmoid(acc)

    if NORMALIZE_QK:
        acc_for_norm = acc.to(tl.bfloat16).to(tl.float32)
        sq = tl.sum(tl.where(mask_d[None, :], acc_for_norm * acc_for_norm, 0.0), axis=1)
        rstd = 1 / tl.sqrt(sq + EPS)
        acc = tl.where(is_v, acc, acc_for_norm * rstd[:, None])
        if SAVE_RSTD:
            if is_q:
                tl.store(rstd_q + (bos + offs_t) * H + h, rstd, mask=mask_t)
            elif is_k:
                tl.store(rstd_k + (bos + offs_t) * H + h, rstd, mask=mask_t)

    if is_q:
        out = q + ((bos + offs_t[:, None]) * H + h) * HD + offs_d[None, :]
        tl.store(out, acc, mask=mask_t[:, None] & mask_d[None, :])
    elif is_k:
        out = k + ((bos + offs_t[:, None]) * H + h) * HD + offs_d[None, :]
        tl.store(out, acc, mask=mask_t[:, None] & mask_d[None, :])
    else:
        vh = h * VALUE_GROUPS + v_local
        out = v + ((bos + offs_t[:, None]) * HV + vh) * HVD + offs_d[None, :]
        tl.store(out, acc, mask=mask_t[:, None] & mask_d[None, :])


@triton.jit(do_not_specialize=["T"])
def _qkv_bwd_pack_kernel(
    dq,
    dk,
    dv,
    q,
    k,
    rstd_q,
    rstd_k,
    dy,
    B: tl.constexpr,
    T,
    BT_: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    HD: tl.constexpr,
    HVD: tl.constexpr,
    D: tl.constexpr,
    GROUP_WIDTH: tl.constexpr,
    VALUE_GROUPS: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BT: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
):
    hs = tl.program_id(0)
    t_block = tl.program_id(1)
    b = tl.program_id(2)

    h = hs // NUM_SEGMENTS
    segment = hs - h * NUM_SEGMENTS

    is_q = segment == 0
    is_k = segment == 1
    is_v = segment >= 2

    seg_width = tl.where(is_v, HVD, HD)
    v_local = segment - 2
    d_base = h * GROUP_WIDTH + tl.where(
        is_q,
        0,
        tl.where(is_k, HD, 2 * HD + v_local * HVD),
    )

    offs_t = t_block * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < seg_width
    packed_d = d_base + offs_d

    if is_q:
        grad = tl.load(
            dq + ((b * T + offs_t[:, None]) * H + h) * HD + offs_d[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        if NORMALIZE_QK:
            norm = tl.load(
                q + ((b * T + offs_t[:, None]) * H + h) * HD + offs_d[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            rstd = tl.load(
                rstd_q + (b * T + offs_t) * H + h,
                mask=mask_t,
                other=0.0,
            ).to(tl.float32)
            dot = tl.sum(tl.where(mask_d[None, :], grad * norm, 0.0), axis=1)
            grad = ((grad - norm * dot[:, None]) * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    elif is_k:
        grad = tl.load(
            dk + ((b * T + offs_t[:, None]) * H + h) * HD + offs_d[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        if NORMALIZE_QK:
            norm = tl.load(
                k + ((b * T + offs_t[:, None]) * H + h) * HD + offs_d[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            rstd = tl.load(
                rstd_k + (b * T + offs_t) * H + h,
                mask=mask_t,
                other=0.0,
            ).to(tl.float32)
            dot = tl.sum(tl.where(mask_d[None, :], grad * norm, 0.0), axis=1)
            grad = ((grad - norm * dot[:, None]) * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    else:
        vh = h * VALUE_GROUPS + v_local
        grad = tl.load(
            dv + ((b * T + offs_t[:, None]) * HV + vh) * HVD + offs_d[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

    tl.store(
        dy + b * T * D + offs_t[:, None] * D + packed_d[None, :],
        grad,
        mask=mask_t[:, None] & mask_d[None, :],
    )



def _prepare_varlen_metadata(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    bt: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if cu_seqlens is None:
        if chunk_indices is not None:
            raise ValueError("chunk_indices requires cu_seqlens")
        return None, cu_seqlens_cpu, None

    if x.shape[0] != 1:
        raise ValueError(
            f"expected x batch size 1 when cu_seqlens is provided, got {x.shape[0]}"
        )
    if cu_seqlens.device != x.device:
        cu_seqlens = cu_seqlens.to(device=x.device)
    if not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()
    if cu_seqlens_cpu is not None and cu_seqlens_cpu.device.type != "cpu":
        cu_seqlens_cpu = cu_seqlens_cpu.cpu()

    if chunk_indices is None:
        from fla.ops.utils import prepare_chunk_indices

        chunk_indices = prepare_chunk_indices(
            cu_seqlens,
            bt,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
    else:
        if chunk_indices.device != x.device or chunk_indices.dtype != cu_seqlens.dtype:
            chunk_indices = chunk_indices.to(device=x.device, dtype=cu_seqlens.dtype)
        if not chunk_indices.is_contiguous():
            chunk_indices = chunk_indices.contiguous()

    return cu_seqlens, cu_seqlens_cpu, chunk_indices


def _fused_qkv_conv1d_unpack_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_heads: int,
    num_v_heads: int,
    head_dim: int,
    head_v_dim: int,
    initial_state: torch.Tensor | None = None,
    cp_context=None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    normalize_qk: bool = False,
    eps: float = 1e-6,
    bt: int = 64,
    rstd_q: torch.Tensor | None = None,
    rstd_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 3:
        raise ValueError(f"expected x with shape [B, T, D], got {tuple(x.shape)}")
    if weight.dim() == 3:
        weight = weight[:, 0, :]
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    use_varlen = cu_seqlens is not None
    if cp_context is not None:
        if initial_state is not None:
            raise ValueError("pass either initial_state or cp_context, not both")
        from fla.modules.conv.cp.ops import CausalConv1dFunctionCP

        initial_state = CausalConv1dFunctionCP._prepare_initial_state_for_cp(
            x=x,
            weight=weight,
            cu_seqlens=cp_context.cu_seqlens,
            context=cp_context,
            group=cp_context.group,
        )
        if use_varlen:
            cu_seqlens = cp_context.cu_seqlens
            cu_seqlens_cpu = cp_context.cu_seqlens_cpu

    bsz, seqlen, dim = x.shape
    cu_seqlens, cu_seqlens_cpu, chunk_indices = _prepare_varlen_metadata(
        x=x,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        bt=bt,
    )
    if num_v_heads % num_heads != 0:
        raise ValueError("num_v_heads must be divisible by num_heads")
    value_groups = num_v_heads // num_heads
    group_width = 2 * head_dim + value_groups * head_v_dim
    expected_dim = num_heads * group_width
    if dim != expected_dim:
        raise ValueError(f"expected qkv dim {expected_dim}, got {dim}")

    q = torch.empty((bsz, seqlen, num_heads, head_dim), device=x.device, dtype=x.dtype)
    k = torch.empty_like(q)
    v = torch.empty((bsz, seqlen, num_v_heads, head_v_dim), device=x.device, dtype=x.dtype)
    save_rstd = rstd_q is not None and rstd_k is not None

    block_d = triton.next_power_of_2(max(head_dim, head_v_dim))
    if block_d > 256:
        raise ValueError(f"unsupported head dimension block {block_d}")
    num_segments = 2 + value_groups
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = x

    if cu_seqlens is None:
        grid = (num_heads * num_segments, triton.cdiv(seqlen, bt), bsz)
        _qkv_conv1d_unpack_kernel[grid](
            x,
            weight,
            initial_state,
            q,
            k,
            v,
            rstd_q if rstd_q is not None else q,
            rstd_k if rstd_k is not None else k,
            bsz,
            seqlen,
            triton.cdiv(seqlen, 8192),
            num_heads,
            num_v_heads,
            head_dim,
            head_v_dim,
            weight.shape[-1],
            dim,
            group_width,
            value_groups,
            num_segments,
            block_d,
            bt,
            has_initial_state,
            normalize_qk,
            save_rstd,
            eps,
        )
    else:
        grid = (num_heads * num_segments, len(chunk_indices), bsz)
        _qkv_conv1d_unpack_varlen_kernel[grid](
            x,
            weight,
            initial_state,
            cu_seqlens,
            chunk_indices,
            q,
            k,
            v,
            rstd_q if rstd_q is not None else q,
            rstd_k if rstd_k is not None else k,
            bsz,
            seqlen,
            triton.cdiv(seqlen, 8192),
            num_heads,
            num_v_heads,
            head_dim,
            head_v_dim,
            weight.shape[-1],
            dim,
            group_width,
            value_groups,
            num_segments,
            block_d,
            bt,
            has_initial_state,
            normalize_qk,
            save_rstd,
            True,
            eps,
        )
    return q, k, v


def _pack_qkv_backward_grad(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    rstd_q: torch.Tensor,
    rstd_k: torch.Tensor,
    *,
    num_heads: int,
    num_v_heads: int,
    head_dim: int,
    head_v_dim: int,
    normalize_qk: bool,
    bt: int,
) -> torch.Tensor:
    bsz, seqlen = dq.shape[:2]
    value_groups = num_v_heads // num_heads
    group_width = 2 * head_dim + value_groups * head_v_dim
    dim = num_heads * group_width

    dy = torch.empty((bsz, seqlen, dim), device=dq.device, dtype=dq.dtype)
    block_d = triton.next_power_of_2(max(head_dim, head_v_dim))
    if block_d > 256:
        raise ValueError(f"unsupported head dimension block {block_d}")
    num_segments = 2 + value_groups

    grid = (num_heads * num_segments, triton.cdiv(seqlen, bt), bsz)
    _qkv_bwd_pack_kernel[grid](
        dq,
        dk,
        dv,
        q,
        k,
        rstd_q,
        rstd_k,
        dy,
        bsz,
        seqlen,
        triton.cdiv(seqlen, 8192),
        num_heads,
        num_v_heads,
        head_dim,
        head_v_dim,
        dim,
        group_width,
        value_groups,
        num_segments,
        block_d,
        bt,
        normalize_qk,
    )
    return dy


def _fused_qkv_conv1d_unpack_backward_via_fla(
    x: torch.Tensor,
    weight: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    rstd_q: torch.Tensor,
    rstd_k: torch.Tensor,
    *,
    num_heads: int,
    num_v_heads: int,
    head_dim: int,
    head_v_dim: int,
    initial_state: torch.Tensor | None,
    cp_context,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    normalize_qk: bool,
    bt: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    dy = _pack_qkv_backward_grad(
        dq=dq,
        dk=dk,
        dv=dv,
        q=q,
        k=k,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
        head_v_dim=head_v_dim,
        normalize_qk=normalize_qk,
        bt=bt,
    )

    from fla.modules.conv.triton.ops import causal_conv1d_bwd

    if cp_context is not None and cu_seqlens is not None:
        cu_seqlens = cp_context.cu_seqlens
        cu_seqlens_cpu = cp_context.cu_seqlens_cpu
    dx, dw, _, _, dh0 = causal_conv1d_bwd(
        x=x,
        dy=dy,
        dht=None,
        weight=weight,
        bias=None,
        residual=None,
        initial_state=initial_state,
        activation="silu",
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        BT=bt,
    )
    return dx, dw, dh0


class _FusedQKVConv1dUnpackFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        initial_state: torch.Tensor | None,
        num_heads: int,
        num_v_heads: int,
        head_dim: int,
        head_v_dim: int,
        cp_context,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        chunk_indices: torch.Tensor | None,
        normalize_qk: bool,
        eps: float,
        bt: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"expected x with shape [B, T, D], got {tuple(x.shape)}")
        weight_was_3d = weight.dim() == 3
        if weight_was_3d:
            weight_2d = weight[:, 0, :]
        else:
            weight_2d = weight
        if not x.is_contiguous():
            x = x.contiguous()
        if not weight_2d.is_contiguous():
            weight_2d = weight_2d.contiguous()

        had_initial_state_input = initial_state is not None
        use_varlen = cu_seqlens is not None
        if cp_context is not None:
            if initial_state is not None:
                raise ValueError("pass either initial_state or cp_context, not both")
            from fla.modules.conv.cp.ops import CausalConv1dFunctionCP

            initial_state = CausalConv1dFunctionCP._prepare_initial_state_for_cp(
                x=x,
                weight=weight_2d,
                cu_seqlens=cp_context.cu_seqlens,
                context=cp_context,
                group=cp_context.group,
            )
            if use_varlen:
                cu_seqlens = cp_context.cu_seqlens
                cu_seqlens_cpu = cp_context.cu_seqlens_cpu

        cu_seqlens, cu_seqlens_cpu, chunk_indices = _prepare_varlen_metadata(
            x=x,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            bt=bt,
        )

        bsz, seqlen = x.shape[:2]
        if normalize_qk:
            rstd_q = torch.empty((bsz, seqlen, num_heads), device=x.device, dtype=torch.float32)
            rstd_k = torch.empty_like(rstd_q)
        else:
            rstd_q = torch.empty(0, device=x.device, dtype=torch.float32)
            rstd_k = torch.empty(0, device=x.device, dtype=torch.float32)

        q, k, v = _fused_qkv_conv1d_unpack_forward(
            x=x,
            weight=weight_2d,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            head_dim=head_dim,
            head_v_dim=head_v_dim,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            normalize_qk=normalize_qk,
            eps=eps,
            bt=bt,
            rstd_q=rstd_q if normalize_qk else None,
            rstd_k=rstd_k if normalize_qk else None,
        )

        initial_state_for_save = (
            initial_state
            if initial_state is not None
            else torch.empty(0, device=x.device, dtype=x.dtype)
        )
        ctx.save_for_backward(x, weight_2d, initial_state_for_save, q, k, rstd_q, rstd_k)
        ctx.has_initial_state = initial_state is not None
        ctx.had_initial_state_input = had_initial_state_input
        ctx.weight_was_3d = weight_was_3d
        ctx.num_heads = num_heads
        ctx.num_v_heads = num_v_heads
        ctx.head_dim = head_dim
        ctx.head_v_dim = head_v_dim
        ctx.normalize_qk = normalize_qk
        ctx.eps = eps
        ctx.bt = bt
        ctx.cp_context = cp_context.copy_for_backward() if cp_context is not None else None
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.chunk_indices = chunk_indices
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        x, weight, initial_state_saved, q, k, rstd_q, rstd_k = ctx.saved_tensors
        initial_state = initial_state_saved if ctx.has_initial_state else None
        bsz, seqlen = x.shape[:2]

        if dq is None:
            dq = x.new_zeros((bsz, seqlen, ctx.num_heads, ctx.head_dim))
        else:
            dq = dq.contiguous()
        if dk is None:
            dk = x.new_zeros((bsz, seqlen, ctx.num_heads, ctx.head_dim))
        else:
            dk = dk.contiguous()
        if dv is None:
            dv = x.new_zeros((bsz, seqlen, ctx.num_v_heads, ctx.head_v_dim))
        else:
            dv = dv.contiguous()

        dx, dw, dh0 = _fused_qkv_conv1d_unpack_backward_via_fla(
            x=x,
            weight=weight,
            dq=dq,
            dk=dk,
            dv=dv,
            q=q,
            k=k,
            rstd_q=rstd_q,
            rstd_k=rstd_k,
            num_heads=ctx.num_heads,
            num_v_heads=ctx.num_v_heads,
            head_dim=ctx.head_dim,
            head_v_dim=ctx.head_v_dim,
            initial_state=initial_state,
            cp_context=ctx.cp_context,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
            chunk_indices=ctx.chunk_indices,
            normalize_qk=ctx.normalize_qk,
            bt=ctx.bt,
        )

        cp_context = ctx.cp_context
        if cp_context is not None:
            from fla.modules.conv.cp.ops import CausalConv1dFunctionCP

            CausalConv1dFunctionCP._correct_dx_for_cp(
                dx=dx,
                dh0=dh0,
                W=weight.shape[-1],
                group=cp_context.group,
                is_first_rank=cp_context.is_first_rank,
                pre_num_conv_tokens=cp_context.pre_num_conv_tokens,
            )

        if ctx.weight_was_3d:
            dw = dw[:, None, :]
        dinitial_state = dh0 if ctx.had_initial_state_input else None
        return (
            dx,
            dw,
            dinitial_state,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_qkv_conv1d_unpack(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_heads: int,
    num_v_heads: int,
    head_dim: int,
    head_v_dim: int,
    initial_state: torch.Tensor | None = None,
    cp_context=None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    normalize_qk: bool = False,
    eps: float = 1e-6,
    bt: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _FusedQKVConv1dUnpackFn.apply(
        x,
        weight,
        initial_state,
        num_heads,
        num_v_heads,
        head_dim,
        head_v_dim,
        cp_context,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
        normalize_qk,
        eps,
        bt,
    )
