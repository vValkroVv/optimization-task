"""Fused bf16 indexer scoring and radix top-k selection on Hopper.

Each CTA keeps its scores and running top-k on chip. Results are an unsorted set
of valid int32 key ids followed by -1 padding, with a length for every query.
"""

import math
import torch
import tilelang as tl
import tilelang.language as T

FP32 = T.float32
INT32 = T.int32
NEG = -3.0e38  # padding sentinel

# Head reduction crosses warp groups and needs automatic shared-memory barriers.
pass_configs = {
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tl.jit(pass_configs=pass_configs, out_idx=None)
def _fused_topk_kernel(
    heads,
    dim,
    topk,
    sm_scale,
    block_Q=2,
    block_N=256,
    dtype="bfloat16",
    num_threads=256,
    num_stages=2,
):
    assert topk == 2 ** int(round(math.log2(topk))), (
        "topk must be a power of two (2K merge buffer)"
    )
    assert topk % block_N == 0 and block_N <= topk, (
        "block_N must divide topk and be <= topk"
    )
    Sq = T.dynamic("Sq")
    Sk = T.dynamic("Sk")
    QH = block_Q * heads
    N = 2 * topk
    RADIX = 256

    def _u16(x):
        h = T.cast(x, T.float16)
        b = T.reinterpret(h, T.uint16)
        b = T.if_then_else(x < 0, ~b & 0xFFFF, b | 0x8000)
        return b >> 8

    def _u32(x):
        b = T.reinterpret(x, T.uint32)
        b = T.if_then_else(
            x < 0, ~b & T.cast(0xFFFFFFFF, T.uint32), b | T.cast(0x80000000, T.uint32)
        )
        return b

    @T.macro
    def radix_merge(
        val_s,
        idx_s,
        val_tmp,
        idx_tmp,
        s_hist,
        s_thr,
        s_ninp,
        s_iidx,
        tau,
        fill,
        bq,
        C,
        tx,
    ):
        # top-`topk` of val_s[bq,0:C] gathered into [0,topk) (value+index moved together at
        # one atomic position, so the pairing stays exact); tau becomes their min, fill resets.
        l_topk = T.alloc_var(INT32)
        l_thr = T.alloc_var(INT32)
        l_bin = T.alloc_var(INT32)
        l_val = T.alloc_var(INT32)
        l_nin = T.alloc_var(INT32)
        l_sp = T.alloc_var(INT32)
        l_pos = T.alloc_var(INT32)
        T.sync_threads()  # drain the previous bq's use of the shared radix scratch
        for i in T.Parallel(topk):
            val_tmp[i] = NEG
            idx_tmp[i] = -1
        T.fill(s_hist, 0)
        if tx == 0:
            s_ninp[0] = 0
            s_thr[0] = 0
        l_topk = topk
        T.sync_threads()
        for s in T.serial(T.ceildiv(C, num_threads)):
            ii = s * num_threads + tx
            if ii < C:
                T.atomic_add(s_hist[_u16(val_s[bq, ii])], 1)
        T.sync_threads()
        if tx < RADIX:
            for i in T.serial(8):
                off = 1 << i
                T.sync_threads(3, RADIX)
                if tx < RADIX - off:
                    l_val = s_hist[tx] + s_hist[tx + off]
                T.sync_threads(3, RADIX)
                if tx < RADIX - off:
                    s_hist[tx] = l_val
            T.sync_threads(3, RADIX)
            if s_hist[tx] >= l_topk and s_hist[tx + 1] < l_topk:
                s_thr[0] = tx
        T.sync_threads()
        l_thr = s_thr[0]
        l_topk = l_topk - s_hist[l_thr + 1]
        T.sync_threads()
        for s in T.serial(T.ceildiv(C, num_threads)):
            ii = s * num_threads + tx
            if ii < C:
                l_bin = T.cast(_u16(val_s[bq, ii]), INT32)
                if l_bin > l_thr:
                    l_pos = T.atomic_add(s_hist[l_bin + 1], 1, return_prev=True)
                    val_tmp[l_pos] = val_s[bq, ii]
                    idx_tmp[l_pos] = idx_s[bq, ii]
                elif l_bin == l_thr and l_topk > 0:
                    l_pos = T.atomic_add(s_ninp[0], 1, return_prev=True)
                    if l_pos < N:
                        s_iidx[0, l_pos] = ii
        for rnd in T.serial(4):
            if l_topk > 0:
                r = rnd % 2
                l_sp = topk - l_topk
                T.sync_threads()
                T.fill(s_hist, 0)
                if tx == 0:
                    s_ninp[r ^ 1] = 0
                T.sync_threads()
                l_nin = s_ninp[r]
                if l_nin > N:
                    l_nin = N
                for s in T.serial(T.ceildiv(l_nin, num_threads)):
                    if s * num_threads + tx < l_nin:
                        l_bin = T.cast(
                            (
                                _u32(val_s[bq, s_iidx[r, s * num_threads + tx]])
                                >> (24 - rnd * 8)
                            )
                            & 0xFF,
                            INT32,
                        )
                        T.atomic_add(s_hist[l_bin], 1)
                T.sync_threads()
                if tx < RADIX:
                    for i in T.serial(8):
                        off = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - off:
                            l_val = s_hist[tx] + s_hist[tx + off]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - off:
                            s_hist[tx] = l_val
                    T.sync_threads(3, RADIX)
                    if s_hist[tx] >= l_topk and s_hist[tx + 1] < l_topk:
                        s_thr[0] = tx
                T.sync_threads()
                l_thr = s_thr[0]
                l_topk = l_topk - s_hist[l_thr + 1]
                T.sync_threads()
                for s in T.serial(T.ceildiv(l_nin, num_threads)):
                    if s * num_threads + tx < l_nin:
                        jj = s_iidx[r, s * num_threads + tx]
                        l_bin = T.cast(
                            (_u32(val_s[bq, jj]) >> (24 - rnd * 8)) & 0xFF, INT32
                        )
                        if l_bin > l_thr:
                            l_pos = (
                                T.atomic_add(s_hist[l_bin + 1], 1, return_prev=True)
                                + l_sp
                            )
                            val_tmp[l_pos] = val_s[bq, jj]
                            idx_tmp[l_pos] = idx_s[bq, jj]
                        elif l_bin == l_thr and l_topk > 0:
                            if rnd == 3:
                                l_pos = (
                                    T.atomic_add(s_hist[l_bin + 1], 1, return_prev=True)
                                    + l_sp
                                )
                                if l_pos < topk:
                                    val_tmp[l_pos] = val_s[bq, jj]
                                    idx_tmp[l_pos] = idx_s[bq, jj]
                            else:
                                l_pos = T.atomic_add(s_ninp[r ^ 1], 1, return_prev=True)
                                if l_pos < N:
                                    s_iidx[r ^ 1, l_pos] = jj
        T.sync_threads()
        for i in T.Parallel(topk):
            val_s[bq, i] = val_tmp[i]
            idx_s[bq, i] = idx_tmp[i]
        T.sync_threads()
        mn = T.alloc_fragment([topk], FP32)
        mr = T.alloc_fragment([1], FP32)
        T.copy(val_tmp, mn)
        T.reduce_min(mn, mr, dim=0, clear=True)
        T.sync_threads()
        if tx == 0:
            tau[bq] = mr[0]
            fill[bq] = 0
        T.sync_threads()

    @T.prim_func
    def dsa_fused_select(
        Q: T.Tensor([Sq, heads, dim], dtype),
        W: T.Tensor([Sq, heads], dtype),  # bf16; upcast to fp32 on the shared load
        K: T.Tensor([Sk, dim], dtype),
        QRange: T.Tensor([Sq, 2], INT32),  # per-query [k_lo, k_hi)
        TopkIdx: T.Tensor([Sq, topk], INT32),
        TopkLen: T.Tensor([Sq], INT32),
    ):
        with T.Kernel(T.ceildiv(Sq, block_Q), threads=num_threads) as (bx):
            q0 = bx * block_Q
            tx = T.get_thread_binding()

            idx_s = T.alloc_shared([block_Q, N], dtype=INT32)
            val_s = T.alloc_shared([block_Q, N], dtype=FP32)
            val_tmp = T.alloc_shared([topk], dtype=FP32)
            idx_tmp = T.alloc_shared([topk], dtype=INT32)
            s_hist = T.alloc_shared([RADIX * 2], dtype=INT32)
            s_thr = T.alloc_shared([1], dtype=INT32)
            s_ninp = T.alloc_shared([2], dtype=INT32)
            s_iidx = T.alloc_shared([2, N], dtype=INT32)
            klo = T.alloc_shared([block_Q], INT32)
            khi = T.alloc_shared([block_Q], INT32)
            tau = T.alloc_shared(
                [block_Q], FP32
            )  # running K-th score (append threshold)
            fill = T.alloc_shared(
                [block_Q], INT32
            )  # overflow occupancy in [topk, 2*topk)
            outcnt = T.alloc_shared([block_Q], INT32)
            T.fill(val_s, NEG)
            T.fill(idx_s, -1)
            T.fill(tau, NEG)
            T.fill(fill, 0)
            T.sync_threads()

            # union scan range over the block_Q queries; each query masks to its own [klo,khi)
            u_lo = T.alloc_var(INT32)
            u_hi = T.alloc_var(INT32)
            u_lo = 2147483647
            u_hi = -2147483648
            for bq in T.serial(block_Q):
                qi = T.min(q0 + bq, Sq - 1)
                klo[bq] = QRange[qi, 0]
                khi[bq] = QRange[qi, 1]
                u_lo = T.min(u_lo, QRange[qi, 0])
                u_hi = T.max(u_hi, QRange[qi, 1])
            T.sync_threads()

            q_s = T.alloc_shared([QH, dim], dtype=dtype)
            w_s = T.alloc_shared([block_Q, heads], FP32)
            for i, j in T.Parallel(QH, dim):
                bq = i // heads
                qi = T.min(q0 + bq, Sq - 1)
                q_s[i, j] = Q[qi, i % heads, j] * sm_scale
            for bq, h in T.Parallel(block_Q, heads):
                qi = T.min(q0 + bq, Sq - 1)
                w_s[bq, h] = W[qi, h]
            T.sync_threads()

            span = u_hi - u_lo
            for tk in T.Pipelined(T.ceildiv(span, block_N), num_stages=num_stages):
                rel = tk * block_N
                # pre-emptive merge before the overflow region can spill this tile
                for bq in T.serial(block_Q):
                    if fill[bq] + block_N > topk:
                        radix_merge(
                            val_s,
                            idx_s,
                            val_tmp,
                            idx_tmp,
                            s_hist,
                            s_thr,
                            s_ninp,
                            s_iidx,
                            tau,
                            fill,
                            bq,
                            topk + fill[bq],
                            tx,
                        )

                k_s = T.alloc_shared([block_N, dim], dtype=dtype)
                for i, j in T.Parallel(block_N, dim):
                    k_s[i, j] = T.if_then_else(rel + i < span, K[u_lo + rel + i, j], 0)

                s = T.alloc_fragment([block_N, QH], FP32)
                T.gemm(
                    k_s,
                    q_s,
                    s,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                s_r = T.reshape(s, (block_N, block_Q, heads))
                for n, bq, h in T.Parallel(block_N, block_Q, heads):
                    s_r[n, bq, h] = T.max(s_r[n, bq, h], 0) * w_s[bq, h]
                logits = T.alloc_shared([block_N, block_Q], FP32)
                T.reduce_sum(s_r, logits, dim=-1, clear=True)
                T.sync_threads()

                # append only keys above the running threshold into the overflow region
                for bq, n in T.Parallel(block_Q, block_N):
                    key = u_lo + rel + n
                    if key >= klo[bq] and key < khi[bq] and logits[n, bq] > tau[bq]:
                        cpos = T.alloc_var(INT32)
                        cpos = T.atomic_add(fill[bq], 1, return_prev=True)
                        if cpos < topk:
                            val_s[bq, topk + cpos] = logits[n, bq]
                            idx_s[bq, topk + cpos] = key
                T.sync_threads()

            for bq in T.serial(block_Q):
                radix_merge(
                    val_s,
                    idx_s,
                    val_tmp,
                    idx_tmp,
                    s_hist,
                    s_thr,
                    s_ninp,
                    s_iidx,
                    tau,
                    fill,
                    bq,
                    topk + fill[bq],
                    tx,
                )

            # The radix gather puts real entries before NEG in its lowest bucket. The -1
            # padding is a contiguous tail and the valid count is just the non-NEG count
            T.fill(outcnt, 0)
            T.sync_threads()
            for bq, i in T.Parallel(block_Q, topk):
                if q0 + bq < Sq:
                    oi = q0 + bq
                    if val_s[bq, i] > NEG:
                        T.atomic_add(outcnt[bq], 1)
                        TopkIdx[oi, i] = idx_s[bq, i]
                    else:
                        TopkIdx[oi, i] = -1
            T.sync_threads()
            if tx == 0:
                for bq in T.serial(block_Q):
                    if q0 + bq < Sq:
                        TopkLen[q0 + bq] = outcnt[bq]

    return dsa_fused_select


@torch.no_grad()
def fused_select_topk(
    index_q,
    index_k,
    weights,
    scale,
    K,
    positions_q,
    cu_seqlens=None,
    q_max=None,
    q_min=None,
):
    """Return ids [1, Sq, K] and valid lengths [Sq], without materializing scores.

    q_min/q_max are trusted host bounds supplied by the layer, avoiding a host
    synchronization. Query positions may be noncontiguous (SP rebalance).
    """
    assert type(K) is int and 128 <= K <= 4096 and (K & (K - 1)) == 0, (
        f"topk K must be a power of two from 128 through 4096, got {K}"
    )
    assert isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0, (
        f"scale must be a finite positive host number, got {scale!r}"
    )
    # K=4096 needs a smaller CTA to fit the radix scratch in Hopper shared memory.
    block_Q, block_N, num_stages = (1, 128, 1) if K == 4096 else (2, min(256, K), 2)

    assert (
        index_q.dim() == 4 and index_q.shape[0] == 1 and index_q.dtype == torch.bfloat16
    ), "index_q must be [1, Sq, H, D] bf16"
    assert (
        index_k.dim() == 4
        and index_k.shape[0] == 1
        and index_k.shape[2] == 1
        and index_k.dtype == torch.bfloat16
    ), "index_k must be [1, Sk, 1, D] bf16"
    assert (
        weights.dim() == 3 and weights.shape[0] == 1 and weights.dtype == torch.bfloat16
    ), "weights must be [1, Sq, H] bf16"
    q = index_q[0]  # [Sq, H, D]
    k = index_k[0, :, 0]  # [Sk, D]
    w = weights[0]
    Sq, H, D = q.shape
    Sk = k.shape[0]
    assert Sq > 0 and Sk > 0, f"empty inputs: Sq={Sq}, Sk={Sk}"
    assert H == 64 and D == 128, f"indexer requires H=64, D=128, got H={H}, D={D}"
    assert index_k.shape[3] == D, "index_q/index_k head_dim mismatch"
    assert weights.shape[1] == Sq and positions_q.shape[0] == Sq, (
        f"Sq mismatch: q={Sq}, weights={weights.shape[1]}, positions={positions_q.shape[0]}"
    )
    assert positions_q.dim() == 1 and positions_q.dtype in (torch.int32, torch.int64), (
        f"positions_q must be 1-D int32/int64, got {positions_q.dim()}-D {positions_q.dtype}"
    )

    assert isinstance(q_max, int) and isinstance(q_min, int), (
        "q_max/q_min are required host Python ints (the max/min query position); got "
        f"{type(q_max).__name__}/{type(q_min).__name__}"
    )
    assert 0 <= q_min <= q_max < Sk, (
        f"positions out of range: q_min={q_min}, q_max={q_max}, Sk={Sk} (would read keys OOB)"
    )
    if cu_seqlens is not None:
        assert (
            cu_seqlens.dim() == 1
            and cu_seqlens.numel() >= 2
            and cu_seqlens.dtype in (torch.int32, torch.int64)
        ), (
            f"cu_seqlens must be 1-D int32/int64 with >= 2 boundaries, got {tuple(cu_seqlens.shape)}"
        )
        # Share the host boundary cache with dsa_operator._cu_bounds.
        bounds = getattr(cu_seqlens, "_dsa_bounds", None)
        if bounds is None:
            bounds = cu_seqlens.tolist()
            cu_seqlens._dsa_bounds = bounds
        assert (
            bounds[0] == 0
            and bounds[-1] == Sk
            and all(a < b for a, b in zip(bounds, bounds[1:]))
        ), (
            f"cu_seqlens must increase strictly from 0 to Sk={Sk}, got {bounds[:3]}..{bounds[-2:]}"
        )

    # per-query scan ranges [doc start, position + 1)
    dev = q.device
    pos = positions_q.to(device=dev, dtype=torch.int64)
    if cu_seqlens is None:
        k_lo = torch.zeros(Sq, device=dev, dtype=torch.int64)
    else:
        inner = cu_seqlens[1:-1].to(device=dev, dtype=torch.int64)
        k_lo = cu_seqlens.to(device=dev, dtype=torch.int64)[
            torch.bucketize(pos, inner, right=True)
        ]
    qrange = torch.stack([k_lo, pos + 1], dim=1).to(torch.int32).contiguous()

    topk_idx = torch.empty(Sq, K, device=dev, dtype=torch.int32)
    topk_len = torch.empty(Sq, device=dev, dtype=torch.int32)
    kern = _fused_topk_kernel(
        H,
        D,
        K,
        float(scale),
        block_Q=block_Q,
        block_N=block_N,
        dtype="bfloat16",
        num_threads=256,
        num_stages=num_stages,
    )
    kern(q.contiguous(), w.contiguous(), k.contiguous(), qrange, topk_idx, topk_len)
    return topk_idx.unsqueeze(0), topk_len
