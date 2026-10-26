"""GatedNorm (weight applied as w) on CuTe DSL for Hopper (SM90): forward kernels.

    rho = rsqrt(mean(x^2) + eps)                        per row
    P   = rho * (x @ W_eff^T),  W_eff = W_up * w       (rho factored out of the GEMM)
    S   = silu(P)
    y   = lgw * rho * x * w * sigmoid(S @ W_down^T)

Forward = three kernels on one stream, F1 -> F2 chained with PDL:
  prep  W_eff = bf16(W_up * w) (one rounding of the fp32 product), w as fp32, W_down as
        bf16, plus the bf16 transposes W_down^T / W_up^T the backward uses.
  F1    split-K statistics: for each 64-row block and H split s,
        p_part[s] = x_s @ W_eff_s^T (wgmma m64n16) and ss_part[s] = sum(x_s^2).
  F2    per 64-row block: reduce the partials -> rho, P, S (S kept in smem as the wgmma A
        operand), then per 64-column sub-tile L = S @ W_down^T (wgmma m64n64k16) and
        y = lgw * rho * x * w * sigmoid(L), written through smem with TMA stores.
        The CTA owning H chunk 0 also writes rho (M,) and P (M, 16) for the backward.

HBM traffic 6 B/elem (x read twice, y written once). Saved for the backward: x, rho, P.

Kernel structure (F1, F2): persistent grid of num_SMs CTAs; warpgroup 0 = one TMA producer
warp (64 x 64 x-tiles + weight tiles through an mbarrier ring), warpgroups 1-2 = consumers
taking sub-tiles round-robin; x is read from smem with ldmatrix directly in the
accumulator layout; outputs go registers -> stmatrix -> swizzled smem -> TMA store,
double-buffered per warpgroup.

Numerics: all reductions and intermediates in fp32; bf16 only for the input x, the tensor
core operands (W_eff, S, W_down) and the output y. Deterministic (no atomics), and batch
invariant: the H split of F1 depends on H only, so a row's results do not depend on M.

Requirements: Hopper GPU, nvidia-cutlass-dsl 4.2.1. x bf16 (..., H) with H % 128 == 0,
gate rank 16. Kernels are compiled once per (H, config) with a symbolic M.
Backward: backward.py. Autograd entry point (gated_norm, GatedNorm): op.py.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch

from .common import (
    BF16, BLK_H, BLK_M, F32, PLANS, RANK, ROW, WG, compile_op, fake, make_mma,
    make_r2s_acc, make_ring, make_s2r_acc, make_smem_layout_a, make_smem_layout_b,
    make_smem_layout_epi, resolve_num_sms, sigmoid, sym_m, tma_load_atom, tma_part,
    tma_store_atom, to_cute_dtype, workspace,
)


# =============================================================================
# prep
# =============================================================================
class GnPrep:
    """W_eff = bf16(W_up * w), w fp32, W_down bf16, W_down^T (R, H) and W_up^T (H, R) bf16."""

    def __init__(self, hidden, rank):
        self.H, self.R = hidden, rank

    @cute.jit
    def __call__(self, mGU: cute.Tensor, mW: cute.Tensor, mGD: cute.Tensor, mGUs: cute.Tensor,
                 mWf: cute.Tensor, mGDb: cute.Tensor, mGDt: cute.Tensor, mGUt: cute.Tensor,
                 stream: cuda.CUstream):
        n = self.R * self.H
        self.kernel(mGU, mW, mGD, mGUs, mWf, mGDb, mGDt, mGUt).launch(
            grid=((n + 4 * 4 * WG - 1) // (4 * 4 * WG), 1, 1), block=(4 * WG, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mGU: cute.Tensor, mW: cute.Tensor, mGD: cute.Tensor, mGUs: cute.Tensor,
               mWf: cute.Tensor, mGDb: cute.Tensor, mGDt: cute.Tensor, mGUt: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        for it in cutlass.range_constexpr(4):
            idx = (bidx * 4 + it) * (4 * WG) + tidx
            if idx < self.R * self.H:
                k = idx // self.H
                h = idx % self.H
                wv = F32(mW[h])
                gu = F32(mGU[(k, h)])
                mGUs[(k, h)] = BF16(gu * wv)
                mGUt[(h, k)] = BF16(gu)
                gd = BF16(F32(mGD[(h, k)]))
                mGDb[(h, k)] = gd
                mGDt[(k, h)] = gd
                if idx < self.H:
                    mWf[idx] = wv


# =============================================================================
# F1: split-K statistics
# =============================================================================
class GnFwdStats:
    """p_part[s, m, :] = x[m, H_s] @ W_eff[:, H_s]^T and ss_part[s, m] = sum(x[m, H_s]^2).

    Tile = 64 rows x Hs = H / n_split columns. Per 64 x 64 sub-tile: 4 wgmma k-steps into a
    64 x 16 fp32 accumulator, overlapped with sum(x^2) from the same smem stage (2 threads
    per row; summed per sub-tile first so fp32 chains stay short). The two consumer
    warpgroups combine accumulators through smem at the end of a tile.
    """

    def __init__(self, hidden, rank, n_split, num_sms, stages=8):
        hs = hidden // n_split
        if rank != RANK or hidden % n_split or hs % (2 * BLK_H):
            raise ValueError(f"H/n_split={hidden}/{n_split} must be a multiple of {2 * BLK_H}")
        self.H, self.R, self.S, self.Hs, self.n_sub = hidden, rank, n_split, hs, hs // BLK_H
        self.num_sms, self.stages = num_sms, stages
        self.n_acc = BLK_M * rank // WG                  # accumulator values per thread
        self.red_words = WG * (self.n_acc + 1)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mGUs: cute.Tensor, mPp: cute.Tensor, mSSp: cute.Tensor,
                 stream: cuda.CUstream):
        tile_mnk = (BLK_M, self.R, BLK_H)                # M, N = r, K = 64
        mma = make_mma(ROW, ROW, (BLK_M, self.R))
        a_l = make_smem_layout_a(ROW, tile_mnk, BF16, self.stages)
        b_l = make_smem_layout_b(ROW, tile_mnk, BF16, self.stages)
        tma_x, tX = tma_load_atom(mX, a_l, (BLK_M, BLK_H))
        tma_b, tB = tma_load_atom(mGUs, b_l, (self.R, BLK_H))

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            sA: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(a_l)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(b_l)], 1024]
            sRed: cute.struct.Align[cute.struct.MemRange[F32, self.red_words], 16]

        self.shared_storage = SharedStorage
        M = mX.shape[0]
        num_tiles = (M + BLK_M - 1) // BLK_M * self.S
        self.kernel(tma_x, tX, tma_b, tB, mPp, mSSp, mma, a_l, b_l, M, num_tiles).launch(
            grid=(self.num_sms, 1, 1), block=(3 * WG, 1, 1), min_blocks_per_mp=1, stream=stream,
            use_pdl=True)

    @cute.kernel
    def kernel(self, tma_x: cute.CopyAtom, mX: cute.Tensor, tma_b: cute.CopyAtom, mGUs: cute.Tensor,
               mPp: cute.Tensor, mSSp: cute.Tensor, mma: cute.TiledMma, a_l: cute.ComposedLayout,
               b_l: cute.ComposedLayout, M: cutlass.Int32, num_tiles: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // WG)
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b)

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        tx = (cute.size_in_bytes(BF16, cute.slice_(a_l, (None, None, 0)))
              + cute.size_in_bytes(BF16, cute.slice_(b_l, (None, None, 0))))
        ring = make_ring(storage.mbar, self.stages, tx)
        sA = storage.sA.get_tensor(a_l.outer, swizzle=a_l.inner)
        sB = storage.sB.get_tensor(b_l.outer, swizzle=b_l.inner)
        tAs, tAg = tma_part(tma_x, sA, cute.local_tile(mX, (BLK_M, BLK_H), (None, None)))
        tBs, tBg = tma_part(tma_b, sB, cute.local_tile(mGUs, (self.R, BLK_H), (None, None)))

        # CuTe DSL user objects must be constructed outside dynamic warp-role guards.
        w = wg_idx - 1
        wt = tidx % WG
        all_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=2 * WG)
        sRed = storage.sRed.get_tensor(cute.make_layout(self.red_words))
        thr = mma.get_slice(wt)
        tCrA = mma.make_fragment_A(thr.partition_A(sA))
        tCrB = mma.make_fragment_B(thr.partition_B(sB))
        acc = thr.make_fragment_C(thr.partition_shape_C((BLK_M, self.R)))
        tCc = thr.partition_C(cute.make_identity_tensor((BLK_M, self.R)))
        num_k = cute.size(tCrA, mode=[2])
        stride = self.n_acc + 1
        cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
        r = wt // 2
        c0 = (wt % 2) * (BLK_H // 2)
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                cute.arch.griddepcontrol_wait()          # W_eff comes from the prep kernel
                prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                tile = bidx
                while tile < num_tiles:
                    bm = tile // self.S
                    h_base = (tile % self.S) * self.n_sub
                    for j in range(self.n_sub):
                        ring.producer_acquire(prod)
                        bar = ring.producer_get_barrier(prod)
                        cute.copy(tma_x, tAg[(None, bm, h_base + j)], tAs[(None, prod.index)], tma_bar_ptr=bar)
                        cute.copy(tma_b, tBg[(None, 0, h_base + j)], tBs[(None, prod.index)], tma_bar_ptr=bar)
                        ring.producer_commit(prod)
                        prod.advance()
                    tile += gdim
                ring.producer_tail(prod)
        if wg_idx != 0:
            cute.arch.setmaxregister_increase(232)
            # Let F2 launch now: its producer prefetches x while F1 is still running,
            # and its consumers wait for these partials with griddepcontrol.wait.
            cute.arch.griddepcontrol_launch_dependents()
            mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            for _ in cutlass.range(w, unroll=1):
                cons.advance()
            tile = bidx
            while tile < num_tiles:
                bm = tile // self.S
                s = tile % self.S
                m0 = bm * BLK_M
                acc.fill(0.0)
                ss = F32(0.0)
                for jj in range(self.n_sub // 2):
                    ring.consumer_wait(cons)
                    st = cons.index
                    cute.nvgpu.warpgroup.fence()
                    for kb in cutlass.range_constexpr(num_k):
                        cute.gemm(mma, acc, tCrA[(None, None, kb, st)], tCrB[(None, None, kb, st)], acc)
                    cute.nvgpu.warpgroup.commit_group()
                    ss_j = F32(0.0)
                    for c in cutlass.range_constexpr(BLK_H // 2):
                        xv = F32(sA[(r, c0 + c, st)])
                        ss_j += xv * xv
                    ss += ss_j
                    cute.nvgpu.warpgroup.wait_group(0)
                    ring.consumer_release(cons)
                    cons.advance()
                    cons.advance()
                ss += cute.arch.shuffle_sync_bfly(ss, offset=1)
                # warpgroup 1 parks its partials, warpgroup 0 adds them and writes
                if w == 1:
                    for i in cutlass.range_constexpr(self.n_acc):
                        sRed[wt * stride + i] = acc[i]
                    sRed[wt * stride + self.n_acc] = ss
                all_bar.arrive_and_wait()
                if w == 0:
                    for i in cutlass.range_constexpr(self.n_acc):
                        crd = tCc[i]
                        if m0 + crd[0] < M:
                            mPp[(s, m0 + crd[0], crd[1])] = acc[i] + sRed[wt * stride + i]
                    if wt % 2 == 0:
                        if m0 + r < M:
                            mSSp[(s, m0 + r)] = ss + sRed[wt * stride + self.n_acc]
                all_bar.arrive_and_wait()                # sRed free for the next tile
                tile += gdim


# =============================================================================
# F2: statistics finalize + gate + output
# =============================================================================
class GnFwdOut:
    """Tile = 64 rows x Hc = H / n_hc columns (persistent, strided over num_SMs CTAs)."""

    def __init__(self, hidden, rank, n_split, n_hc, eps, lgw, num_sms, stages, epi_stage=2):
        hc = hidden // n_hc
        if rank != RANK or hidden % n_hc or hc % (2 * BLK_H):
            raise ValueError(f"H/n_hc={hidden}/{n_hc} must be a multiple of {2 * BLK_H}")
        self.H, self.R, self.S, self.n_hc = hidden, rank, n_split, n_hc
        self.Hc, self.n_sub = hc, hc // BLK_H
        self.eps, self.lgw, self.inv_h = float(eps), float(lgw), 1.0 / float(hidden)
        self.num_sms, self.stages, self.epi_stage = num_sms, stages, epi_stage

    @cute.jit
    def __call__(self, mX: cute.Tensor, mW: cute.Tensor, mGD: cute.Tensor, mPp: cute.Tensor,
                 mSSp: cute.Tensor, mY: cute.Tensor, mRho: cute.Tensor, mP: cute.Tensor,
                 stream: cuda.CUstream):
        tile_mnk = (BLK_M, BLK_H, self.R)
        mma = make_mma(ROW, ROW, (BLK_M, BLK_H))
        s_l = make_smem_layout_a(ROW, tile_mnk, BF16, 1)            # S: A, 64 x 16
        b_l = make_smem_layout_b(ROW, tile_mnk, BF16, self.stages)  # W_down: B, 64 x 16
        x_l = make_smem_layout_epi(BF16, ROW, (BLK_M, BLK_H), self.stages)
        y_l = make_smem_layout_epi(BF16, ROW, (BLK_M, BLK_H), 2 * self.epi_stage)
        tma_x, tX = tma_load_atom(mX, x_l, (BLK_M, BLK_H))
        tma_b, tGD = tma_load_atom(mGD, b_l, (BLK_H, self.R))
        tma_y, tY = tma_store_atom(mY, y_l, (BLK_M, BLK_H))

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            sX: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(x_l)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(b_l)], 1024]
            sY: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(y_l)], 1024]
            sS: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(s_l)], 1024]
            sW: cute.struct.Align[cute.struct.MemRange[F32, self.Hc], 16]
            sRho: cute.struct.Align[cute.struct.MemRange[F32, BLK_M], 16]

        self.shared_storage = SharedStorage
        M = mX.shape[0]
        num_tiles = (M + BLK_M - 1) // BLK_M * self.n_hc
        self.kernel(tma_x, tX, tma_b, tGD, tma_y, tY, mW, mPp, mSSp, mRho, mP, mma, x_l, b_l, y_l,
                    s_l, M, num_tiles).launch(
            grid=(self.num_sms, 1, 1), block=(3 * WG, 1, 1), min_blocks_per_mp=1, stream=stream,
            use_pdl=True)

    @cute.kernel
    def kernel(self, tma_x: cute.CopyAtom, mX: cute.Tensor, tma_b: cute.CopyAtom, mGD: cute.Tensor,
               tma_y: cute.CopyAtom, mY: cute.Tensor, mW: cute.Tensor, mPp: cute.Tensor,
               mSSp: cute.Tensor, mRho: cute.Tensor, mP: cute.Tensor, mma: cute.TiledMma,
               x_l: cute.ComposedLayout, b_l: cute.ComposedLayout, y_l: cute.ComposedLayout,
               s_l: cute.ComposedLayout, M: cutlass.Int32, num_tiles: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // WG)
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_y)

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        tx = (cute.size_in_bytes(BF16, cute.slice_(x_l, (None, None, 0)))
              + cute.size_in_bytes(BF16, cute.slice_(b_l, (None, None, 0))))
        ring = make_ring(storage.mbar, self.stages, tx)
        sX = storage.sX.get_tensor(x_l.outer, swizzle=x_l.inner)
        sB = storage.sB.get_tensor(b_l.outer, swizzle=b_l.inner)
        sY = storage.sY.get_tensor(y_l.outer, swizzle=y_l.inner)
        sS = storage.sS.get_tensor(s_l.outer, swizzle=s_l.inner)
        sW = storage.sW.get_tensor(cute.make_layout(self.Hc))
        sRho = storage.sRho.get_tensor(cute.make_layout(BLK_M))
        tXs, tXg = tma_part(tma_x, sX, cute.local_tile(mX, (BLK_M, BLK_H), (None, None)))
        tBs, tBg = tma_part(tma_b, sB, cute.local_tile(mGD, (BLK_H, self.R), (None, None)))
        tYs, tYg = tma_part(tma_y, sY, cute.local_tile(mY, (BLK_M, BLK_H), (None, None)))

        # CuTe DSL user objects must be constructed outside dynamic warp-role guards.
        w = wg_idx - 1
        ct = tidx - WG
        wt = tidx % WG
        all_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=2 * WG)
        wg_bar_id = 2 + w
        thr = mma.get_slice(wt)
        tCrA = mma.make_fragment_A(thr.partition_A(sS))
        tCrB = mma.make_fragment_B(thr.partition_B(sB))
        acc = thr.make_fragment_C(thr.partition_shape_C((BLK_M, BLK_H)))
        tCc = thr.partition_C(cute.make_identity_tensor((BLK_M, BLK_H)))
        s2r, tSX, rX, vX = make_s2r_acc(mma, wt, sX)
        r2s, tRS_s, tRS_acc, tRS_f, tRS_b = make_r2s_acc(mma, wt, sY, acc)
        store_ring = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, WG))
        cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
        sr = ct // 4
        sk0 = (ct % 4) * 4
        pv = cute.make_rmem_tensor((4,), F32)
        n_store = cutlass.Int32(0)
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                # x does not depend on F1: the producer starts while F1 is still running
                prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                tile = bidx
                while tile < num_tiles:
                    bm = tile // self.n_hc
                    h_base = (tile % self.n_hc) * self.n_sub
                    for j in range(self.n_sub):
                        ring.producer_acquire(prod)
                        bar = ring.producer_get_barrier(prod)
                        cute.copy(tma_x, tXg[(None, bm, h_base + j)], tXs[(None, prod.index)], tma_bar_ptr=bar)
                        cute.copy(tma_b, tBg[(None, h_base + j, 0)], tBs[(None, prod.index)], tma_bar_ptr=bar)
                        ring.producer_commit(prod)
                        prod.advance()
                    tile += gdim
                ring.producer_tail(prod)
        if wg_idx != 0:
            cute.arch.setmaxregister_increase(232)
            cute.arch.griddepcontrol_wait()              # F1 partials (and prep outputs)
            mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

            for _ in cutlass.range(w, unroll=1):
                cons.advance()

            tile = bidx
            while tile < num_tiles:
                bm = tile // self.n_hc
                bc = tile % self.n_hc
                m0 = bm * BLK_M
                all_bar.arrive_and_wait()                # previous tile's sS / sW / sRho readers done
                # ---- reduce the split-K partials: rho, P, S = silu(P) ----
                m = m0 + sr
                ss = F32(0.0)
                pv.fill(0.0)
                if m < M:
                    for s in cutlass.range_constexpr(self.S):
                        ss += mSSp[(s, m)]
                        for kk in cutlass.range_constexpr(4):
                            pv[kk] += mPp[(s, m, sk0 + kk)]
                rho = cute.math.rsqrt(ss * self.inv_h + self.eps)
                for kk in cutlass.range_constexpr(4):
                    pk = pv[kk] * rho
                    pv[kk] = pk
                    sS[(sr, sk0 + kk, 0)] = BF16(pk * sigmoid(pk))
                if sk0 == 0:
                    sRho[sr] = rho * self.lgw
                if bc == 0:
                    if m < M:
                        for kk in cutlass.range_constexpr(4):
                            mP[(m, sk0 + kk)] = pv[kk]
                        if sk0 == 0:
                            mRho[m] = rho
                hc0 = bc * self.Hc
                for i in cutlass.range_constexpr(-(-self.Hc // (2 * WG))):
                    hw = i * 2 * WG + ct
                    if hw < self.Hc:
                        sW[hw] = mW[hc0 + hw]
                cute.arch.fence_proxy("async.shared", space="cta")   # sS visible to wgmma
                all_bar.arrive_and_wait()

                # ---- this warpgroup's sub-tiles ----
                for jj in range(self.n_sub // 2):
                    j = jj * 2 + w
                    ring.consumer_wait(cons)
                    st = cons.index
                    cute.nvgpu.warpgroup.fence()
                    cute.gemm(mma, acc, tCrA[(None, None, 0, 0)], tCrB[(None, None, 0, st)], acc)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.copy(s2r, tSX[(None, None, None, st)], vX)
                    cute.nvgpu.warpgroup.wait_group(0)
                    for i in cutlass.range_constexpr(cute.size(acc)):
                        crd = tCc[i]
                        acc[i] = F32(rX[i]) * sRho[crd[0]] * sW[j * BLK_H + crd[1]] * sigmoid(acc[i])
                    ring.consumer_release(cons)
                    cons.advance()
                    cons.advance()

                    for iv in cutlass.range_constexpr(cute.size(tRS_f)):
                        tRS_f[iv] = tRS_acc[iv]
                    tRS_b.store(tRS_f.load().to(BF16))
                    buf = w * self.epi_stage + n_store % self.epi_stage
                    cute.copy(r2s, tRS_b, tRS_s[(None, None, None, buf)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=wg_bar_id, number_of_threads=WG)
                    if warp_idx % 4 == 0:
                        cute.copy(tma_y, tYs[(None, buf)], tYg[(None, bm, bc * self.n_sub + j)])
                        store_ring.producer_commit()
                        store_ring.producer_acquire()
                    cute.arch.barrier(barrier_id=wg_bar_id, number_of_threads=WG)
                    n_store += 1
                tile += gdim
            store_ring.producer_tail()


# =============================================================================
def _splits(H, unit, limit=None):
    return [
        n
        for n in range(1, H // unit + 1)
        if H % n == 0 and (H // n) % unit == 0 and (limit is None or n <= limit)
    ]


def pick_n_split(H):
    """Choose the deterministic F1 split from H only (independent of token count)."""
    valid = _splits(H, 2 * BLK_H, limit=8)
    for n in valid:
        if (4096 // BLK_M) * n >= 2 * 132:
            return n
    return valid[-1]


SMEM_CAP = 227 * 1024


def f2_stage_smem():
    return BLK_M * BLK_H * 2 + BLK_H * RANK * 2


def f2_fixed_smem(H, n_hc):
    return (
        4 * BLK_M * BLK_H * 2
        + BLK_M * RANK * 2
        + (H // n_hc) * 4
        + BLK_M * 4
        + 2048
    )


def pick_f2_stages(H, n_hc, budget=220 * 1024):
    return max(
        2,
        min(12, (budget - f2_fixed_smem(H, n_hc)) // f2_stage_smem()),
    )


def f2_smem(H, n_hc):
    return f2_fixed_smem(H, n_hc) + pick_f2_stages(H, n_hc) * f2_stage_smem()


def f2_chunk_options(H, sms):
    """Return the bounded set of F2 variants that may be selected for this H."""
    del sms  # Kept in the signature for callers that reason about one launch budget.
    options = [
        n
        for n in _splits(H, 2 * BLK_H)
        if f2_smem(H, n) <= SMEM_CAP
    ]
    if not options:
        raise ValueError(f"no F2 split for hidden size {H} fits in shared memory")
    return options


def pick_n_hc(M, H, sms):
    """Choose the F2 split with the lowest estimated bytes per occupied SM."""
    split_k = pick_n_split(H)
    row_blocks = -(-M // BLK_M)

    def cost(n_hc):
        tiles = row_blocks * n_hc
        fill = tiles / (-(-tiles // sms) * sms)
        return (4 * H + 68 * n_hc * split_k) / fill

    return min(f2_chunk_options(H, sms), key=lambda n_hc: (cost(n_hc), n_hc))


def _fwd_plan(H, R, dtypes, eps, lgw, dev, n_hc, sms):
    """Compile an M-independent forward plan for one bounded F2 split."""
    key = ("fwd", H, R, dtypes, eps, lgw, dev.index, n_hc, sms)
    plan = PLANS.get(key)
    if plan is None:
        split_k = pick_n_split(H)
        m = sym_m()
        tu, tw, td = (to_cute_dtype(dtype) for dtype in dtypes)
        prep = compile_op(
            ("prep", H, R, dtypes),
            lambda: GnPrep(H, R),
            (
                fake(tu, (R, H)),
                fake(tw, (H,)),
                fake(td, (H, R)),
                fake(BF16, (R, H)),
                fake(F32, (H,)),
                fake(BF16, (H, R)),
                fake(BF16, (R, H)),
                fake(BF16, (H, R)),
            ),
        )
        f1 = compile_op(
            ("f1", H, R, split_k, sms),
            lambda: GnFwdStats(H, R, split_k, sms),
            (
                fake(BF16, (m, H)),
                fake(BF16, (R, H)),
                fake(F32, (split_k, m, R)),
                fake(F32, (split_k, m), 4),
            ),
        )
        stages = pick_f2_stages(H, n_hc)
        f2 = compile_op(
            ("f2", H, R, split_k, n_hc, eps, lgw, sms, stages),
            lambda: GnFwdOut(
                H,
                R,
                split_k,
                n_hc,
                eps,
                lgw,
                sms,
                stages,
            ),
            (
                fake(BF16, (m, H)),
                fake(F32, (H,)),
                fake(BF16, (H, R)),
                fake(F32, (split_k, m, R)),
                fake(F32, (split_k, m), 4),
                fake(BF16, (m, H)),
                fake(F32, (m,)),
                fake(F32, (m, R)),
            ),
        )
        plan = (prep, f1, f2, split_k)
        PLANS[key] = plan
    return plan


def precompile_fwd(
    H,
    R,
    dtypes,
    eps,
    lgw,
    dev,
    *,
    num_sms=None,
    n_hc=None,
):
    """Compile all F2 variants reachable by the supplied configuration."""
    sms = resolve_num_sms(dev, num_sms)
    options = [n_hc] if n_hc is not None else f2_chunk_options(H, sms)
    for option in options:
        _fwd_plan(H, R, dtypes, eps, lgw, dev, option, sms)


def gated_norm_fwd(
    x,
    weight,
    gate_up_weight,
    gate_down_weight,
    eps,
    lgw,
    *,
    num_sms=None,
    n_hc=None,
):
    """Run forward and return output, saved row statistics and prepared weights."""
    H = x.shape[-1]
    R = gate_up_weight.shape[0]
    x2d = x.reshape(-1, H)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    M = x2d.shape[0]
    dev = x.device
    sms = resolve_num_sms(dev, num_sms)
    if n_hc is None:
        n_hc = pick_n_hc(M, H, sms)
    dtypes = (gate_up_weight.dtype, weight.dtype, gate_down_weight.dtype)
    prep, f1, f2, split_k = _fwd_plan(
        H,
        R,
        dtypes,
        float(eps),
        float(lgw),
        dev,
        n_hc,
        sms,
    )

    w_eff = torch.empty(R, H, device=dev, dtype=torch.bfloat16)
    w_f = torch.empty(H, device=dev, dtype=torch.float32)
    gd_bf = torch.empty(H, R, device=dev, dtype=torch.bfloat16)
    gd_t = torch.empty(R, H, device=dev, dtype=torch.bfloat16)
    gu_t = torch.empty(H, R, device=dev, dtype=torch.bfloat16)
    prep(
        gate_up_weight.contiguous(),
        weight.contiguous(),
        gate_down_weight.contiguous(),
        w_eff,
        w_f,
        gd_bf,
        gd_t,
        gu_t,
    )
    p_part, ss_part = workspace(
        ("fwd", H, R, split_k),
        dev,
        [(split_k, M, R), (split_k, M)],
    )
    f1(x2d, w_eff, p_part, ss_part)
    y = torch.empty_like(x2d)
    rho = torch.empty(M, device=dev, dtype=torch.float32)
    p = torch.empty(M, R, device=dev, dtype=torch.float32)
    f2(x2d, w_f, gd_bf, p_part, ss_part, y, rho, p)
    return y.view(x.shape), rho, p, (w_f, gd_bf, gd_t, gu_t)
