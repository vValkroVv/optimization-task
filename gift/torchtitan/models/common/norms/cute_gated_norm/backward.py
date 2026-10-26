"""GatedNorm backward on CuTe DSL (Hopper): two passes over x / dy. Forward: forward.py.

From the forward: x, rho (M,), p (M, 16) = rho * x @ W_eff^T (pre-SiLU). Per row, with
Y = rho x w, S = silu(p), G = sigmoid(S @ W_down^T), dy the incoming output gradient:
    dY1 = lgw * dy * G
    dL  = dY1 * Y * (1 - G)                 (gradient of the gate logits)
    dS  = dL @ W_down,   dP = dS * silu'(p)
    c   = sum_h dY * Y = sum_h dY1 * Y + sum_k dP_k p_k        (because P = Y @ W_up^T)
    dY  = dY1 + dP @ W_up
    dx  = rho * w * dY - rho^2 * c / H * x
    dW_down = dL^T S,   dW_up^T = w * x^T (rho dP),   dw = sum_m dY * rho * x
The identity for c removes a third sweep: HBM traffic 10 B/elem (B1 reads x, dy; B2 reads
x, dy and writes dx).

Kernels (B1 -> B1f -> B2 -> Wfin; B1f, B2, Wfin are PDL dependents of their predecessor):
  B1    per 64 x 64 sub-tile: logits wgmma (S @ W_down^T) -> dY1, dL in the epilogue; dL goes
        to smem with stmatrix and feeds dS += dL @ W_down (K-major view) and
        dW_down += dL^T S (M-major view of the same bytes); those two wgmmas are waited for
        at the next sub-tile. Writes ds_part / c1_part per (H chunk, row).
  B1f   dP = silu'(p) * sum ds_part,  c = sum c1_part + dP . p.
  B2    per sub-tile, one commit group: logits wgmma, dP @ W_up^T wgmma and
        dW_up^T += x^T (rho dP) with the x smem stage itself as the M-major A operand;
        epilogue dY, dx (stmatrix + TMA store) and per-column dw partials.
  Wfin  weight grads = sum of the per-row-group partials, written in the parameters' dtypes.

B1 / B2: one tile per CTA = (H chunk, group of 64-row blocks), grid <= num_SMs. Weight
gradients are accumulated in registers over the CTA's rows and written once to a
(groups, H, 16) workspace: deterministic, no atomics, no zero fill. Per-row results (dx)
only depend on H-chunking, so they are batch invariant. Weight-grad wgmma operands are bf16.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_compact_tensor

from .common import (
    BF16,
    BLK_H,
    BLK_M,
    COL,
    F32,
    PLANS,
    RANK,
    ROW,
    WG,
    compile_op,
    resolve_num_sms,
    sigmoid,
    to_cute_dtype,
    workspace,
)


class _BwdTile:
    """Tile = (H chunk bc, row group g): Hc = H / n_hc columns, rows split into G groups."""

    def __init__(self, hidden, rank, n_hc, n_groups, lgw, stages):
        hc = hidden // n_hc
        if rank != RANK or hidden % n_hc or hc % (2 * BLK_H):
            raise ValueError(
                f"H/n_hc={hidden}/{n_hc} must be a multiple of {2 * BLK_H}"
            )
        self.H, self.R, self.n_hc, self.G = hidden, rank, n_hc, n_groups
        self.Hc, self.n_sub, self.n_jj = hc, hc // BLK_H, hc // BLK_H // 2
        self.lgw, self.stages = float(lgw), stages


def _row_range(bidx, n_hc, G, M):
    bc = bidx % n_hc
    g = bidx // n_hc
    n_rb = (M + BLK_M - 1) // BLK_M
    rpg = (n_rb + G - 1) // G
    rb0 = g * rpg
    # DSL for-loops require int32 counters. These count 64-row tiles and
    # are bounded by the validated row count; tensor offsets stay int64.
    return bc, g, cutlass.Int32(rb0), cutlass.Int32(cutlass.min(rb0 + rpg, n_rb))


# =============================================================================
# B1: dS / c1 partials per H chunk, dW_down per row group
# =============================================================================
class GnBwdDs(_BwdTile):
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mDY: cute.Tensor,
        mW: cute.Tensor,
        mGD: cute.Tensor,
        mGDt: cute.Tensor,
        mRho: cute.Tensor,
        mP: cute.Tensor,
        mDSp: cute.Tensor,
        mC1p: cute.Tensor,
        mWsGD: cute.Tensor,
        stream: cuda.CUstream,
    ):
        mma_g = sm90_utils.make_trivial_tiled_mma(
            BF16,
            BF16,
            ROW.sm90_mma_major_mode(),
            ROW.sm90_mma_major_mode(),
            F32,
            (1, 1, 1),
            tiler_mn=(BLK_M, BLK_H),
        )  # S @ W_down^T
        mma_ds = sm90_utils.make_trivial_tiled_mma(
            BF16,
            BF16,
            ROW.sm90_mma_major_mode(),
            ROW.sm90_mma_major_mode(),
            F32,
            (1, 1, 1),
            tiler_mn=(BLK_M, self.R),
        )  # dL @ W_down
        mma_wg = sm90_utils.make_trivial_tiled_mma(
            BF16,
            BF16,
            COL.sm90_mma_major_mode(),
            COL.sm90_mma_major_mode(),
            F32,
            (1, 1, 1),
            tiler_mn=(BLK_H, self.R),
        )  # dL^T @ S
        x_l = sm90_utils.make_smem_layout_epi(BF16, ROW, (BLK_M, BLK_H), self.stages)
        gd_l = sm90_utils.make_smem_layout_b(
            ROW, (BLK_M, BLK_H, self.R), BF16, self.stages
        )
        gdt_l = sm90_utils.make_smem_layout_b(
            ROW, (BLK_M, self.R, BLK_H), BF16, self.stages
        )
        s_l = sm90_utils.make_smem_layout_a(ROW, (BLK_M, BLK_H, self.R), BF16, 1)
        st_l = sm90_utils.make_smem_layout_b(COL, (BLK_H, self.R, BLK_M), BF16, 1)
        # dL tile per warpgroup, row-major (m, h) SW128: the same bytes are the K-major A of
        # dL @ W_down and the M-major A (M = h, K = m) of dL^T @ S.
        dlk_l = sm90_utils.make_smem_layout_a(ROW, (BLK_M, self.R, BLK_H), BF16, 2)
        dlm_l = sm90_utils.make_smem_layout_a(COL, (BLK_H, self.R, BLK_M), BF16, 2)
        tma_x, tX = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mX,
            cute.slice_(x_l, (None, None, 0)),
            (BLK_M, BLK_H),
        )
        tma_dy, tDY = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mDY,
            cute.slice_(x_l, (None, None, 0)),
            (BLK_M, BLK_H),
        )
        tma_gd, tGD = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mGD,
            cute.slice_(gd_l, (None, None, 0)),
            (BLK_H, self.R),
        )
        tma_gdt, tGDt = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mGDt,
            cute.slice_(gdt_l, (None, None, 0)),
            (self.R, BLK_H),
        )

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            s_x: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(x_l)], 1024]
            s_dy: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(x_l)], 1024]
            s_gd: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(gd_l)], 1024]
            s_gdt: cute.struct.Align[
                cute.struct.MemRange[BF16, cute.cosize(gdt_l)], 1024
            ]
            s_s: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(s_l)], 1024]
            s_st: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(st_l)], 1024]
            s_dl: cute.struct.Align[
                cute.struct.MemRange[BF16, cute.cosize(dlk_l)], 1024
            ]
            s_w: cute.struct.Align[cute.struct.MemRange[F32, self.Hc], 16]
            s_rho: cute.struct.Align[cute.struct.MemRange[F32, BLK_M], 16]
            s_red: cute.struct.Align[cute.struct.MemRange[F32, WG * 8], 16]
            s_c1: cute.struct.Align[cute.struct.MemRange[F32, BLK_M], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_x,
            tX,
            tma_dy,
            tDY,
            tma_gd,
            tGD,
            tma_gdt,
            tGDt,
            mW,
            mRho,
            mP,
            mDSp,
            mC1p,
            mWsGD,
            mma_g,
            mma_ds,
            mma_wg,
            x_l,
            gd_l,
            gdt_l,
            s_l,
            st_l,
            dlk_l,
            dlm_l,
            cutlass.Int64(mX.shape[0]),
        ).launch(
            grid=(self.n_hc * self.G, 1, 1),
            block=(3 * WG, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_x: cute.CopyAtom,
        mX: cute.Tensor,
        tma_dy: cute.CopyAtom,
        mDY: cute.Tensor,
        tma_gd: cute.CopyAtom,
        mGD: cute.Tensor,
        tma_gdt: cute.CopyAtom,
        mGDt: cute.Tensor,
        mW: cute.Tensor,
        mRho: cute.Tensor,
        mP: cute.Tensor,
        mDSp: cute.Tensor,
        mC1p: cute.Tensor,
        mWsGD: cute.Tensor,
        mma_g: cute.TiledMma,
        mma_ds: cute.TiledMma,
        mma_wg: cute.TiledMma,
        x_l: cute.ComposedLayout,
        gd_l: cute.ComposedLayout,
        gdt_l: cute.ComposedLayout,
        s_l: cute.ComposedLayout,
        st_l: cute.ComposedLayout,
        dlk_l: cute.ComposedLayout,
        dlm_l: cute.ComposedLayout,
        M: cutlass.Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // WG)
        bc, g, rb0, rb1 = _row_range(bidx, self.n_hc, self.G, M)
        h_base = bc * self.n_sub
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_x)
            cpasync.prefetch_descriptor(tma_dy)
            cpasync.prefetch_descriptor(tma_gd)
            cpasync.prefetch_descriptor(tma_gdt)

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        tx = (
            2 * cute.size_in_bytes(BF16, cute.slice_(x_l, (None, None, 0)))
            + cute.size_in_bytes(BF16, cute.slice_(gd_l, (None, None, 0)))
            + cute.size_in_bytes(BF16, cute.slice_(gdt_l, (None, None, 0)))
        )
        ring = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=tx,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        sX = storage.s_x.get_tensor(x_l.outer, swizzle=x_l.inner)
        sDY = storage.s_dy.get_tensor(x_l.outer, swizzle=x_l.inner)
        sGD = storage.s_gd.get_tensor(gd_l.outer, swizzle=gd_l.inner)
        sGDt = storage.s_gdt.get_tensor(gdt_l.outer, swizzle=gdt_l.inner)
        sS = storage.s_s.get_tensor(s_l.outer, swizzle=s_l.inner)
        sSt = storage.s_st.get_tensor(st_l.outer, swizzle=st_l.inner)
        sDLk = storage.s_dl.get_tensor(dlk_l.outer, swizzle=dlk_l.inner)
        sDLm = storage.s_dl.get_tensor(dlm_l.outer, swizzle=dlm_l.inner)
        sW = storage.s_w.get_tensor(cute.make_layout(self.Hc))
        sRho = storage.s_rho.get_tensor(cute.make_layout(BLK_M))
        sRed = storage.s_red.get_tensor(cute.make_layout(WG * 8))
        sC1 = storage.s_c1.get_tensor(cute.make_layout(BLK_M))
        tXs, tXg = cpasync.tma_partition(
            tma_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(cute.local_tile(mX, (BLK_M, BLK_H), (None, None)), 0, 2),
        )
        tDYs, tDYg = cpasync.tma_partition(
            tma_dy,
            0,
            cute.make_layout(1),
            cute.group_modes(sDY, 0, 2),
            cute.group_modes(cute.local_tile(mDY, (BLK_M, BLK_H), (None, None)), 0, 2),
        )
        tGDs, tGDg = cpasync.tma_partition(
            tma_gd,
            0,
            cute.make_layout(1),
            cute.group_modes(sGD, 0, 2),
            cute.group_modes(cute.local_tile(mGD, (BLK_H, self.R), (None, None)), 0, 2),
        )
        tGDts, tGDtg = cpasync.tma_partition(
            tma_gdt,
            0,
            cute.make_layout(1),
            cute.group_modes(sGDt, 0, 2),
            cute.group_modes(
                cute.local_tile(mGDt, (self.R, BLK_H), (None, None)), 0, 2
            ),
        )

        # CuTe DSL user objects must be constructed outside dynamic warp-role guards.
        w = wg_idx - 1
        ct = tidx - WG
        wt = tidx % WG
        lane = wt % 32
        all_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=2 * WG)
        wg_bar_id = 2 + w
        thr_g = mma_g.get_slice(wt)
        tArS = mma_g.make_fragment_A(thr_g.partition_A(sS))
        tBrGD = mma_g.make_fragment_B(thr_g.partition_B(sGD))
        acc = thr_g.make_fragment_C(thr_g.partition_shape_C((BLK_M, BLK_H)))
        tCc = thr_g.partition_C(cute.make_identity_tensor((BLK_M, BLK_H)))
        s2r = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BF16
            ),
            mma_g,
        )
        s2r_thr = s2r.get_slice(wt)
        tSX = s2r_thr.partition_S(sX)
        rX = cute.make_rmem_tensor_like(acc, BF16)
        vX = s2r_thr.retile(rX)
        tSDY = s2r_thr.partition_S(sDY)
        rDY = cute.make_rmem_tensor_like(acc, BF16)
        vDY = s2r_thr.retile(rDY)
        r2s = cute.make_tiled_copy_C(
            sm90_utils.get_smem_store_op(ROW, BF16, F32), mma_g
        )
        tRS_s = r2s.get_slice(wt).partition_D(sDLk)
        r_out = cute.make_rmem_tensor_like(acc, BF16)
        tRS_b = r2s.retile(r_out)
        thr_ds = mma_ds.get_slice(wt)
        tArDLk = mma_ds.make_fragment_A(thr_ds.partition_A(sDLk))
        tBrGDt = mma_ds.make_fragment_B(thr_ds.partition_B(sGDt))
        num_k_ds = cute.size(tArDLk, mode=[2])
        acc_ds = thr_ds.make_fragment_C(thr_ds.partition_shape_C((BLK_M, self.R)))
        tDSc = thr_ds.partition_C(cute.make_identity_tensor((BLK_M, self.R)))
        thr_wg = mma_wg.get_slice(wt)
        tArDL = mma_wg.make_fragment_A(thr_wg.partition_A(sDLm))
        tBrSt = mma_wg.make_fragment_B(thr_wg.partition_B(sSt))
        acc_dw = [
            thr_wg.make_fragment_C(thr_wg.partition_shape_C((BLK_H, self.R)))
            for _ in range(self.n_jj)
        ]
        tWc = thr_wg.partition_C(cute.make_identity_tensor((BLK_H, self.R)))
        num_k_wg = cute.size(tArDL, mode=[2])
        cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages
        )
        rel = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages
        )
        sr = ct // 4
        sk0 = (ct % 4) * 4
        c1 = cute.make_rmem_tensor((2,), F32)
        row0 = (wt // 32) * 16 + lane // 4
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages
                )
                for rb in range(rb0, rb1):
                    for j in range(self.n_sub):
                        ring.producer_acquire(prod)
                        bar = ring.producer_get_barrier(prod)
                        st = prod.index
                        cute.copy(
                            tma_x,
                            tXg[(None, rb, h_base + j)],
                            tXs[(None, st)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_dy,
                            tDYg[(None, rb, h_base + j)],
                            tDYs[(None, st)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_gd,
                            tGDg[(None, h_base + j, 0)],
                            tGDs[(None, st)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_gdt,
                            tGDtg[(None, 0, h_base + j)],
                            tGDts[(None, st)],
                            tma_bar_ptr=bar,
                        )
                        ring.producer_commit(prod)
                        prod.advance()
                ring.producer_tail(prod)
        if wg_idx != 0:
            cute.arch.setmaxregister_increase(232)
            for jj in cutlass.range_constexpr(self.n_jj):
                acc_dw[jj].fill(0.0)
            mma_ds.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            mma_wg.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            mma_g.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

            hc0 = bc * self.Hc
            for i in cutlass.range_constexpr(-(-self.Hc // (2 * WG))):
                hw = i * 2 * WG + ct
                if hw < self.Hc:
                    sW[hw] = mW[hc0 + hw]

            for _ in cutlass.range(w, unroll=1):
                cons.advance()
                rel.advance()

            for rb in range(rb0, rb1):
                m0 = rb * BLK_M
                all_bar.arrive_and_wait()
                # ---- row stats: S = silu(p) (A of the logits gemm, B of dW_down), rho ----
                m = m0 + sr
                for kk in cutlass.range_constexpr(4):
                    pk = F32(0.0)
                    if m < M:
                        pk = mP[(m, sk0 + kk)]
                    sv = BF16(pk * sigmoid(pk))
                    sS[(sr, sk0 + kk, 0)] = sv
                    sSt[(sk0 + kk, sr, 0)] = sv
                if sk0 == 0:
                    rv = F32(0.0)
                    if m < M:
                        rv = mRho[m]
                    sRho[sr] = rv
                cute.arch.fence_proxy("async.shared", space="cta")
                all_bar.arrive_and_wait()

                acc_ds.fill(0.0)
                c1.fill(0.0)
                for jj in cutlass.range_constexpr(self.n_jj):
                    j = jj * 2 + w
                    ring.consumer_wait(cons)
                    st = cons.index
                    cute.nvgpu.warpgroup.fence()
                    cute.gemm(
                        mma_g,
                        acc,
                        tArS[(None, None, 0, 0)],
                        tBrGD[(None, None, 0, st)],
                        acc,
                    )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.copy(s2r, tSX[(None, None, None, st)], vX)
                    cute.copy(s2r, tSDY[(None, None, None, st)], vDY)
                    cute.nvgpu.warpgroup.wait_group(
                        0
                    )  # also retires the previous dS / dW_down gemms
                    if jj > 0:
                        ring.consumer_release(rel)
                        rel.advance()
                        rel.advance()
                    for i in cutlass.range_constexpr(cute.size(acc)):
                        crd = tCc[i]
                        gt = sigmoid(acc[i])
                        yv = sRho[crd[0]] * F32(rX[i]) * sW[j * BLK_H + crd[1]]
                        d1 = self.lgw * F32(rDY[i]) * gt
                        dl = d1 * yv * (1.0 - gt)
                        c1[(i // 2) % 2] += (
                            d1 * yv
                        )  # accumulator value i lies in row half (i // 2) % 2
                        acc[i] = dl
                    r_out.store(acc.load().to(BF16))
                    cute.copy(r2s, tRS_b, tRS_s[(None, None, None, w)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=wg_bar_id, number_of_threads=WG)

                    # dS += dL @ W_down, dW_down += dL^T @ S; waited for at the next sub-tile
                    cute.nvgpu.warpgroup.fence()
                    for kb in cutlass.range_constexpr(num_k_ds):
                        cute.gemm(
                            mma_ds,
                            acc_ds,
                            tArDLk[(None, None, kb, w)],
                            tBrGDt[(None, None, kb, st)],
                            acc_ds,
                        )
                    for kb in cutlass.range_constexpr(num_k_wg):
                        cute.gemm(
                            mma_wg,
                            acc_dw[jj],
                            tArDL[(None, None, kb, w)],
                            tBrSt[(None, None, kb, 0)],
                            acc_dw[jj],
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cons.advance()
                    cons.advance()
                cute.nvgpu.warpgroup.wait_group(0)
                ring.consumer_release(rel)
                rel.advance()
                rel.advance()

                # ---- per-row results of this H chunk: combine the two warpgroups ----
                for hh in cutlass.range_constexpr(2):
                    cv = c1[hh]
                    # Preserve the ascending XOR tree; warp_reduction_sum reverses it.
                    cv += cute.arch.shuffle_sync_bfly(cv, offset=1)
                    cv += cute.arch.shuffle_sync_bfly(cv, offset=2)
                    c1[hh] = cv
                if w == 1:
                    for i in cutlass.range_constexpr(cute.size(acc_ds)):
                        sRed[wt * 8 + i] = acc_ds[i]
                    if lane % 4 == 0:
                        sC1[row0] = c1[0]
                        sC1[row0 + 8] = c1[1]
                all_bar.arrive_and_wait()
                if w == 0:
                    for i in cutlass.range_constexpr(cute.size(acc_ds)):
                        crd = tDSc[i]
                        if m0 + crd[0] < M:
                            mDSp[(bc, m0 + crd[0], crd[1])] = (
                                acc_ds[i] + sRed[wt * 8 + i]
                            )
                    if lane % 4 == 0:
                        for hh in cutlass.range_constexpr(2):
                            if m0 + row0 + 8 * hh < M:
                                mC1p[(bc, m0 + row0 + 8 * hh)] = (
                                    c1[hh] + sC1[row0 + 8 * hh]
                                )

            # ---- dW_down partial of this row group (each warpgroup owns its sub-tiles) ----
            for jj in cutlass.range_constexpr(self.n_jj):
                h0 = hc0 + (jj * 2 + w) * BLK_H
                for i in cutlass.range_constexpr(cute.size(acc_dw[jj])):
                    crd = tWc[i]
                    mWsGD[(g, h0 + crd[0], crd[1])] = acc_dw[jj][i]


# =============================================================================
# B1f: dP and c per row
# =============================================================================
class GnBwdFin:
    def __init__(self, n_part, rank):
        self.S, self.R = n_part, rank

    @cute.jit
    def __call__(
        self,
        mDSp: cute.Tensor,
        mC1p: cute.Tensor,
        mP: cute.Tensor,
        mDP: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        M = cutlass.Int64(mP.shape[0])
        self.kernel(mDSp, mC1p, mP, mDP, mC, M).launch(
            grid=((M * self.R + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mDSp: cute.Tensor,
        mC1p: cute.Tensor,
        mP: cute.Tensor,
        mDP: cute.Tensor,
        mC: cute.Tensor,
        M: cutlass.Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        idx = cutlass.Int64(bidx) * 256 + tidx
        m = idx // self.R
        k = idx % self.R
        t = F32(0.0)
        if m < M:
            ds = F32(0.0)
            for s in cutlass.range_constexpr(self.S):
                ds += mDSp[(s, m, k)]
            pk = mP[(m, k)]
            sg = sigmoid(pk)
            dpk = ds * sg * (1.0 + pk * (1.0 - sg))
            mDP[(m, k)] = dpk
            t = dpk * pk
        # sum_k dP_k p_k over the 16 lanes of the row
        # Preserve the ascending XOR tree used by the source kernel.
        t += cute.arch.shuffle_sync_bfly(t, offset=1)
        t += cute.arch.shuffle_sync_bfly(t, offset=2)
        t += cute.arch.shuffle_sync_bfly(t, offset=4)
        t += cute.arch.shuffle_sync_bfly(t, offset=8)
        if k == 0:
            if m < M:
                c = t
                for s in cutlass.range_constexpr(self.S):
                    c += mC1p[(s, m)]
                mC[m] = c


# =============================================================================
# B2: dx, dW_up, dw
# =============================================================================
class GnBwdDx(_BwdTile):
    def __init__(self, hidden, rank, n_hc, n_groups, lgw, stages, epi_stage=2):
        super().__init__(hidden, rank, n_hc, n_groups, lgw, stages)
        self.epi_stage = epi_stage
        self.inv_h = 1.0 / float(hidden)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mDY: cute.Tensor,
        mW: cute.Tensor,
        mGD: cute.Tensor,
        mGUt: cute.Tensor,
        mRho: cute.Tensor,
        mP: cute.Tensor,
        mDP: cute.Tensor,
        mC: cute.Tensor,
        mDX: cute.Tensor,
        mWsGU: cute.Tensor,
        mWsW: cute.Tensor,
        stream: cuda.CUstream,
    ):
        mma_g = sm90_utils.make_trivial_tiled_mma(
            BF16,
            BF16,
            ROW.sm90_mma_major_mode(),
            ROW.sm90_mma_major_mode(),
            F32,
            (1, 1, 1),
            tiler_mn=(BLK_M, BLK_H),
        )  # S @ W_down^T, dP @ W_up
        mma_wg = sm90_utils.make_trivial_tiled_mma(
            BF16,
            BF16,
            COL.sm90_mma_major_mode(),
            COL.sm90_mma_major_mode(),
            F32,
            (1, 1, 1),
            tiler_mn=(BLK_H, self.R),
        )  # x^T @ (rho dP)
        x_l = sm90_utils.make_smem_layout_epi(BF16, ROW, (BLK_M, BLK_H), self.stages)
        o_l = sm90_utils.make_smem_layout_epi(
            BF16, ROW, (BLK_M, BLK_H), 2 * self.epi_stage
        )
        gd_l = sm90_utils.make_smem_layout_b(
            ROW, (BLK_M, BLK_H, self.R), BF16, self.stages
        )
        s_l = sm90_utils.make_smem_layout_a(ROW, (BLK_M, BLK_H, self.R), BF16, 1)
        dpt_l = sm90_utils.make_smem_layout_b(COL, (BLK_H, self.R, BLK_M), BF16, 1)
        # the x stages viewed as the M-major A (M = h, K = m) of the dW_up gemm
        xa_l = sm90_utils.make_smem_layout_a(
            COL, (BLK_H, self.R, BLK_M), BF16, self.stages
        )
        tma_x, tX = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mX,
            cute.slice_(x_l, (None, None, 0)),
            (BLK_M, BLK_H),
        )
        tma_dy, tDY = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mDY,
            cute.slice_(x_l, (None, None, 0)),
            (BLK_M, BLK_H),
        )
        tma_gd, tGD = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mGD,
            cute.slice_(gd_l, (None, None, 0)),
            (BLK_H, self.R),
        )
        tma_gu, tGU = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mGUt,
            cute.slice_(gd_l, (None, None, 0)),
            (BLK_H, self.R),
        )
        tma_o, tO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mDX,
            cute.slice_(o_l, (None, None, 0)),
            (BLK_M, BLK_H),
        )

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            s_x: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(x_l)], 1024]
            s_dy: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(x_l)], 1024]
            s_gd: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(gd_l)], 1024]
            s_gu: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(gd_l)], 1024]
            s_o: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(o_l)], 1024]
            s_s: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(s_l)], 1024]
            s_dp: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(s_l)], 1024]
            s_dpt: cute.struct.Align[
                cute.struct.MemRange[BF16, cute.cosize(dpt_l)], 1024
            ]
            s_w: cute.struct.Align[cute.struct.MemRange[F32, self.Hc], 16]
            s_rho: cute.struct.Align[cute.struct.MemRange[F32, BLK_M], 16]
            s_b: cute.struct.Align[cute.struct.MemRange[F32, BLK_M], 16]
            s_dw: cute.struct.Align[cute.struct.MemRange[F32, 2 * 4 * BLK_H], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_x,
            tX,
            tma_dy,
            tDY,
            tma_gd,
            tGD,
            tma_gu,
            tGU,
            tma_o,
            tO,
            mW,
            mRho,
            mP,
            mDP,
            mC,
            mWsGU,
            mWsW,
            mma_g,
            mma_wg,
            x_l,
            o_l,
            gd_l,
            s_l,
            dpt_l,
            xa_l,
            cutlass.Int64(mX.shape[0]),
        ).launch(
            grid=(self.n_hc * self.G, 1, 1),
            block=(3 * WG, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tma_x: cute.CopyAtom,
        mX: cute.Tensor,
        tma_dy: cute.CopyAtom,
        mDY: cute.Tensor,
        tma_gd: cute.CopyAtom,
        mGD: cute.Tensor,
        tma_gu: cute.CopyAtom,
        mGUt: cute.Tensor,
        tma_o: cute.CopyAtom,
        mDX: cute.Tensor,
        mW: cute.Tensor,
        mRho: cute.Tensor,
        mP: cute.Tensor,
        mDP: cute.Tensor,
        mC: cute.Tensor,
        mWsGU: cute.Tensor,
        mWsW: cute.Tensor,
        mma_g: cute.TiledMma,
        mma_wg: cute.TiledMma,
        x_l: cute.ComposedLayout,
        o_l: cute.ComposedLayout,
        gd_l: cute.ComposedLayout,
        s_l: cute.ComposedLayout,
        dpt_l: cute.ComposedLayout,
        xa_l: cute.ComposedLayout,
        M: cutlass.Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // WG)
        bc, g, rb0, rb1 = _row_range(bidx, self.n_hc, self.G, M)
        h_base = bc * self.n_sub
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_x)
            cpasync.prefetch_descriptor(tma_dy)
            cpasync.prefetch_descriptor(tma_gd)
            cpasync.prefetch_descriptor(tma_gu)
            cpasync.prefetch_descriptor(tma_o)

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        tx = 2 * cute.size_in_bytes(
            BF16, cute.slice_(x_l, (None, None, 0))
        ) + 2 * cute.size_in_bytes(BF16, cute.slice_(gd_l, (None, None, 0)))
        ring = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=tx,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        sX = storage.s_x.get_tensor(x_l.outer, swizzle=x_l.inner)
        sXa = storage.s_x.get_tensor(xa_l.outer, swizzle=xa_l.inner)
        sDY = storage.s_dy.get_tensor(x_l.outer, swizzle=x_l.inner)
        sGD = storage.s_gd.get_tensor(gd_l.outer, swizzle=gd_l.inner)
        sGU = storage.s_gu.get_tensor(gd_l.outer, swizzle=gd_l.inner)
        sO = storage.s_o.get_tensor(o_l.outer, swizzle=o_l.inner)
        sS = storage.s_s.get_tensor(s_l.outer, swizzle=s_l.inner)
        sDP = storage.s_dp.get_tensor(s_l.outer, swizzle=s_l.inner)
        sDPt = storage.s_dpt.get_tensor(dpt_l.outer, swizzle=dpt_l.inner)
        sW = storage.s_w.get_tensor(cute.make_layout(self.Hc))
        sRho = storage.s_rho.get_tensor(cute.make_layout(BLK_M))
        sB = storage.s_b.get_tensor(cute.make_layout(BLK_M))
        sDw = storage.s_dw.get_tensor(cute.make_layout(2 * 4 * BLK_H))
        tXs, tXg = cpasync.tma_partition(
            tma_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(cute.local_tile(mX, (BLK_M, BLK_H), (None, None)), 0, 2),
        )
        tDYs, tDYg = cpasync.tma_partition(
            tma_dy,
            0,
            cute.make_layout(1),
            cute.group_modes(sDY, 0, 2),
            cute.group_modes(cute.local_tile(mDY, (BLK_M, BLK_H), (None, None)), 0, 2),
        )
        tGDs, tGDg = cpasync.tma_partition(
            tma_gd,
            0,
            cute.make_layout(1),
            cute.group_modes(sGD, 0, 2),
            cute.group_modes(cute.local_tile(mGD, (BLK_H, self.R), (None, None)), 0, 2),
        )
        tGUs, tGUg = cpasync.tma_partition(
            tma_gu,
            0,
            cute.make_layout(1),
            cute.group_modes(sGU, 0, 2),
            cute.group_modes(
                cute.local_tile(mGUt, (BLK_H, self.R), (None, None)), 0, 2
            ),
        )
        tOs, tOg = cpasync.tma_partition(
            tma_o,
            0,
            cute.make_layout(1),
            cute.group_modes(sO, 0, 2),
            cute.group_modes(cute.local_tile(mDX, (BLK_M, BLK_H), (None, None)), 0, 2),
        )

        # CuTe DSL user objects must be constructed outside dynamic warp-role guards.
        w = wg_idx - 1
        ct = tidx - WG
        wt = tidx % WG
        lane = wt % 32
        all_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=2 * WG)
        wg_bar_id = 2 + w
        thr_g = mma_g.get_slice(wt)
        tArS = mma_g.make_fragment_A(thr_g.partition_A(sS))
        tArDP = mma_g.make_fragment_A(thr_g.partition_A(sDP))
        tBrGD = mma_g.make_fragment_B(thr_g.partition_B(sGD))
        tBrGU = mma_g.make_fragment_B(thr_g.partition_B(sGU))
        acc = thr_g.make_fragment_C(thr_g.partition_shape_C((BLK_M, BLK_H)))
        acc_u = thr_g.make_fragment_C(thr_g.partition_shape_C((BLK_M, BLK_H)))
        tCc = thr_g.partition_C(cute.make_identity_tensor((BLK_M, BLK_H)))
        n_acc = cute.size(acc)
        s2r = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BF16
            ),
            mma_g,
        )
        s2r_thr = s2r.get_slice(wt)
        tSX = s2r_thr.partition_S(sX)
        rX = cute.make_rmem_tensor_like(acc, BF16)
        vX = s2r_thr.retile(rX)
        tSDY = s2r_thr.partition_S(sDY)
        rDY = cute.make_rmem_tensor_like(acc, BF16)
        vDY = s2r_thr.retile(rDY)
        r2s = cute.make_tiled_copy_C(
            sm90_utils.get_smem_store_op(ROW, BF16, F32), mma_g
        )
        tRS_s = r2s.get_slice(wt).partition_D(sO)
        r_out = cute.make_rmem_tensor_like(acc, BF16)
        tRS_b = r2s.retile(r_out)
        store_ring = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, WG),
        )
        thr_wg = mma_wg.get_slice(wt)
        tArX = mma_wg.make_fragment_A(thr_wg.partition_A(sXa))
        tBrDPt = mma_wg.make_fragment_B(thr_wg.partition_B(sDPt))
        acc_dw = [
            thr_wg.make_fragment_C(thr_wg.partition_shape_C((BLK_H, self.R)))
            for _ in range(self.n_jj)
        ]
        tWc = thr_wg.partition_C(cute.make_identity_tensor((BLK_H, self.R)))
        num_k_wg = cute.size(tArX, mode=[2])
        n_col = n_acc // 2
        dw = cute.make_rmem_tensor((self.n_jj * n_col,), F32)
        cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages
        )
        sr = ct // 4
        sk0 = (ct % 4) * 4
        n_store = cutlass.Int32(0)
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages
                )
                for rb in range(rb0, rb1):
                    for j in range(self.n_sub):
                        ring.producer_acquire(prod)
                        bar = ring.producer_get_barrier(prod)
                        st = prod.index
                        cute.copy(
                            tma_x,
                            tXg[(None, rb, h_base + j)],
                            tXs[(None, st)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_dy,
                            tDYg[(None, rb, h_base + j)],
                            tDYs[(None, st)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_gd,
                            tGDg[(None, h_base + j, 0)],
                            tGDs[(None, st)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_gu,
                            tGUg[(None, h_base + j, 0)],
                            tGUs[(None, st)],
                            tma_bar_ptr=bar,
                        )
                        ring.producer_commit(prod)
                        prod.advance()
                ring.producer_tail(prod)
        if wg_idx != 0:
            cute.arch.setmaxregister_increase(232)
            cute.arch.griddepcontrol_wait()  # dP / c come from B1f
            dw.fill(0.0)
            for jj in cutlass.range_constexpr(self.n_jj):
                acc_dw[jj].fill(0.0)
            mma_wg.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            mma_g.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

            hc0 = bc * self.Hc
            for i in cutlass.range_constexpr(-(-self.Hc // (2 * WG))):
                hw = i * 2 * WG + ct
                if hw < self.Hc:
                    sW[hw] = mW[hc0 + hw]
            all_bar.arrive_and_wait()  # sW is read below even with no rows

            for _ in cutlass.range(w, unroll=1):
                cons.advance()

            for rb in range(rb0, rb1):
                m0 = rb * BLK_M
                all_bar.arrive_and_wait()
                # ---- row stats: S, dP (bf16 A operand), rho * dP (B of dW_up), rho, rho^2 c / H ----
                m = m0 + sr
                rho_r = F32(0.0)
                c_r = F32(0.0)
                if m < M:
                    rho_r = mRho[m]
                    c_r = mC[m]
                for kk in cutlass.range_constexpr(4):
                    pk = F32(0.0)
                    dpk = F32(0.0)
                    if m < M:
                        pk = mP[(m, sk0 + kk)]
                        dpk = mDP[(m, sk0 + kk)]
                    sS[(sr, sk0 + kk, 0)] = BF16(pk * sigmoid(pk))
                    sDP[(sr, sk0 + kk, 0)] = BF16(dpk)
                    sDPt[(sk0 + kk, sr, 0)] = BF16(dpk * rho_r)
                if sk0 == 0:
                    sRho[sr] = rho_r
                    sB[sr] = rho_r * rho_r * c_r * self.inv_h
                cute.arch.fence_proxy("async.shared", space="cta")
                all_bar.arrive_and_wait()

                for jj in cutlass.range_constexpr(self.n_jj):
                    j = jj * 2 + w
                    ring.consumer_wait(cons)
                    st = cons.index
                    cute.nvgpu.warpgroup.fence()
                    cute.gemm(
                        mma_g,
                        acc,
                        tArS[(None, None, 0, 0)],
                        tBrGD[(None, None, 0, st)],
                        acc,
                    )
                    cute.gemm(
                        mma_g,
                        acc_u,
                        tArDP[(None, None, 0, 0)],
                        tBrGU[(None, None, 0, st)],
                        acc_u,
                    )
                    for kb in cutlass.range_constexpr(num_k_wg):
                        cute.gemm(
                            mma_wg,
                            acc_dw[jj],
                            tArX[(None, None, kb, st)],
                            tBrDPt[(None, None, kb, 0)],
                            acc_dw[jj],
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.copy(s2r, tSX[(None, None, None, st)], vX)
                    cute.copy(s2r, tSDY[(None, None, None, st)], vDY)
                    cute.nvgpu.warpgroup.wait_group(0)
                    for i in cutlass.range_constexpr(n_acc):
                        crd = tCc[i]
                        gt = sigmoid(acc[i])
                        rv = sRho[crd[0]]
                        xv = F32(rX[i])
                        wv = sW[j * BLK_H + crd[1]]
                        dyv = self.lgw * F32(rDY[i]) * gt + acc_u[i]
                        xh = rv * xv
                        dw[jj * n_col + (i % 2) + 2 * (i // 4)] += dyv * xh
                        acc[i] = rv * wv * dyv - sB[crd[0]] * xv
                    ring.consumer_release(cons)
                    cons.advance()
                    cons.advance()

                    r_out.store(acc.load().to(BF16))
                    buf = w * self.epi_stage + n_store % self.epi_stage
                    cute.copy(r2s, tRS_b, tRS_s[(None, None, None, buf)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=wg_bar_id, number_of_threads=WG)
                    if warp_idx % 4 == 0:
                        cute.copy(tma_o, tOs[(None, buf)], tOg[(None, rb, h_base + j)])
                        store_ring.producer_commit()
                        store_ring.producer_acquire()
                    cute.arch.barrier(barrier_id=wg_bar_id, number_of_threads=WG)
                    n_store += 1

            # ---- weight-grad partials of this row group ----
            for jj in cutlass.range_constexpr(self.n_jj):
                h0 = hc0 + (jj * 2 + w) * BLK_H
                for i in cutlass.range_constexpr(cute.size(acc_dw[jj])):
                    crd = tWc[i]
                    mWsGU[(g, h0 + crd[0], crd[1])] = (
                        acc_dw[jj][i] * sW[h0 - hc0 + crd[0]]
                    )
                # dw: sum the 8 row lanes of each warp, then the 4 warps through smem
                for q in cutlass.range_constexpr(n_col):
                    dv = dw[jj * n_col + q]
                    dv += cute.arch.shuffle_sync_bfly(dv, offset=4)
                    dv += cute.arch.shuffle_sync_bfly(dv, offset=8)
                    dv += cute.arch.shuffle_sync_bfly(dv, offset=16)
                    if lane < 4:
                        sDw[
                            (w * 4 + wt // 32) * BLK_H + 8 * (q // 2) + 2 * lane + q % 2
                        ] = dv
                cute.arch.barrier(barrier_id=wg_bar_id, number_of_threads=WG)
                if wt < BLK_H:
                    tot = F32(0.0)
                    for wp in cutlass.range_constexpr(4):
                        tot += sDw[(w * 4 + wp) * BLK_H + wt]
                    mWsW[(g, h0 + wt)] = tot
                cute.arch.barrier(barrier_id=wg_bar_id, number_of_threads=WG)
            store_ring.producer_tail()


# =============================================================================
# Wfin: weight grads = sum over row groups, written in the parameters' dtypes
# =============================================================================
class GnBwdWFin:
    def __init__(self, hidden, rank, g1, g2):
        self.H, self.R, self.G1, self.G2 = hidden, rank, g1, g2

    @cute.jit
    def __call__(
        self,
        mWsGD: cute.Tensor,
        mWsGU: cute.Tensor,
        mWsW: cute.Tensor,
        mDGD: cute.Tensor,
        mDGU: cute.Tensor,
        mDW: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mWsGD, mWsGU, mWsW, mDGD, mDGU, mDW).launch(
            grid=((self.H * self.R + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mWsGD: cute.Tensor,
        mWsGU: cute.Tensor,
        mWsW: cute.Tensor,
        mDGD: cute.Tensor,
        mDGU: cute.Tensor,
        mDW: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        idx = cutlass.Int64(bidx) * 256 + tidx
        if idx < self.H * self.R:
            h = idx // self.R
            k = idx % self.R
            a = F32(0.0)
            for g in range(self.G1):
                a += mWsGD[(g, h, k)]
            mDGD[(h, k)] = mDGD.element_type(a)
            b = F32(0.0)
            for g in range(self.G2):
                b += mWsGU[(g, h, k)]
            mDGU[(k, h)] = mDGU.element_type(b)
            if k == 0:
                c = F32(0.0)
                for g in range(self.G2):
                    c += mWsW[(g, h)]
                mDW[h] = mDW.element_type(c)


# =============================================================================
# host
# =============================================================================
B1_MAX_HC, B2_MAX_HC = 1024, 512
B1_STAGES, B2_STAGES = 8, 7


def pick_bwd_split(H, sms, max_hc):
    """Return H chunks and row groups that use at most the requested SM budget."""
    best: tuple[tuple[int, int], int, int] | None = None
    for n_hc in range(1, H // (2 * BLK_H) + 1):
        if (
            H % n_hc
            or (H // n_hc) % (2 * BLK_H)
            or H // n_hc > max_hc
            or sms // n_hc == 0
        ):
            continue
        key = (n_hc * (sms // n_hc), -n_hc)
        if best is None or key > best[0]:
            best = (key, n_hc, sms // n_hc)
    if best is None:
        raise ValueError(
            f"num_sms={sms} cannot schedule hidden size {H} with max H chunk {max_hc}"
        )
    return best[1], best[2]


def _bwd_plan(H, R, lgw, dev, grad_dtypes, sms):
    """Compile an M-independent backward plan for one SM budget."""
    key = ("bwd", H, R, lgw, dev.index, grad_dtypes, sms)
    plan = PLANS.get(key)
    if plan is None:
        n1, groups1 = pick_bwd_split(H, sms, B1_MAX_HC)
        n2, groups2 = pick_bwd_split(H, sms, B2_MAX_HC)
        m = cute.sym_int(width=64)
        b1 = compile_op(
            ("b1", H, R, n1, groups1, lgw),
            lambda: GnBwdDs(H, R, n1, groups1, lgw, B1_STAGES),
            (
                make_fake_compact_tensor(
                    BF16, (m, H), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    BF16, (m, H), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (H,), stride_order=(0,), assumed_align=16
                ),
                make_fake_compact_tensor(
                    BF16, (H, R), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    BF16, (R, H), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (m,), stride_order=(0,), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (m, R), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (n1, m, R), stride_order=(2, 1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (n1, m), stride_order=(1, 0), assumed_align=4
                ),
                make_fake_compact_tensor(
                    F32, (groups1, H, R), stride_order=(2, 1, 0), assumed_align=16
                ),
            ),
        )
        b1f = compile_op(
            ("b1f", R, n1),
            lambda: GnBwdFin(n1, R),
            (
                make_fake_compact_tensor(
                    F32, (n1, m, R), stride_order=(2, 1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (n1, m), stride_order=(1, 0), assumed_align=4
                ),
                make_fake_compact_tensor(
                    F32, (m, R), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (m, R), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (m,), stride_order=(0,), assumed_align=16
                ),
            ),
        )
        b2 = compile_op(
            ("b2", H, R, n2, groups2, lgw),
            lambda: GnBwdDx(H, R, n2, groups2, lgw, B2_STAGES),
            (
                make_fake_compact_tensor(
                    BF16, (m, H), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    BF16, (m, H), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (H,), stride_order=(0,), assumed_align=16
                ),
                make_fake_compact_tensor(
                    BF16, (H, R), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    BF16, (H, R), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (m,), stride_order=(0,), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (m, R), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (m, R), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (m,), stride_order=(0,), assumed_align=16
                ),
                make_fake_compact_tensor(
                    BF16, (m, H), stride_order=(1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (groups2, H, R), stride_order=(2, 1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (groups2, H), stride_order=(1, 0), assumed_align=4
                ),
            ),
        )
        tw, tu, td = (to_cute_dtype(dtype) for dtype in grad_dtypes)
        wfin = compile_op(
            ("wfin", H, R, groups1, groups2, grad_dtypes),
            lambda: GnBwdWFin(H, R, groups1, groups2),
            (
                make_fake_compact_tensor(
                    F32, (groups1, H, R), stride_order=(2, 1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (groups2, H, R), stride_order=(2, 1, 0), assumed_align=16
                ),
                make_fake_compact_tensor(
                    F32, (groups2, H), stride_order=(1, 0), assumed_align=4
                ),
                make_fake_compact_tensor(
                    td, (H, R), stride_order=(1, 0), assumed_align=4
                ),
                make_fake_compact_tensor(
                    tu, (R, H), stride_order=(1, 0), assumed_align=4
                ),
                make_fake_compact_tensor(tw, (H,), stride_order=(0,), assumed_align=4),
            ),
        )
        plan = (b1, b1f, b2, wfin, n1, groups1, groups2)
        PLANS[key] = plan
    return plan


def precompile_bwd(H, R, lgw, dev, grad_dtypes, *, num_sms=None):
    """Compile the M-independent backward kernels for this configuration."""
    sms = resolve_num_sms(dev, num_sms)
    _bwd_plan(H, R, lgw, dev, tuple(grad_dtypes), sms)


def gated_norm_bwd(
    dy,
    x,
    rho,
    p,
    w_f,
    gd_bf,
    gd_t,
    gu_t,
    lgw,
    grad_dtypes,
    *,
    num_sms=None,
):
    """Run backward using the forward's saved row statistics and prepared weights."""
    H = x.shape[-1]
    R = gd_bf.shape[1]
    x2d = x.reshape(-1, H).contiguous()
    dy2d = dy.reshape(-1, H).contiguous()
    M = x2d.shape[0]
    dev = x.device
    sms = resolve_num_sms(dev, num_sms)
    b1, b1f, b2, wfin, n1, groups1, groups2 = _bwd_plan(
        H,
        R,
        float(lgw),
        dev,
        tuple(grad_dtypes),
        sms,
    )
    shapes = [
        (n1, M, R),
        (n1, M),
        (groups1, H, R),
        (M, R),
        (M,),
        (groups2, H, R),
        (groups2, H),
    ]
    ds_part, c1_part, ws_gd, dp, c, ws_gu, ws_w = workspace(
        dev,
        shapes,
    )
    b1(x2d, dy2d, w_f, gd_bf, gd_t, rho, p, ds_part, c1_part, ws_gd)
    b1f(ds_part, c1_part, p, dp, c)
    dx = torch.empty_like(x2d)
    b2(x2d, dy2d, w_f, gd_bf, gu_t, rho, p, dp, c, dx, ws_gu, ws_w)
    dweight = torch.empty(H, device=dev, dtype=grad_dtypes[0])
    dgate_up = torch.empty(R, H, device=dev, dtype=grad_dtypes[1])
    dgate_down = torch.empty(H, R, device=dev, dtype=grad_dtypes[2])
    wfin(ws_gd, ws_gu, ws_w, dgate_down, dgate_up, dweight)
    return dx.view(x.shape), dweight, dgate_up, dgate_down
