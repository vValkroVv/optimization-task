# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Exact per-query-row statistics of causal document softmax attention.

flash-attn returns the log-partition ``LSE_i`` of every query row and nothing
else about the attention distribution. With it the distribution of row ``i`` is
known up to the logits: ``p_ij = exp(x_ij)``, ``x_ij = scale * q_i . k_j - LSE_i``
over the keys ``j`` of the row's document with ``j <= i``. ``x_ij <= 0`` up to
the rounding of ``LSE_i``, so ``-sum p x`` is a sum of non-negative terms and
has no cancellation.

``_row_stats_kernel`` recomputes ``q_i . k_j`` tile by tile -- the QK^T half of
the flash-attn forward, with no PV product -- and reduces it to the fields of
``RowField`` per (head, query row) in one pass over the causal key tiles. It
works in the log2 domain: ``t = x * log2(e)`` is one FMA from the raw dot
product and ``exp2`` is the hardware instruction. Per pair that is eight
instructions: the FMA, ``exp2``, three FMAs into the entropy, collision and
distance sums, one subtract for the distance, and a running max and min. Those
accumulators are elementwise ``[BLOCK_M, BLOCK_N]`` tiles that are reduced over
the columns once, after the last key tile, so the loop has no cross-lane
reductions. Key tiles that end before the local window of the first row of the
query tile need no causal mask and no window term, which is nearly all of
them; only the tiles around the diagonal take the masked path. The sink
probability (the first key of the document) is one extra dot product per row
instead of a per-pair select.

The same program then reduces its rows with at least ``MIN_KEYS`` allowed keys
to one partial record (sums, minima, maxima and the 32-bin histogram of the
normalized entropy), and ``_finish_kernel`` merges the partial records of a head
in a fixed order with float64 sums, so the records 174-176 come out of two
launches, deterministic, without a per-row round trip. The exact quantiles of
``entn`` (records 177-178) are selected by ``exact_quantiles`` from the one
fp32 value per (head, token) the kernel keeps.

Numerics: q.k accumulates in fp32 on bf16/fp16 inputs, the precision of the
flash-attn forward that produced ``LSE``, so the logits are exact to about
6e-8 * max |logit| and an entropy to a few times that. ``entn`` is ``ent / ln n``,
so its error is the entropy's divided by ``ln n``. Non-finite q, k or ``LSE``
give NaN entropy, probability, collision, distance and local-window fields (the
max logit may stay finite: the hardware max skips NaN operands). A NaN row makes
the sums of its head NaN; the minima and maxima ignore it; the histograms leave
non-finite entropies out and the quantile counts report them. The entropies of a
row may leave their mathematical range by a few fp32 ulps (``ent`` slightly
below 0, ``entn`` slightly above 1); they are not clamped. ``entn`` of a row
with one key is 0/0 and never used: such rows are excluded from every record.
"""

import enum
import math
from typing import NamedTuple, Optional, Sequence

import torch
import triton
import triton.language as tl
from metric_sampler.capture.kernels.triton_launch import launch as launch_kernel, launch_autotuned
from metric_sampler.capture.stats.helpers.quantiles import exact_quantiles, exact_quantiles_into


def _split_head_dim(head_dim: int):
    """Power-of-two dot chunks covering one attention head."""
    lo = 1 << (head_dim.bit_length() - 1)
    rest = head_dim - lo
    if rest == 0 and lo >= 16:
        return lo, 0, False
    if lo >= 16 and rest >= 16 and rest & (rest - 1) == 0:
        return lo, rest, False
    return max(16, triton.next_power_of_2(head_dim)), 0, True

__all__ = [
    "RECORD_FIELDS",
    "RowField",
    "RowStatsRecords",
    "attention_row_stats",
    "attention_row_values",
]

MIN_KEYS = tl.constexpr(32)
HIST_BINS = 32

# The fields of the attn/softmax/row_stats record (id 174), in its order after
# the row count; the partial record of a program has the same order.
RECORD_FIELDS = (
    "rows",
    "ent_sum",
    "ent_sumsq",
    "ent_min",
    "ent_max",
    "entn_sum",
    "entn_min",
    "entn_max",
    "pmax_sum",
    "pmax_min",
    "pmax_max",
    "pmin_sum",
    "lpmin_sum",
    "lpmin_min",
    "lpmin_max",
    "mlogit_sum",
    "mlogit_max",
    "coll_sum",
    "sink_sum",
    "sink_max",
    "dist_sum",
    "local_sum",
)
NUM_STATS = len(RECORD_FIELDS)
MIN_FIELDS = sum(
    1 << i for i, name in enumerate(RECORD_FIELDS) if name.endswith("_min")
)
MAX_FIELDS = sum(
    1 << i for i, name in enumerate(RECORD_FIELDS) if name.endswith("_max")
)
# A partial record is the stats, then the histogram, padded to a power of two.
PARTIAL_WIDTH = 64
# A partial record covers SLOT_ROWS rows of a document, whatever BLOCK_M is: every
# BLOCK_M is a multiple of SLOT_ROWS and divides MAX_BLOCK_M, a program writes the
# record of each group of SLOT_ROWS rows it owns, and a document has
# MAX_BLOCK_M-rounded many records. Every config therefore writes the same records
# wherever a row counts, so launches that share the buffers, like the autotuner's
# benchmark runs, leave nothing behind that the next one does not overwrite.
SLOT_ROWS = 64
MAX_BLOCK_M = 128
FINISH_BLOCK = 64

# Kernel globals must be constexpr.
LOCAL_WINDOW = tl.constexpr(128)
LOG2E = tl.constexpr(math.log2(math.e))
LN2 = tl.constexpr(math.log(2.0))


class RowField(enum.IntEnum):
    """Rows of the per-row tensor of ``attention_row_values``."""

    ENT = 0
    ENTN = 1
    PMAX = 2
    LPMIN = 3
    MLOGIT = 4
    SINK = 5
    PMIN = 6
    COLL = 7
    DIST = 8
    LOCAL = 9


class RowStatsRecords(NamedTuple):
    """The attn/softmax records 174-178, in that order.

    174 and 177 are float32, 175, 176 and 178 int32. 174 is ``[rows, 21 stats x
    H]`` stat-major (``RECORD_FIELDS``); 175 and 176 are ``HIST_BINS`` bins of
    ``entn`` over [0, 1] pooled over heads and per head (head-major); 177 and 178
    are the exact quantiles of ``entn`` pooled over heads and their counts, as
    ``exact_quantiles`` lays them out. Only rows with at least ``MIN_KEYS``
    allowed keys count. Without any, the sums are 0 and the minima and maxima
    are +inf and -inf, the identities of the merge across ranks and steps.
    """

    row_stats: torch.Tensor
    entropy_hist: torch.Tensor
    head_entropy_hist: torch.Tensor
    entropy_quantiles: torch.Tensor
    entropy_quantile_counts: torch.Tensor


class _Launch(NamedTuple):
    partials: torch.Tensor  # fp32 [H, slots, PARTIAL_WIDTH]
    entn: torch.Tensor  # fp32 [H, T]
    valid: torch.Tensor  # uint8 [H, T], 1 for the rows that count
    rows: Optional[torch.Tensor]  # fp32 [len(RowField), H, T]


def _configs():
    # ELEMENTWISE keeps five [BLOCK_M, BLOCK_N] accumulator tiles, that is
    # 5 * BLOCK_M * BLOCK_N / (32 * num_warps) registers per thread, and no
    # reduction in the key loop; it needs a small tile. Without it the loop
    # reduces over the columns of every tile, which leaves room for BLOCK_N 128.
    # BLOCK_M 64 with 4 warps is one warpgroup per program, small enough for two
    # or three programs per SM, whose tensor-core and elementwise phases then
    # overlap; BLOCK_M 128 with 8 warps does it with two warpgroups inside one
    # program, and BLOCK_M 64 with 8 warps splits the columns between them.
    # Every BLOCK_M is a multiple of SLOT_ROWS, and there is no BLOCK_M % BLOCK_N
    # constraint: every tile that reaches the diagonal is masked per element.
    configs = [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "ELEMENTWISE": elementwise},
            num_warps=w,
            num_stages=st,
        )
        for bm, bn, w, st, elementwise in [
            (64, 64, 4, 3, True),
            (64, 64, 4, 2, True),
            (64, 32, 4, 4, True),
            (64, 32, 4, 3, True),
            (64, 64, 8, 3, True),
            (128, 64, 8, 3, True),
            (128, 64, 8, 2, True),
            (128, 32, 8, 3, True),
            (64, 64, 4, 3, False),
            (64, 128, 4, 3, False),
            (128, 64, 8, 3, False),
            (128, 128, 8, 2, False),
        ]
    ]
    assert all(
        config.kwargs["BLOCK_M"] % SLOT_ROWS == 0
        and MAX_BLOCK_M % config.kwargs["BLOCK_M"] == 0
        for config in configs
    )
    return configs


@triton.jit
def _masked_sum(values, ok):
    return tl.sum(tl.where(ok, values, 0.0), axis=0)


@triton.jit
def _masked_min(values, ok):
    return tl.min(tl.where(ok, values, float("inf")), axis=0)


@triton.jit
def _masked_max(values, ok):
    return tl.max(tl.where(ok, values, float("-inf")), axis=0)


# Every config writes the same partial records wherever a row counts (see
# SLOT_ROWS), so the autotuner's benchmark runs on the same buffers leave nothing
# behind and it needs no reset_to_zero.
@triton.autotune(configs=_configs(), key=["DIM_A", "DIM_B", "STORE_ROWS", "VARLEN"])
@triton.jit
def _row_stats_kernel(
    Q,
    K,
    LSE,
    CU_SEQLENS,
    PARTIALS,
    ENTN,
    VALID,
    ROWS,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_ld,
    stride_lh,
    stride_ls,
    stride_rf,
    num_tokens,
    max_seqlen,
    slots_per_doc,
    softmax_scale,
    DIM_A: tl.constexpr,
    DIM_B: tl.constexpr,
    VARLEN: tl.constexpr,
    STORE_ROWS: tl.constexpr,
    ELEMENTWISE: tl.constexpr,
    NUM_STATS: tl.constexpr,
    HIST_BINS: tl.constexpr,
    PARTIAL_WIDTH: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per (q tile, document, head), as in `_max_qk_logit_kernel`.
    # Tile m scans m + 1 key tiles, so the longest tiles are launched first.
    m_blk = tl.num_programs(0) - 1 - tl.program_id(0)
    doc = tl.program_id(1)
    head = tl.program_id(2)

    if VARLEN:
        start = tl.load(CU_SEQLENS + doc)
        seqlen = tl.load(CU_SEQLENS + doc + 1) - start
    else:
        start = doc * max_seqlen
        seqlen = max_seqlen
    m_start = m_blk * BLOCK_M
    if m_start >= seqlen:
        return

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    row_ok = offs_m < seqlen
    offs_da = tl.arange(0, DIM_A)

    qk_scale = softmax_scale * LOG2E
    lse2 = (
        tl.load(
            LSE + doc * stride_ld + head * stride_lh + start * stride_ls + offs_m,
            mask=row_ok,
            other=0.0,
        )
        * LOG2E
    )

    q_base = Q + (start + offs_m)[:, None].to(tl.int64) * stride_qt + head * stride_qh
    k_base = K + (start + offs_n)[:, None].to(tl.int64) * stride_kt + head * stride_kh
    k_first = K + start.to(tl.int64) * stride_kt + head * stride_kh
    qa = tl.load(q_base + offs_da[None, :], mask=row_ok[:, None], other=0.0)
    first_logit = tl.sum(
        qa.to(tl.float32) * tl.load(k_first + offs_da).to(tl.float32)[None, :], axis=1
    )
    if DIM_B > 0:
        offs_db = DIM_A + tl.arange(0, DIM_B)
        qb = tl.load(q_base + offs_db[None, :], mask=row_ok[:, None], other=0.0)
        first_logit += tl.sum(
            qb.to(tl.float32) * tl.load(k_first + offs_db).to(tl.float32)[None, :],
            axis=1,
        )

    # Per row: sum p t, sum p^2, sum p (i - j) and the extremes of the raw logit
    # (monotone in x within a row). ELEMENTWISE accumulates them per element and
    # reduces over the columns after the last tile; otherwise every tile is
    # reduced. The window mass is only gathered by the few masked tiles.
    local_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    if ELEMENTWISE:
        ent2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        coll2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dist2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        smax2 = tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=tl.float32)
        smin2 = tl.full([BLOCK_M, BLOCK_N], float("inf"), dtype=tl.float32)
    else:
        ent_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        coll_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        dist_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        smax = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        smin = tl.full([BLOCK_M], float("inf"), dtype=tl.float32)
    rows_f = offs_m.to(tl.float32)
    cols_f = offs_n.to(tl.float32)

    # Key tiles entirely before the window of the first row of the tile: every
    # pair is allowed and outside the local window, so there is nothing to mask.
    far_end = tl.maximum(m_start - (LOCAL_WINDOW - 1), 0) // BLOCK_N * BLOCK_N
    for n_start in range(0, far_end, BLOCK_N):
        k_ptrs = k_base + n_start * stride_kt
        s = tl.dot(qa, tl.trans(tl.load(k_ptrs + offs_da[None, :])))
        if DIM_B > 0:
            s = tl.dot(qb, tl.trans(tl.load(k_ptrs + offs_db[None, :])), s)
        t = s * qk_scale - lse2[:, None]
        p = tl.exp2(t)
        dist = (rows_f - n_start)[:, None] - cols_f[None, :]
        if ELEMENTWISE:
            ent2 += p * t
            coll2 += p * p
            dist2 += p * dist
            smax2 = tl.maximum(smax2, s)
            smin2 = tl.minimum(smin2, s)
        else:
            ent_acc += tl.sum(p * t, axis=1)
            coll_acc += tl.sum(p * p, axis=1)
            dist_acc += tl.sum(p * dist, axis=1)
            smax = tl.maximum(smax, tl.max(s, axis=1))
            smin = tl.minimum(smin, tl.min(s, axis=1))

    # Window and diagonal tiles; keys past the document end only exist in the
    # last of them and belong to no allowed pair.
    for n_start in range(far_end, tl.minimum(m_start + BLOCK_M, seqlen), BLOCK_N):
        cols = n_start + offs_n
        col_ok = (cols < seqlen)[:, None]
        k_ptrs = k_base + n_start * stride_kt
        s = tl.dot(
            qa, tl.trans(tl.load(k_ptrs + offs_da[None, :], mask=col_ok, other=0.0))
        )
        if DIM_B > 0:
            kb = tl.load(k_ptrs + offs_db[None, :], mask=col_ok, other=0.0)
            s = tl.dot(qb, tl.trans(kb), s)
        allowed = cols[None, :] <= offs_m[:, None]
        in_window = cols[None, :] > offs_m[:, None] - LOCAL_WINDOW
        t = s * qk_scale - lse2[:, None]
        p = tl.where(allowed, tl.exp2(t), 0.0)
        dist = (rows_f - n_start)[:, None] - cols_f[None, :]
        local_acc += tl.sum(tl.where(in_window, p, 0.0), axis=1)
        s_high = tl.where(allowed, s, float("-inf"))
        s_low = tl.where(allowed, s, float("inf"))
        if ELEMENTWISE:
            ent2 += p * t
            coll2 += p * p
            dist2 += p * dist
            smax2 = tl.maximum(smax2, s_high)
            smin2 = tl.minimum(smin2, s_low)
        else:
            ent_acc += tl.sum(p * t, axis=1)
            coll_acc += tl.sum(p * p, axis=1)
            dist_acc += tl.sum(p * dist, axis=1)
            smax = tl.maximum(smax, tl.max(s_high, axis=1))
            smin = tl.minimum(smin, tl.min(s_low, axis=1))

    if ELEMENTWISE:
        ent_acc = tl.sum(ent2, axis=1)
        coll_acc = tl.sum(coll2, axis=1)
        dist_acc = tl.sum(dist2, axis=1)
        smax = tl.max(smax2, axis=1)
        smin = tl.min(smin2, axis=1)
    t_max = smax * qk_scale - lse2
    t_min = smin * qk_scale - lse2
    ent = -ent_acc * LN2
    entn = -ent_acc / tl.log2((offs_m + 1).to(tl.float32))
    pmax = tl.exp2(t_max)
    pmin = tl.exp2(t_min)
    lpmin = t_min * LN2
    mlogit = smax * softmax_scale
    sink = tl.exp2(first_logit * qk_scale - lse2)

    token = head * num_tokens + start + offs_m
    ok = row_ok & (offs_m + 1 >= MIN_KEYS)
    tl.store(ENTN + token, entn, mask=row_ok)
    tl.store(VALID + token, ok.to(tl.uint8), mask=row_ok)
    if STORE_ROWS:
        # Field order is `RowField`.
        rows = ROWS + head * num_tokens + start + offs_m
        tl.store(rows + 0 * stride_rf, ent, mask=row_ok)
        tl.store(rows + 1 * stride_rf, entn, mask=row_ok)
        tl.store(rows + 2 * stride_rf, pmax, mask=row_ok)
        tl.store(rows + 3 * stride_rf, lpmin, mask=row_ok)
        tl.store(rows + 4 * stride_rf, mlogit, mask=row_ok)
        tl.store(rows + 5 * stride_rf, sink, mask=row_ok)
        tl.store(rows + 6 * stride_rf, pmin, mask=row_ok)
        tl.store(rows + 7 * stride_rf, coll_acc, mask=row_ok)
        tl.store(rows + 8 * stride_rf, dist_acc, mask=row_ok)
        tl.store(rows + 9 * stride_rf, local_acc, mask=row_ok)

    # One partial record per group of SLOT_ROWS rows, in `RECORD_FIELDS` order,
    # then the histogram.
    row_group = tl.arange(0, BLOCK_M) // SLOT_ROWS
    for group in tl.static_range(BLOCK_M // SLOT_ROWS):
        counted_rows = ok & (row_group == group)
        slot = doc * slots_per_doc + m_start // SLOT_ROWS + group
        part = (
            PARTIALS
            + (head * tl.num_programs(1) * slots_per_doc + slot) * PARTIAL_WIDTH
        )
        tl.store(part + 0, tl.sum(counted_rows.to(tl.float32), axis=0))
        tl.store(part + 1, _masked_sum(ent, counted_rows))
        tl.store(part + 2, _masked_sum(ent * ent, counted_rows))
        tl.store(part + 3, _masked_min(ent, counted_rows))
        tl.store(part + 4, _masked_max(ent, counted_rows))
        tl.store(part + 5, _masked_sum(entn, counted_rows))
        tl.store(part + 6, _masked_min(entn, counted_rows))
        tl.store(part + 7, _masked_max(entn, counted_rows))
        tl.store(part + 8, _masked_sum(pmax, counted_rows))
        tl.store(part + 9, _masked_min(pmax, counted_rows))
        tl.store(part + 10, _masked_max(pmax, counted_rows))
        tl.store(part + 11, _masked_sum(pmin, counted_rows))
        tl.store(part + 12, _masked_sum(lpmin, counted_rows))
        tl.store(part + 13, _masked_min(lpmin, counted_rows))
        tl.store(part + 14, _masked_max(lpmin, counted_rows))
        tl.store(part + 15, _masked_sum(mlogit, counted_rows))
        tl.store(part + 16, _masked_max(mlogit, counted_rows))
        tl.store(part + 17, _masked_sum(coll_acc, counted_rows))
        tl.store(part + 18, _masked_sum(sink, counted_rows))
        tl.store(part + 19, _masked_max(sink, counted_rows))
        tl.store(part + 20, _masked_sum(dist_acc, counted_rows))
        tl.store(part + 21, _masked_sum(local_acc, counted_rows))
        # A non-finite entropy gets no bin, but its bin index must be computable.
        binned = counted_rows & (entn == entn) & (tl.abs(entn) != float("inf"))
        bins = tl.minimum(
            tl.maximum(tl.where(binned, entn, 0.0) * HIST_BINS, 0.0), HIST_BINS - 1.0
        ).to(tl.int32)
        bin_ids = tl.arange(0, HIST_BINS)
        hist = tl.sum(
            ((bins[:, None] == bin_ids[None, :]) & binned[:, None]).to(tl.int32), axis=0
        )
        tl.store(part + NUM_STATS + bin_ids, hist.to(tl.float32))


@triton.jit
def _finish_kernel(
    PARTIALS,
    STATS,
    HEAD_HIST,
    POOLED_HIST,
    num_slots,
    num_heads,
    MIN_FIELDS: tl.constexpr,
    MAX_FIELDS: tl.constexpr,
    NUM_STATS: tl.constexpr,
    HIST_BINS: tl.constexpr,
    PARTIAL_WIDTH: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Merges the partial records of one head. Slots no program wrote, or whose
    # rows all fell short of MIN_KEYS, hold a zero row count and are left out of
    # the minima and maxima. The loop only accumulates per slot lane; the lanes
    # are reduced once, in a fixed order.
    head = tl.program_id(0)
    fields = tl.arange(0, PARTIAL_WIDTH)
    base = PARTIALS + head.to(tl.int64) * num_slots * PARTIAL_WIDTH
    total = tl.zeros([BLOCK_P, PARTIAL_WIDTH], dtype=tl.float64)
    lowest = tl.full([BLOCK_P, PARTIAL_WIDTH], float("inf"), dtype=tl.float32)
    highest = tl.full([BLOCK_P, PARTIAL_WIDTH], float("-inf"), dtype=tl.float32)
    for first in range(0, num_slots, BLOCK_P):
        slot = first + tl.arange(0, BLOCK_P)
        in_range = slot < num_slots
        count = tl.load(base + slot * PARTIAL_WIDTH, mask=in_range, other=0.0)
        record = tl.load(
            base + slot[:, None] * PARTIAL_WIDTH + fields[None, :],
            mask=in_range[:, None],
            other=0.0,
        )
        counted = (count > 0)[:, None]
        total += record.to(tl.float64)
        lowest = tl.minimum(lowest, tl.where(counted, record, float("inf")))
        highest = tl.maximum(highest, tl.where(counted, record, float("-inf")))
    total = tl.sum(total, axis=0)
    lowest = tl.min(lowest, axis=0)
    highest = tl.max(highest, axis=0)

    is_min = ((tl.full([PARTIAL_WIDTH], MIN_FIELDS, tl.int32) >> fields) & 1) == 1
    is_max = ((tl.full([PARTIAL_WIDTH], MAX_FIELDS, tl.int32) >> fields) & 1) == 1
    stats = tl.where(is_min, lowest, tl.where(is_max, highest, total.to(tl.float32)))
    # The row count is the same for every head; the stats are stat-major.
    destination = tl.where(fields == 0, 0, 1 + (fields - 1) * num_heads + head)
    tl.store(STATS + destination, stats, mask=fields < NUM_STATS)
    hist = total.to(tl.int32)
    in_hist = (fields >= NUM_STATS) & (fields < NUM_STATS + HIST_BINS)
    tl.store(HEAD_HIST + (head * HIST_BINS + fields - NUM_STATS), hist, mask=in_hist)
    tl.atomic_add(POOLED_HIST + (fields - NUM_STATS), hist, mask=in_hist)


def _launch(q, k, lse, softmax_scale, cu_seqlens, max_seqlen, store_rows, configs=None):
    """Runs the row kernel; `configs` (tests) launches each given config, in order,
    on the same buffers, as the autotuner does when it benchmarks them."""
    assert q.dim() == 4 and q.shape == k.shape, (q.shape, k.shape)
    assert q.dtype == k.dtype and q.dtype in (torch.bfloat16, torch.float16), q.dtype
    assert lse.dtype == torch.float32, lse.dtype
    bsz, seq_len, num_heads, head_dim = q.shape
    dim_a, dim_b, mask_d = _split_head_dim(head_dim)
    assert not mask_d, f"head_dim {head_dim} is not a sum of two powers of two"
    num_tokens = bsz * seq_len
    if lse.stride(-1) != 1:
        lse = lse.contiguous()
    varlen = cu_seqlens is not None
    if varlen:
        assert lse.shape == (num_heads, num_tokens), (lse.shape, q.shape)
        lse_strides = (0, lse.stride(0), lse.stride(1))
        num_docs = cu_seqlens.numel() - 1
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32)
        max_seqlen = int(max_seqlen)
    else:
        assert lse.dim() == 3 and lse.shape[:2] == (bsz, num_heads), lse.shape
        assert lse.shape[2] >= seq_len, (lse.shape, q.shape)
        lse_strides = (lse.stride(0), lse.stride(1), 0)
        num_docs = bsz
        max_seqlen = seq_len
        cu_seqlens = lse  # unused without VARLEN
    q, k = q.flatten(0, 1), k.flatten(0, 1)
    # The kernel indexes the last dim with unit stride.
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()

    slots_per_doc = triton.cdiv(max_seqlen, MAX_BLOCK_M) * (MAX_BLOCK_M // SLOT_ROWS)
    # The partial records and the validity bytes, zero-filled in one launch: a
    # record no program writes counts no rows.
    partial_bytes = 4 * num_heads * num_docs * slots_per_doc * PARTIAL_WIDTH
    workspace = torch.zeros(
        partial_bytes + num_heads * num_tokens, device=q.device, dtype=torch.uint8
    )
    partials = workspace[:partial_bytes].view(torch.float32)
    partials = partials.view(num_heads, num_docs * slots_per_doc, PARTIAL_WIDTH)
    valid = workspace[partial_bytes:].view(num_heads, num_tokens)
    entn = torch.empty((num_heads, num_tokens), device=q.device, dtype=torch.float32)
    rows = None
    if store_rows:
        rows = torch.empty(
            (len(RowField), num_heads, num_tokens), device=q.device, dtype=torch.float32
        )
    def grid(meta):
        return (
            triton.cdiv(max_seqlen, meta["BLOCK_M"]),
            num_docs,
            num_heads,
        )
    args = (
        q,
        k,
        lse,
        cu_seqlens,
        partials,
        entn,
        valid,
        # The per-row fields are only written with STORE_ROWS.
        entn if rows is None else rows,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        *lse_strides,
        num_heads * num_tokens,
        num_tokens,
        max_seqlen,
        slots_per_doc,
        float(softmax_scale),
    )
    constants = dict(
        DIM_A=dim_a,
        DIM_B=dim_b,
        VARLEN=varlen,
        STORE_ROWS=store_rows,
        NUM_STATS=NUM_STATS,
        HIST_BINS=HIST_BINS,
        PARTIAL_WIDTH=PARTIAL_WIDTH,
        SLOT_ROWS=SLOT_ROWS,
    )
    if configs is None:
        launch_autotuned(_row_stats_kernel, grid, *args, **constants)
    else:
        for config in configs:
            _row_stats_kernel.fn[grid](
                *args,
                **constants,
                **config.kwargs,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
    return _Launch(partials, entn, valid, rows)


def _records(launch, out=None):
    """The merge of the partial records and the exact quantiles of a launch, written into `out` (see
    ``attention_row_stats``) if given."""
    device = launch.partials.device
    num_heads, num_slots = launch.partials.shape[:2]
    if out is None:
        stats = torch.empty(
            1 + (NUM_STATS - 1) * num_heads, device=device, dtype=torch.float32
        )
        # One extra zeroed row of bins pools the heads.
        hist = torch.zeros((num_heads + 1, HIST_BINS), device=device, dtype=torch.int32)
        pooled_hist, head_hist = hist[0], hist[1:]
    else:
        stats, pooled_hist, head_hist, quantiles, quantile_counts = out
        if head_hist is None:
            head_hist = torch.empty(num_heads * HIST_BINS, device=device, dtype=torch.int32)
    launch_kernel(
        _finish_kernel,
        (num_heads, 1, 1),
        launch.partials,
        stats,
        head_hist,
        pooled_hist,
        num_slots,
        num_heads,
        MIN_FIELDS=MIN_FIELDS,
        MAX_FIELDS=MAX_FIELDS,
        NUM_STATS=NUM_STATS,
        HIST_BINS=HIST_BINS,
        PARTIAL_WIDTH=PARTIAL_WIDTH,
        BLOCK_P=FINISH_BLOCK,
        num_warps=4,
    )
    valid = launch.valid.view(torch.bool)
    if out is None:
        quantiles, quantile_counts = exact_quantiles(launch.entn, valid)
        return RowStatsRecords(
            stats, pooled_hist, head_hist.reshape(-1), quantiles, quantile_counts
        )
    exact_quantiles_into(launch.entn, valid, quantiles, quantile_counts)
    return RowStatsRecords(*out)


@torch.no_grad()
def attention_row_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    out: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> RowStatsRecords:
    """The attn/softmax records 174-178 of causal attention, the mask of the MLA flash-attn calls.

    Args:
        q, k: ``[bsz, seq_len, num_heads, head_dim]`` after RoPE, bf16 or fp16,
            equal shapes. ``head_dim`` is a power of two or a sum of two (MLA:
            192 = 128 nope + 64 rope).
        lse: fp32 flash-attn log-partition of the same call, natural log. Dense
            (no ``cu_seqlens``): ``[bsz, num_heads, seq_len]``, which may be
            padded along the last dim as flash-attn 2 does. With ``cu_seqlens``:
            ``[num_heads, bsz * seq_len]``.
        cu_seqlens, max_seqlen: document boundaries over the flattened
            ``bsz * seq_len`` tokens, as for ``flash_attn_varlen_func``;
            ``max_seqlen`` should be an int (a CUDA tensor costs a sync), and may
            exceed the longest document, which only adds idle programs.
        out: the five tensors of the records, in their order, to write them
            into instead of new ones: ``entropy_hist`` zeros (the kernel adds
            to it), the others of the dtype and size the records have, and
            ``head_entropy_hist`` None where it is not wanted.

    Returns:
        ``RowStatsRecords``, which holds the tensors of ``out`` if given.
        Launches: two zero fills (one with ``out``), the row kernel, the
        merge kernel, and ``exact_quantiles``, which costs one call of
        metric_sampler's prebuilt extension, or its several Triton launches
        without it. All of it is stream-ordered with no host sync, so a CUDA
        graph can capture it once the kernel is tuned, for a fixed
        ``max_seqlen``.
    """
    with torch.cuda.device(q.device):
        launch = _launch(q, k, lse, softmax_scale, cu_seqlens, max_seqlen, False)
        return _records(launch, out)


@torch.no_grad()
def attention_row_values(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """The per-row statistics behind ``attention_row_stats``: fp32 ``[len(RowField), H, T]``.

    For tests and debugging; the records are reduced inside the kernel and do
    not go through this tensor. ``entn`` is undefined for rows with one key.
    """
    with torch.cuda.device(q.device):
        return _launch(q, k, lse, softmax_scale, cu_seqlens, max_seqlen, True).rows
