from __future__ import annotations

import torch
import triton
import triton.language as tl

from metric_sampler.contract.metrics import HistogramBins

# The rows of a tile follow from this and the row width; it keeps the tile within the register budget.
_TILE_ELEMENTS = 4096
# Enough programs to fill the GPU several times over while keeping histogram flushes (one vector
# atomic per program) rare.
_TARGET_PROGRAMS = 1024
_MIN_WARPS = 2
_MAX_WARPS = 8
_COLUMNS_PER_WARP = 512


def _num_warps(block_hidden: int) -> int:
    return max(_MIN_WARPS, min(_MAX_WARPS, block_hidden // _COLUMNS_PER_WARP))


@triton.jit
def _floor_log2_bin(values, bins_start, BINS: tl.constexpr):
    # floor(log2|v|) read directly from the fp32 exponent field; zeros/denormals land in bin 0.
    exponent = ((values.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
    return tl.minimum(tl.maximum(exponent - bins_start, 0), BINS - 1)


# The row count changes every step for MoE points, so what follows from it is a runtime argument, not a constexpr.
@triton.jit(
    do_not_specialize=[
        "moments_offset",
        "counts_offset",
        "amax_offset",
        "power_sums_offset",
        "rows",
        "tiles_per_program",
    ]
)
def _activation_stats_kernel(
    x_ptr,
    row_stride,
    buf_ptr,
    moments_offset,
    counts_offset,
    amax_offset,
    power_sums_offset,
    rows,
    h,
    bins_start,
    tiles_per_program,
    BINS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Remove padding in integer accumulators: separate float32 add/subtract atomics can lose
    # genuine bin-0 counts when many programs temporarily add a large amount of padding.
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < h
    tile_rows = tl.arange(0, TILE_ROWS)

    sum_acc = 0.0
    sumsq_acc = 0.0
    cube_acc = 0.0
    fourth_acc = 0.0
    nonfinite_acc = 0
    finite_acc = 0
    amax_acc = 0.0
    excluded_elements = 0
    bins = tl.arange(0, BINS)
    exp_hist_acc = tl.zeros([BINS], dtype=tl.int32)
    rms_hist_acc = tl.zeros([BINS], dtype=tl.int32)

    for tile in range(tiles_per_program):
        row_ids = (tl.program_id(axis=0) * tiles_per_program + tile) * TILE_ROWS + tile_rows
        row_mask = row_ids < rows
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(
            x_ptr + row_ids[:, None].to(tl.int64) * row_stride + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)

        abs_x = tl.abs(x)
        # NaN compares false and +-Inf is not below Inf, so both are excluded here.
        finite = mask & (abs_x < float("inf"))
        zeroed = tl.where(finite, x, 0.0)
        row_sumsq = tl.sum(zeroed * zeroed, axis=1)
        row_count = tl.sum(finite.to(tl.int32), axis=1)
        tile_finite = tl.sum(row_count, axis=0)
        sum_acc += tl.sum(tl.sum(zeroed, axis=1), axis=0)
        sumsq_acc += tl.sum(row_sumsq, axis=0)
        squares = zeroed * zeroed
        cube_acc += tl.sum(tl.sum(squares * zeroed, axis=1), axis=0)
        fourth_acc += tl.sum(tl.sum(squares * squares, axis=1), axis=0)
        finite_acc += tile_finite
        nonfinite_acc += tl.sum(tl.sum((mask & ~finite).to(tl.int32), axis=1), axis=0)
        amax_acc = tl.maximum(amax_acc, tl.max(tl.max(tl.abs(zeroed), axis=1), axis=0))

        exp_bin = tl.where(finite, _floor_log2_bin(zeroed, bins_start, BINS), 0)
        exp_hist_acc += tl.histogram(tl.reshape(exp_bin, [TILE_ROWS * BLOCK_H]), BINS)
        excluded_elements += TILE_ROWS * BLOCK_H - tile_finite

        has_finite = row_mask & (row_count > 0)
        row_rms = tl.sqrt(row_sumsq / tl.maximum(row_count, 1).to(tl.float32))
        rms_bin = _floor_log2_bin(row_rms, bins_start, BINS)
        # Count valid rows directly; histogram-then-padding-subtraction failed the
        # partial-tile and narrow-row cases on Triton 3.2.
        rms_hist_acc += tl.sum(
            (has_finite[:, None] & (rms_bin[:, None] == bins[None, :])).to(tl.int32), axis=0
        )

    moments_ptr = buf_ptr + moments_offset
    tl.atomic_add(moments_ptr, sum_acc)
    tl.atomic_add(moments_ptr + 1, sumsq_acc)
    tl.atomic_add(buf_ptr + power_sums_offset, cube_acc)
    tl.atomic_add(buf_ptr + power_sums_offset + 1, fourth_acc)
    counts_ptr = (buf_ptr + counts_offset).to(tl.pointer_type(tl.int32))
    tl.atomic_add(counts_ptr, nonfinite_acc)
    tl.atomic_add(counts_ptr + 1, finite_acc)
    tl.atomic_max(buf_ptr + amax_offset, amax_acc)
    tl.atomic_add(counts_ptr + 2 + BINS + bins, rms_hist_acc)
    exp_hist_acc -= tl.where(bins == 0, excluded_elements, 0)
    tl.atomic_add(counts_ptr + 2 + bins, exp_hist_acc)


def launch_activation_stats_kernel(
    activations: torch.Tensor,
    buffer: torch.Tensor,
    moments_offset: int,
    counts_offset: int,
    amax_offset: int,
    power_sums_offset: int,
    bins: HistogramBins,
    tile_elements: int = _TILE_ELEMENTS,
    num_warps: int | None = None,
) -> None:
    """Folds the rows of `activations` [rows, hidden] into the slots at the given offsets of `buffer`, a
    float32 view of the shared word buffer; counts use an int32 view of their slot. `tile_elements` and
    `num_warps` exist only for tuning sweeps."""
    if bins.width != 1.0 or bins.start != int(bins.start):
        raise ValueError("the kernel reads floor(log2|x|) from the exponent field: unit-width integer grid only")
    if activations.stride(1) != 1:
        activations = activations.contiguous()
    rows, hidden = activations.shape
    block_hidden = triton.next_power_of_2(hidden)
    tile_rows = max(1, tile_elements // block_hidden)
    tiles = triton.cdiv(rows, tile_rows)
    tiles_per_program = max(1, triton.cdiv(tiles, _TARGET_PROGRAMS))
    grid = (triton.cdiv(tiles, tiles_per_program),)
    _activation_stats_kernel[grid](
        activations,
        activations.stride(0),
        buffer,
        moments_offset,
        counts_offset,
        amax_offset,
        power_sums_offset,
        rows,
        hidden,
        int(bins.start),
        tiles_per_program,
        BINS=bins.count,
        TILE_ROWS=tile_rows,
        BLOCK_H=block_hidden,
        num_warps=_num_warps(block_hidden) if num_warps is None else num_warps,
    )
