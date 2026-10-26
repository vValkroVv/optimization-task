from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from metric_sampler.capture.kernels.triton_launch import launch
from metric_sampler.contract.metrics import (
    ENTROPY_TOKEN_STRIDE,
    EXPERT_LOSS_BINS,
    LOSS_POSITION_BUCKETS,
    ROUTER_ENTROPY_BINS,
    TOKEN_LOSS_BINS,
)

# Every kernel writes into the recorder's flat step buffer through two pointers to the same memory, `buf_i32_ptr` as
# int32 and `buf_f32_ptr` as float32; each slot is written through the one that matches its value type.

# A multiple of ENTROPY_TOKEN_STRIDE so a tile's strided rows stay aligned to absolute row index
# regardless of which tile a program owns.
# Small/wide score matrices favor a compact atomic tile; long/narrow matrices benefit
# from aggregating repeated expert assignments before global updates (H100 stand).
_ROUTING_TILE_ROWS = 4
_ROUTING_HISTOGRAM_TILE_ROWS = 16
_ROUTING_HISTOGRAM_MIN_TOKENS = 4096
_ROUTING_HISTOGRAM_MAX_EXPERTS = 64
_TOKEN_LOSS_TILE = 1024
_EXPERT_LOSS_TILE = 1024


@triton.jit
def _bin_index_tl(value, start, inv_width, BINS: tl.constexpr):
    # value is never negative here (a loss or an entropy), so truncating towards zero equals floor.
    idx = ((value - start) * inv_width).to(tl.int32)
    return tl.minimum(tl.maximum(idx, 0), BINS - 1)


@triton.jit(do_not_specialize=["position", "entropy_hist_offset", "counts_offset", "stream_counts_offset", "tokens"])
def _routing_kernel(
    topk_idx_ptr,
    topk_idx_row_stride,
    scores_ptr,
    scores_row_stride,
    topk_out_ptr,
    topk_out_layer_stride,
    topk_out_row_stride,
    position,
    buf_i32_ptr,
    entropy_hist_offset,
    counts_offset,
    stream_counts_offset,
    stream_ids_ptr,
    tokens_per_sample,
    tokens,
    streams,
    experts,
    top_k,
    bins_start,
    bins_inv_width,
    BINS: tl.constexpr,
    STRIDE: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_ids = tl.program_id(axis=0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    row_mask = row_ids < tokens

    k_cols = tl.arange(0, BLOCK_K)
    k_mask = k_cols < top_k
    mask2d = row_mask[:, None] & k_mask[None, :]
    topk_vals = tl.load(
        topk_idx_ptr + row_ids[:, None].to(tl.int64) * topk_idx_row_stride + k_cols[None, :], mask=mask2d, other=0
    ).to(tl.int32)
    tl.store(
        topk_out_ptr + position.to(tl.int64) * topk_out_layer_stride + row_ids[:, None].to(tl.int64) * topk_out_row_stride
        + k_cols[None, :],
        topk_vals,
        mask=mask2d,
    )
    e_cols = tl.arange(0, BLOCK_E)
    e_mask = e_cols < experts

    if TILE_ROWS == 4:
        tl.atomic_add(buf_i32_ptr + counts_offset + topk_vals, 1, mask=mask2d)
        if streams == 1:
            atomic_streams = tl.full((TILE_ROWS,), 0, tl.int32)
        else:
            atomic_samples = row_ids // tokens_per_sample
            atomic_streams = tl.load(stream_ids_ptr + atomic_samples, mask=row_mask, other=0).to(tl.int32)
        atomic_offsets = topk_vals * streams + atomic_streams[:, None]
        valid_atomic_streams = (atomic_streams >= 0) & (atomic_streams < streams)
        tl.atomic_add(
            buf_i32_ptr + stream_counts_offset + atomic_offsets,
            1,
            mask=mask2d & valid_atomic_streams[:, None],
        )
    else:
        # Aggregate assignments by expert before updating the global counters. Masked entries
        # temporarily occupy bin zero, so remove them from that bin before the atomic update.
        flat_expert = tl.reshape(tl.where(mask2d, topk_vals, 0), (TILE_ROWS * BLOCK_K,))
        expert_histogram = tl.histogram(flat_expert, BLOCK_E)
        valid_assignments = tl.sum(tl.sum(mask2d.to(tl.int32), axis=1), axis=0)
        invalid_assignments = TILE_ROWS * BLOCK_K - valid_assignments
        expert_counts = expert_histogram - tl.where(e_cols == 0, invalid_assignments, 0)
        tl.atomic_add(
            buf_i32_ptr + counts_offset + e_cols,
            expert_counts,
            mask=e_mask & (expert_counts > 0),
        )

        if streams == 1:
            tl.atomic_add(
                buf_i32_ptr + stream_counts_offset + e_cols,
                expert_counts,
                mask=e_mask & (expert_counts > 0),
            )
        else:
            samples = row_ids // tokens_per_sample
            first_sample = tl.min(tl.where(row_mask, samples, 2147483647), axis=0)
            last_sample = tl.max(tl.where(row_mask, samples, -1), axis=0)
            if first_sample == last_sample:
                sample_stream = tl.load(stream_ids_ptr + first_sample).to(tl.int32)
                expert_stream_offsets = e_cols * streams + sample_stream
                valid_sample_stream = (sample_stream >= 0) & (sample_stream < streams)
                tl.atomic_add(
                    buf_i32_ptr + stream_counts_offset + expert_stream_offsets,
                    expert_counts,
                    mask=e_mask & (expert_counts > 0) & valid_sample_stream,
                )
            else:
                row_streams = tl.load(stream_ids_ptr + samples, mask=row_mask, other=0).to(tl.int32)
                assignment_offsets = topk_vals * streams + row_streams[:, None]
                valid_row_streams = (row_streams >= 0) & (row_streams < streams)
                tl.atomic_add(
                    buf_i32_ptr + stream_counts_offset + assignment_offsets,
                    1,
                    mask=mask2d & valid_row_streams[:, None],
                )

    entropy_rows = tl.program_id(axis=0) * TILE_ROWS + tl.arange(0, TILE_ROWS // STRIDE) * STRIDE
    entropy_mask = entropy_rows < tokens
    scores = tl.load(
        scores_ptr + entropy_rows[:, None].to(tl.int64) * scores_row_stride + e_cols[None, :],
        mask=entropy_mask[:, None] & e_mask[None, :],
        other=0.0,
    ).to(tl.float64)

    total = tl.sum(scores, axis=1)
    probabilities = scores / total[:, None]
    weighted = tl.sum(tl.where(probabilities > 0, probabilities * libdevice.log(probabilities), 0.0), axis=1)
    entropy = -weighted / libdevice.log(experts.to(tl.float64))
    # Not tl.histogram: a row reduction's result is replicated across threads and would be counted twice.
    bin_index = _bin_index_tl(entropy, bins_start, bins_inv_width, BINS)
    tl.atomic_add(buf_i32_ptr + entropy_hist_offset + bin_index, 1, mask=entropy_mask)


def launch_routing_kernel(
    topk_indices: torch.Tensor,
    scores: torch.Tensor,
    topk_buffer: torch.Tensor,
    position: int,
    buffer_int32: torch.Tensor,
    entropy_hist_offset: int,
    counts_offset: int,
    stream_counts_offset: int,
    stream_ids: torch.Tensor,
    tokens_per_sample: int,
    streams: int,
    experts: int,
    tile_rows: int | None = None,
) -> None:
    """Records top-k experts, all-row per-stream expert counts, and the score entropy histogram."""
    tokens, top_k = topk_indices.shape
    if tile_rows is None:
        tile_rows = (
            _ROUTING_HISTOGRAM_TILE_ROWS
            if tokens >= _ROUTING_HISTOGRAM_MIN_TOKENS and experts <= _ROUTING_HISTOGRAM_MAX_EXPERTS
            else _ROUTING_TILE_ROWS
        )
    launch(
        _routing_kernel,
        (triton.cdiv(tokens, tile_rows), 1, 1),
        topk_indices,
        topk_indices.stride(0),
        scores,
        scores.stride(0),
        topk_buffer,
        topk_buffer.stride(0),
        topk_buffer.stride(1),
        position,
        buffer_int32,
        entropy_hist_offset,
        counts_offset,
        stream_counts_offset,
        stream_ids,
        tokens_per_sample,
        tokens,
        streams,
        experts,
        top_k,
        float(ROUTER_ENTROPY_BINS.start),
        1.0 / ROUTER_ENTROPY_BINS.width,
        BINS=ROUTER_ENTROPY_BINS.count,
        STRIDE=ENTROPY_TOKEN_STRIDE,
        TILE_ROWS=tile_rows,
        BLOCK_E=triton.next_power_of_2(experts),
        BLOCK_K=triton.next_power_of_2(top_k),
    )


@triton.jit(
    do_not_specialize=[
        "token_hist_offset",
        "stream_sum_offset",
        "stream_sumsq_offset",
        "stream_count_offset",
        "stream_samples_offset",
        "position_sum_offset",
        "position_count_offset",
        "token_max_offset",
        "nonfinite_offset",
        "seq_len",
    ]
)
def _token_loss_kernel(
    loss_ptr,
    loss_seg_stride,
    loss_sample_stride,
    valid_ptr,
    valid_seg_stride,
    valid_sample_stride,
    stream_ids_ptr,
    buf_i32_ptr,
    buf_f32_ptr,
    token_hist_offset,
    stream_sum_offset,
    stream_sumsq_offset,
    stream_count_offset,
    stream_samples_offset,
    position_sum_offset,
    position_count_offset,
    token_max_offset,
    nonfinite_offset,
    mtp_hist_offset_ptr,
    seq_len,
    bins_start,
    bins_inv_width,
    POSITION_BUCKETS: tl.constexpr,
    TOKEN_BINS: tl.constexpr,
    TILE_TOKENS: tl.constexpr,
):
    # One program covers part of one sample, so it has a single stream and one atomic per sum.
    segment = tl.program_id(axis=0)
    sample = tl.program_id(axis=1)
    chunk = tl.program_id(axis=2)
    is_main = segment == 0
    stream_id = tl.load(stream_ids_ptr + sample).to(tl.int32)

    cols = chunk * TILE_TOKENS + tl.arange(0, TILE_TOKENS)
    col_mask = cols < seq_len

    loss_row = loss_ptr + segment.to(tl.int64) * loss_seg_stride + sample.to(tl.int64) * loss_sample_stride
    valid_row = valid_ptr + segment.to(tl.int64) * valid_seg_stride + sample.to(tl.int64) * valid_sample_stride
    loss = tl.load(loss_row + cols, mask=col_mask, other=0.0).to(tl.float32)
    label_valid = tl.load(valid_row + cols, mask=col_mask, other=0) != 0

    finite = col_mask & (tl.abs(loss) < float("inf"))
    valid = label_valid & finite
    lossv = tl.where(valid, loss, 0.0)

    nonfinite_local = tl.sum((col_mask & label_valid & ~finite).to(tl.int32), axis=0)
    tl.atomic_add(buf_i32_ptr + nonfinite_offset, nonfinite_local)

    bin_index = tl.where(valid, _bin_index_tl(lossv, bins_start, bins_inv_width, TOKEN_BINS), 0)
    hist = tl.histogram(bin_index, TOKEN_BINS)
    excluded = TILE_TOKENS - tl.sum(valid.to(tl.int32), axis=0)
    bins = tl.arange(0, TOKEN_BINS)

    if is_main:
        hist_ptr = buf_i32_ptr + token_hist_offset + stream_id * TOKEN_BINS
        tl.atomic_add(hist_ptr + bins, hist)
        tl.atomic_add(hist_ptr, -excluded)

        tl.atomic_add(buf_f32_ptr + stream_sum_offset + stream_id, tl.sum(lossv, axis=0))
        tl.atomic_add(buf_f32_ptr + stream_sumsq_offset + stream_id, tl.sum(lossv * lossv, axis=0))
        tl.atomic_add(buf_i32_ptr + stream_count_offset + stream_id, tl.sum(valid.to(tl.int32), axis=0))
        if chunk == 0:
            tl.atomic_add(buf_i32_ptr + stream_samples_offset + stream_id, 1)

        tl.atomic_max(buf_f32_ptr + token_max_offset, tl.max(lossv, axis=0))

        bucket = (cols * POSITION_BUCKETS) // seq_len
        tl.atomic_add(buf_f32_ptr + position_sum_offset + bucket, lossv, mask=valid)
        tl.atomic_add(buf_i32_ptr + position_count_offset + bucket, 1, mask=valid)
    else:
        head_offset = tl.load(mtp_hist_offset_ptr + (segment - 1))
        hist_ptr = buf_i32_ptr + head_offset
        tl.atomic_add(hist_ptr + bins, hist)
        tl.atomic_add(hist_ptr, -excluded)


def launch_token_loss_kernel(
    loss: torch.Tensor,
    labels_valid: torch.Tensor,
    stream_ids: torch.Tensor,
    buffer_int32: torch.Tensor,
    buffer_float32: torch.Tensor,
    token_hist_offset: int,
    stream_sum_offset: int,
    stream_sumsq_offset: int,
    stream_count_offset: int,
    stream_samples_offset: int,
    position_sum_offset: int,
    position_count_offset: int,
    token_max_offset: int,
    nonfinite_offset: int,
    mtp_hist_offsets: torch.Tensor,
    tile_tokens: int = _TOKEN_LOSS_TILE,
) -> None:
    """`loss` and int32 `labels_valid` are `[segments, samples, sequence_length]`, segment 0 main;
    `mtp_hist_offsets` has one entry per MTP head (a dummy one when there is none)."""
    segments, samples, sequence_length = loss.shape
    if loss.stride(2) != 1:
        loss = loss.contiguous()
    launch(
        _token_loss_kernel,
        (segments, samples, triton.cdiv(sequence_length, tile_tokens)),
        loss,
        loss.stride(0),
        loss.stride(1),
        labels_valid,
        labels_valid.stride(0),
        labels_valid.stride(1),
        stream_ids,
        buffer_int32,
        buffer_float32,
        token_hist_offset,
        stream_sum_offset,
        stream_sumsq_offset,
        stream_count_offset,
        stream_samples_offset,
        position_sum_offset,
        position_count_offset,
        token_max_offset,
        nonfinite_offset,
        mtp_hist_offsets,
        sequence_length,
        float(TOKEN_LOSS_BINS.start),
        1.0 / TOKEN_LOSS_BINS.width,
        POSITION_BUCKETS=LOSS_POSITION_BUCKETS,
        TOKEN_BINS=TOKEN_LOSS_BINS.count,
        TILE_TOKENS=tile_tokens,
    )


@triton.jit(do_not_specialize=["tokens", "seq_len"])
def _expert_loss_kernel(
    loss_ptr,
    loss_sample_stride,
    valid_ptr,
    valid_sample_stride,
    stream_ids_ptr,
    topk_ptr,
    topk_layer_stride,
    topk_row_stride,
    positions_ptr,
    offsets_ptr,
    offsets_row_stride,
    buf_i32_ptr,
    buf_f32_ptr,
    tokens,
    seq_len,
    top_k,
    streams,
    bins_start,
    bins_inv_width,
    WITH_DETAILED: tl.constexpr,
    HIST_BINS: tl.constexpr,
    TILE_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Slot offsets come from a device table so that no layer's offset alignment triggers a compile.
    block = tl.program_id(axis=0)
    layer = tl.program_id(axis=1)

    position = tl.load(positions_ptr + layer)
    row = offsets_ptr + layer.to(tl.int64) * offsets_row_stride
    sum_offset = tl.load(row + 0)
    sumsq_offset = tl.load(row + 1)
    count_offset = tl.load(row + 2)
    hist_offset = tl.load(row + 3)
    stream_count_offset = tl.load(row + 4)
    stream_sum_offset = tl.load(row + 5)

    rows = block * TILE_TOKENS + tl.arange(0, TILE_TOKENS)
    row_mask = rows < tokens
    sample = rows // seq_len
    col = rows - sample * seq_len
    k_cols = tl.arange(0, BLOCK_K)
    k_mask = k_cols < top_k
    mask2d = row_mask[:, None] & k_mask[None, :]

    loss = tl.load(loss_ptr + sample.to(tl.int64) * loss_sample_stride + col, mask=row_mask, other=0.0).to(tl.float32)
    label_valid = tl.load(valid_ptr + sample.to(tl.int64) * valid_sample_stride + col, mask=row_mask, other=0) != 0
    valid = label_valid & row_mask & (tl.abs(loss) < float("inf"))
    lossv = tl.where(valid, loss, 0.0)
    stream_index = tl.load(stream_ids_ptr + sample, mask=row_mask, other=0)

    expert = tl.load(
        topk_ptr + position.to(tl.int64) * topk_layer_stride + rows[:, None].to(tl.int64) * topk_row_stride
        + k_cols[None, :],
        mask=mask2d,
        other=0,
    ).to(tl.int32)

    weight2d = mask2d & valid[:, None]
    loss2d = tl.where(weight2d, lossv[:, None], 0.0)

    tl.atomic_add(buf_f32_ptr + sum_offset + expert, loss2d, mask=weight2d)
    tl.atomic_add(buf_f32_ptr + sumsq_offset + expert, loss2d * loss2d, mask=weight2d)
    tl.atomic_add(buf_i32_ptr + count_offset + expert, 1, mask=weight2d)

    if WITH_DETAILED:
        bin_index = _bin_index_tl(loss2d, bins_start, bins_inv_width, HIST_BINS)
        tl.atomic_add(buf_i32_ptr + hist_offset + expert * HIST_BINS + bin_index, 1, mask=weight2d)

        combined = expert * streams + stream_index[:, None]
        tl.atomic_add(buf_i32_ptr + stream_count_offset + combined, 1, mask=weight2d)
        tl.atomic_add(buf_f32_ptr + stream_sum_offset + combined, loss2d, mask=weight2d)


def launch_expert_loss_kernel(
    loss: torch.Tensor,
    labels_valid: torch.Tensor,
    stream_ids: torch.Tensor,
    topk_buffer: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    buffer_int32: torch.Tensor,
    buffer_float32: torch.Tensor,
    top_k: int,
    streams: int,
    detailed: bool,
    tile_tokens: int = _EXPERT_LOSS_TILE,
) -> None:
    """One launch for every routed layer of one segment. `loss` and int32 `labels_valid` are the
    segment's `[samples, sequence_length]`; `positions` (int32) and `offsets` (int64 `[layers, 6]`: sum,
    sumsq, count, hist, stream_count, stream_sum) select each layer's topk row and slots."""
    if loss.stride(1) != 1:
        loss = loss.contiguous()
    samples, sequence_length = loss.shape
    tokens = samples * sequence_length
    launch(
        _expert_loss_kernel,
        (triton.cdiv(tokens, tile_tokens), positions.shape[0], 1),
        loss,
        loss.stride(0),
        labels_valid,
        labels_valid.stride(0),
        stream_ids,
        topk_buffer,
        topk_buffer.stride(0),
        topk_buffer.stride(1),
        positions,
        offsets,
        offsets.stride(0),
        buffer_int32,
        buffer_float32,
        tokens,
        sequence_length,
        top_k,
        streams,
        float(EXPERT_LOSS_BINS.start),
        1.0 / EXPERT_LOSS_BINS.width,
        WITH_DETAILED=detailed,
        HIST_BINS=EXPERT_LOSS_BINS.count,
        TILE_TOKENS=tile_tokens,
        BLOCK_K=triton.next_power_of_2(top_k),
    )
