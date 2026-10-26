from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from metric_sampler.capture.kernels.triton_launch import launch

_TILE_ROWS = 32
_NUM_WARPS = 4
# Programs a launch aims for: enough in flight to hide memory latency, few enough that a head's records stay
# small for its last program to merge.
_TARGET_PROGRAMS = 1024
_MAX_BLOCKS = 256
# The per-head counters, as int32, then the records of up to _MAX_PROGRAMS programs of at most _MAX_RECORD words.
_COUNTER_WORDS = 1024
_MAX_PROGRAMS = 2048
_MAX_RECORD = 16
_SCRATCH_WORDS = _COUNTER_WORDS + _MAX_PROGRAMS * _MAX_RECORD

# The scratch of the launches on each device's stream, kept for the life of the process.
_scratches: dict[tuple[torch.device, int], torch.Tensor] = {}


def _scratch(device: torch.device) -> torch.Tensor:
    """The records and counters that every launch on `device`'s current stream shares. Its counters are zero
    except while a launch is under way, whose last program of each head zeroes that head's counter again, and
    every launch writes the records it reads. There is one per stream, so launches that run on different
    streams at once share none."""
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    scratch = _scratches.get(key)
    if scratch is None:
        scratch = _scratches[key] = torch.zeros(_SCRATCH_WORDS, dtype=torch.float64, device=device)
    return scratch


# Each program of the kernels below reduces a block of rows of one head into a record in a scratch buffer. The last
# program of a head to finish, told by a per-head counter whose release/acquire atomic orders it after every other
# program's record, merges the head's records in block order into the head's words of the slot and resets the
# counter. The merge order is therefore fixed whichever program arrives last, and no second launch is needed.
# A NaN row makes the minimum, maximum or amax of its head NaN, which a Triton reduction would otherwise drop.
@triton.jit
def _store_sum(ptr, new, first):
    old = tl.load(ptr).to(tl.float64)
    tl.store(ptr, tl.where(first, new, old + new).to(tl.float32))


@triton.jit
def _store_min(ptr, new, first):
    old = tl.load(ptr)
    merged = tl.where((old != old) | (new != new), float("nan"), tl.minimum(old, new))
    tl.store(ptr, tl.where(first, new, merged))


@triton.jit
def _store_max(ptr, new, first):
    old = tl.load(ptr)
    merged = tl.where((old != old) | (new != new), float("nan"), tl.maximum(old, new))
    tl.store(ptr, tl.where(first, new, merged))


@triton.jit
def _load_tile(ptr, row_offsets, head_offset, first_col, cols, width, valid):
    # Float32 [rows, cols] tile of the head's columns [first_col, first_col + width); rows past the end read 0.
    return tl.load(
        ptr + row_offsets[:, None] + head_offset + first_col + cols[None, :],
        mask=valid[:, None] & (cols < width)[None, :],
        other=0.0,
    ).to(tl.float32)


@triton.jit
def _row_norm(tile):
    return libdevice.sqrt(tl.sum(tile * tile, axis=1))


@triton.jit
def _gather(partials, field, present, other):
    # Read past the L1 cache: the records were written by programs of other SMs.
    return tl.load(partials + field, mask=present, other=other, cache_modifier=".cg")


@triton.jit
def _merged_sum(partials, field, present):
    return tl.sum(_gather(partials, field, present, 0.0), axis=0)


@triton.jit
def _merged_min(partials, field, flag, present):
    low = tl.min(_gather(partials, field, present, float("inf")), axis=0).to(tl.float32)
    return tl.where(tl.max(_gather(partials, flag, present, 0.0), axis=0) > 0, float("nan"), low)


@triton.jit
def _merged_max(partials, field, flag, present):
    high = tl.max(_gather(partials, field, present, float("-inf")), axis=0).to(tl.float32)
    return tl.where(tl.max(_gather(partials, flag, present, 0.0), axis=0) > 0, float("nan"), high)


@triton.jit
def _flag(bad):
    return tl.max(bad, axis=0).to(tl.float64)


# Partial record of the softmax kernel: lse_sum, lse_low, lse_high, lse_bad, then q_nope_sum, q_rope_sum,
# k_nope_sum, k_rope_sum, v_sum, q_peak, k_peak, v_peak, q_bad, k_bad, v_bad (15 words).
@triton.jit(do_not_specialize=["rows", "rows_before", "rows_per_program", "blocks", "lse_offset", "qkv_offset"])
def _softmax_stats_kernel(
    q_ptr, k_ptr, v_ptr, lse_ptr, out_ptr, scratch_ptr,
    q_row_stride, q_head_stride, k_row_stride, k_head_stride, v_row_stride, v_head_stride,
    lse_head_stride, lse_batch_stride, lse_seq_stride, seq,
    rows, rows_before, rows_per_program, blocks, lse_offset, qkv_offset,
    NOPE: tl.constexpr, ROPE: tl.constexpr, VDIM: tl.constexpr,
    NOPE_PAD: tl.constexpr, ROPE_PAD: tl.constexpr, V_PAD: tl.constexpr,
    TILE_ROWS: tl.constexpr, BLOCKS_PAD: tl.constexpr, COUNTER_WORDS: tl.constexpr,
):
    block = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    heads = tl.num_programs(axis=1)
    lanes = tl.arange(0, TILE_ROWS)
    nope_cols = tl.arange(0, NOPE_PAD)
    rope_cols = tl.arange(0, ROPE_PAD)
    v_cols = tl.arange(0, V_PAD)

    q_nope_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    q_rope_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    k_nope_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    k_rope_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    v_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    q_peak = tl.zeros([TILE_ROWS], dtype=tl.float32)
    k_peak = tl.zeros([TILE_ROWS], dtype=tl.float32)
    v_peak = tl.zeros([TILE_ROWS], dtype=tl.float32)
    q_bad = tl.zeros([TILE_ROWS], dtype=tl.int32)
    k_bad = tl.zeros([TILE_ROWS], dtype=tl.int32)
    v_bad = tl.zeros([TILE_ROWS], dtype=tl.int32)
    lse_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    lse_low = tl.full([TILE_ROWS], float("inf"), dtype=tl.float32)
    lse_high = tl.full([TILE_ROWS], float("-inf"), dtype=tl.float32)
    lse_bad = tl.zeros([TILE_ROWS], dtype=tl.int32)

    first_row = block * rows_per_program
    for tile in range(0, rows_per_program, TILE_ROWS):
        row = (first_row + tile + lanes).to(tl.int64)
        valid = row < rows

        # Every load of the tile before its first use, so they are all in flight together.
        q_nope = _load_tile(q_ptr, row * q_row_stride, head * q_head_stride, 0, nope_cols, NOPE, valid)
        q_rope = _load_tile(q_ptr, row * q_row_stride, head * q_head_stride, NOPE, rope_cols, ROPE, valid)
        k_nope = _load_tile(k_ptr, row * k_row_stride, head * k_head_stride, 0, nope_cols, NOPE, valid)
        k_rope = _load_tile(k_ptr, row * k_row_stride, head * k_head_stride, NOPE, rope_cols, ROPE, valid)
        v_tile = _load_tile(v_ptr, row * v_row_stride, head * v_head_stride, 0, v_cols, VDIM, valid)
        lse_row = head * lse_head_stride + (row // seq) * lse_batch_stride + (row % seq) * lse_seq_stride
        lse_value = tl.load(lse_ptr + lse_row, mask=valid, other=0.0).to(tl.float32)

        q_nope_norm = _row_norm(q_nope)
        q_rope_norm = _row_norm(q_rope)
        q_nope_sum += q_nope_norm.to(tl.float64)
        q_rope_sum += q_rope_norm.to(tl.float64)
        q_peak = tl.maximum(q_peak, tl.maximum(tl.max(tl.abs(q_nope), axis=1), tl.max(tl.abs(q_rope), axis=1)))
        q_bad = q_bad | ((q_nope_norm != q_nope_norm) | (q_rope_norm != q_rope_norm)).to(tl.int32)

        k_nope_norm = _row_norm(k_nope)
        k_rope_norm = _row_norm(k_rope)
        k_nope_sum += k_nope_norm.to(tl.float64)
        k_rope_sum += k_rope_norm.to(tl.float64)
        k_peak = tl.maximum(k_peak, tl.maximum(tl.max(tl.abs(k_nope), axis=1), tl.max(tl.abs(k_rope), axis=1)))
        k_bad = k_bad | ((k_nope_norm != k_nope_norm) | (k_rope_norm != k_rope_norm)).to(tl.int32)

        v_norm = _row_norm(v_tile)
        v_sum += v_norm.to(tl.float64)
        v_peak = tl.maximum(v_peak, tl.max(tl.abs(v_tile), axis=1))
        v_bad = v_bad | (v_norm != v_norm).to(tl.int32)

        lse_sum += tl.where(valid, lse_value, 0.0).to(tl.float64)
        lse_low = tl.minimum(lse_low, tl.where(valid, lse_value, float("inf")))
        lse_high = tl.maximum(lse_high, tl.where(valid, lse_value, float("-inf")))
        lse_bad = lse_bad | (valid & (lse_value != lse_value)).to(tl.int32)

    record = scratch_ptr + COUNTER_WORDS + (head * blocks + block) * 15
    tl.store(record, tl.sum(lse_sum, axis=0))
    tl.store(record + 1, tl.min(lse_low, axis=0).to(tl.float64))
    tl.store(record + 2, tl.max(lse_high, axis=0).to(tl.float64))
    tl.store(record + 3, _flag(lse_bad))
    tl.store(record + 4, tl.sum(q_nope_sum, axis=0))
    tl.store(record + 5, tl.sum(q_rope_sum, axis=0))
    tl.store(record + 6, tl.sum(k_nope_sum, axis=0))
    tl.store(record + 7, tl.sum(k_rope_sum, axis=0))
    tl.store(record + 8, tl.sum(v_sum, axis=0))
    tl.store(record + 9, tl.max(q_peak, axis=0).to(tl.float64))
    tl.store(record + 10, tl.max(k_peak, axis=0).to(tl.float64))
    tl.store(record + 11, tl.max(v_peak, axis=0).to(tl.float64))
    tl.store(record + 12, _flag(q_bad))
    tl.store(record + 13, _flag(k_bad))
    tl.store(record + 14, _flag(v_bad))

    tl.debug_barrier()
    counters = scratch_ptr.to(tl.pointer_type(tl.int32))
    arrived = tl.atomic_add(counters + head, 1, sem="acq_rel")
    if arrived == blocks - 1:
        slots = tl.arange(0, BLOCKS_PAD)
        present = slots < blocks
        partials = scratch_ptr + COUNTER_WORDS + (head * blocks + slots) * 15
        first = rows_before == 0
        total_rows = (rows_before + rows).to(tl.float32)
        tl.store(out_ptr + lse_offset, total_rows, mask=head == 0)
        tl.store(out_ptr + qkv_offset, total_rows, mask=head == 0)

        lse_base = out_ptr + lse_offset + 1 + head
        _store_sum(lse_base, _merged_sum(partials, 0, present), first)
        _store_min(lse_base + heads, _merged_min(partials, 1, 3, present), first)
        _store_max(lse_base + 2 * heads, _merged_max(partials, 2, 3, present), first)

        qkv_base = out_ptr + qkv_offset + 1 + head
        _store_sum(qkv_base, _merged_sum(partials, 4, present), first)
        _store_sum(qkv_base + heads, _merged_sum(partials, 5, present), first)
        _store_sum(qkv_base + 2 * heads, _merged_sum(partials, 6, present), first)
        _store_sum(qkv_base + 3 * heads, _merged_sum(partials, 7, present), first)
        _store_sum(qkv_base + 4 * heads, _merged_sum(partials, 8, present), first)
        _store_max(qkv_base + 5 * heads, _merged_max(partials, 9, 12, present), first)
        _store_max(qkv_base + 6 * heads, _merged_max(partials, 10, 13, present), first)
        _store_max(qkv_base + 7 * heads, _merged_max(partials, 11, 14, present), first)
        tl.store(counters + head, 0)


# Partial record of the gate kernel: sum, low, high of the per-row means, bad (4 words).
@triton.jit(do_not_specialize=["rows", "rows_before", "rows_per_program", "blocks", "offset"])
def _gate_stats_kernel(
    gate_ptr, out_ptr, scratch_ptr, row_stride, head_stride,
    rows, rows_before, rows_per_program, blocks, offset,
    DIM: tl.constexpr, DIM_PAD: tl.constexpr, TILE_ROWS: tl.constexpr, BLOCKS_PAD: tl.constexpr,
    COUNTER_WORDS: tl.constexpr,
):
    block = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    heads = tl.num_programs(axis=1)
    lanes = tl.arange(0, TILE_ROWS)
    cols = tl.arange(0, DIM_PAD)

    total = tl.zeros([TILE_ROWS], dtype=tl.float64)
    low = tl.full([TILE_ROWS], float("inf"), dtype=tl.float32)
    high = tl.full([TILE_ROWS], float("-inf"), dtype=tl.float32)
    bad = tl.zeros([TILE_ROWS], dtype=tl.int32)

    first_row = block * rows_per_program
    for tile in range(0, rows_per_program, TILE_ROWS):
        row = (first_row + tile + lanes).to(tl.int64)
        valid = row < rows
        mean = tl.sum(_load_tile(gate_ptr, row * row_stride, head * head_stride, 0, cols, DIM, valid), axis=1) * (1.0 / DIM)
        total += tl.where(valid, mean, 0.0).to(tl.float64)
        low = tl.minimum(low, tl.where(valid, mean, float("inf")))
        high = tl.maximum(high, tl.where(valid, mean, float("-inf")))
        bad = bad | (valid & (mean != mean)).to(tl.int32)

    record = scratch_ptr + COUNTER_WORDS + (head * blocks + block) * 4
    tl.store(record, tl.sum(total, axis=0))
    tl.store(record + 1, tl.min(low, axis=0).to(tl.float64))
    tl.store(record + 2, tl.max(high, axis=0).to(tl.float64))
    tl.store(record + 3, _flag(bad))

    tl.debug_barrier()
    counters = scratch_ptr.to(tl.pointer_type(tl.int32))
    arrived = tl.atomic_add(counters + head, 1, sem="acq_rel")
    if arrived == blocks - 1:
        slots = tl.arange(0, BLOCKS_PAD)
        present = slots < blocks
        partials = scratch_ptr + COUNTER_WORDS + (head * blocks + slots) * 4
        first = rows_before == 0
        tl.store(out_ptr + offset, (rows_before + rows).to(tl.float32), mask=head == 0)
        base = out_ptr + offset + 1 + head
        _store_sum(base, _merged_sum(partials, 0, present), first)
        _store_min(base + heads, _merged_min(partials, 1, 3, present), first)
        _store_max(base + 2 * heads, _merged_max(partials, 2, 3, present), first)
        tl.store(counters + head, 0)


# Partial record of the linear kernel: beta_sum, beta_low, beta_high, beta_bad, alpha_sum, alpha_low,
# alpha_high, alpha_bad, g_sum, v_sum, o_sum, o_peak, o_bad (13 words).
@triton.jit(
    do_not_specialize=["rows", "rows_before", "rows_per_program", "blocks", "gate_offset", "norm_offset", "decay_offset"]
)
def _linear_stats_kernel(
    beta_ptr, g_ptr, v_ptr, o_ptr, a_log_ptr, dt_bias_ptr, out_ptr, scratch_ptr,
    beta_row_stride, beta_head_stride, g_row_stride, g_head_stride,
    v_row_stride, v_head_stride, o_row_stride, o_head_stride,
    rows, rows_before, rows_per_program, blocks, gate_offset, norm_offset, decay_offset,
    DIM: tl.constexpr, DIM_PAD: tl.constexpr, TILE_ROWS: tl.constexpr, BLOCKS_PAD: tl.constexpr,
    COUNTER_WORDS: tl.constexpr,
):
    block = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    heads = tl.num_programs(axis=1)
    lanes = tl.arange(0, TILE_ROWS)
    cols = tl.arange(0, DIM_PAD)

    beta_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    alpha_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    g_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    beta_low = tl.full([TILE_ROWS], float("inf"), dtype=tl.float32)
    beta_high = tl.full([TILE_ROWS], float("-inf"), dtype=tl.float32)
    alpha_low = tl.full([TILE_ROWS], float("inf"), dtype=tl.float32)
    alpha_high = tl.full([TILE_ROWS], float("-inf"), dtype=tl.float32)
    beta_bad = tl.zeros([TILE_ROWS], dtype=tl.int32)
    alpha_bad = tl.zeros([TILE_ROWS], dtype=tl.int32)
    v_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    o_sum = tl.zeros([TILE_ROWS], dtype=tl.float64)
    o_peak = tl.zeros([TILE_ROWS], dtype=tl.float32)
    o_bad = tl.zeros([TILE_ROWS], dtype=tl.int32)

    first_row = block * rows_per_program
    for tile in range(0, rows_per_program, TILE_ROWS):
        row = (first_row + tile + lanes).to(tl.int64)
        valid = row < rows
        beta = tl.load(beta_ptr + row * beta_row_stride + head * beta_head_stride, mask=valid, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * g_row_stride + head * g_head_stride, mask=valid, other=0.0).to(tl.float32)
        v_tile = _load_tile(v_ptr, row * v_row_stride, head * v_head_stride, 0, cols, DIM, valid)
        o_tile = _load_tile(o_ptr, row * o_row_stride, head * o_head_stride, 0, cols, DIM, valid)

        alpha = libdevice.exp(g)
        beta_sum += tl.where(valid, beta, 0.0).to(tl.float64)
        alpha_sum += tl.where(valid, alpha, 0.0).to(tl.float64)
        g_sum += g.to(tl.float64)
        beta_low = tl.minimum(beta_low, tl.where(valid, beta, float("inf")))
        beta_high = tl.maximum(beta_high, tl.where(valid, beta, float("-inf")))
        alpha_low = tl.minimum(alpha_low, tl.where(valid, alpha, float("inf")))
        alpha_high = tl.maximum(alpha_high, tl.where(valid, alpha, float("-inf")))
        beta_bad = beta_bad | (valid & (beta != beta)).to(tl.int32)
        alpha_bad = alpha_bad | (valid & (alpha != alpha)).to(tl.int32)

        o_norm = _row_norm(o_tile)
        v_sum += _row_norm(v_tile).to(tl.float64)
        o_sum += o_norm.to(tl.float64)
        o_peak = tl.maximum(o_peak, tl.max(tl.abs(o_tile), axis=1))
        o_bad = o_bad | (o_norm != o_norm).to(tl.int32)

    record = scratch_ptr + COUNTER_WORDS + (head * blocks + block) * 13
    tl.store(record, tl.sum(beta_sum, axis=0))
    tl.store(record + 1, tl.min(beta_low, axis=0).to(tl.float64))
    tl.store(record + 2, tl.max(beta_high, axis=0).to(tl.float64))
    tl.store(record + 3, _flag(beta_bad))
    tl.store(record + 4, tl.sum(alpha_sum, axis=0))
    tl.store(record + 5, tl.min(alpha_low, axis=0).to(tl.float64))
    tl.store(record + 6, tl.max(alpha_high, axis=0).to(tl.float64))
    tl.store(record + 7, _flag(alpha_bad))
    tl.store(record + 8, tl.sum(g_sum, axis=0))
    tl.store(record + 9, tl.sum(v_sum, axis=0))
    tl.store(record + 10, tl.sum(o_sum, axis=0))
    tl.store(record + 11, tl.max(o_peak, axis=0).to(tl.float64))
    tl.store(record + 12, _flag(o_bad))

    tl.debug_barrier()
    counters = scratch_ptr.to(tl.pointer_type(tl.int32))
    arrived = tl.atomic_add(counters + head, 1, sem="acq_rel")
    if arrived == blocks - 1:
        slots = tl.arange(0, BLOCKS_PAD)
        present = slots < blocks
        partials = scratch_ptr + COUNTER_WORDS + (head * blocks + slots) * 13
        first = rows_before == 0
        total_rows = (rows_before + rows).to(tl.float32)
        tl.store(out_ptr + gate_offset, total_rows, mask=head == 0)
        tl.store(out_ptr + norm_offset, total_rows, mask=head == 0)

        gate_base = out_ptr + gate_offset + 1 + head
        _store_sum(gate_base, _merged_sum(partials, 0, present), first)
        _store_min(gate_base + heads, _merged_min(partials, 1, 3, present), first)
        _store_max(gate_base + 2 * heads, _merged_max(partials, 2, 3, present), first)
        _store_sum(gate_base + 3 * heads, _merged_sum(partials, 4, present), first)
        _store_min(gate_base + 4 * heads, _merged_min(partials, 5, 7, present), first)
        _store_max(gate_base + 5 * heads, _merged_max(partials, 6, 7, present), first)
        _store_sum(gate_base + 6 * heads, _merged_sum(partials, 8, present), first)

        norm_base = out_ptr + norm_offset + 1 + head
        _store_sum(norm_base, _merged_sum(partials, 9, present), first)
        _store_sum(norm_base + heads, _merged_sum(partials, 10, present), first)
        _store_max(norm_base + 2 * heads, _merged_max(partials, 11, 12, present), first)

        if decay_offset >= 0:
            a_log = tl.load(a_log_ptr + head).to(tl.float32)
            dt_bias = tl.load(dt_bias_ptr + head).to(tl.float32)
            tl.store(out_ptr + decay_offset + head, libdevice.exp(a_log))
            # torch's softplus: the identity above 20, where exp would overflow its log1p.
            softplus = tl.where(dt_bias > 20.0, dt_bias, libdevice.log1p(libdevice.exp(dt_bias)))
            tl.store(out_ptr + decay_offset + heads + head, softplus)
        tl.store(counters + head, 0)


def _rows(x: torch.Tensor, heads: int) -> torch.Tensor:
    """`x` [..., heads, dim] as [rows, heads, dim] with unit stride along dim."""
    if x.ndim != 3:
        x = x.reshape(-1, heads, x.shape[-1])
    return x if x.stride(-1) == 1 else x.contiguous()


def _pad(dim: int) -> int:
    return triton.next_power_of_2(dim)


def _grid(rows: int, heads: int, tile_rows: int, target_programs: int) -> tuple[int, int, int]:
    """(blocks, rows per program, padded blocks): enough programs to fill the GPU, each a whole number of
    tiles of rows, with none left without a row."""
    if heads > _MAX_PROGRAMS:
        raise ValueError(f"attention statistics take at most {_MAX_PROGRAMS} heads, got {heads}")
    blocks = max(1, min(_MAX_BLOCKS, min(target_programs, _MAX_PROGRAMS) // heads, triton.cdiv(rows, tile_rows)))
    rows_per_program = triton.cdiv(triton.cdiv(rows, blocks), tile_rows) * tile_rows
    blocks = triton.cdiv(rows, rows_per_program)
    return blocks, rows_per_program, _pad(blocks)


def launch_softmax_stats_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lse: torch.Tensor,
    buffer: torch.Tensor,
    lse_offset: int,
    qkv_offset: int,
    nope_dim: int,
    rows_before: int,
    tile_rows: int = _TILE_ROWS,
    num_warps: int = _NUM_WARPS,
    target_programs: int = _TARGET_PROGRAMS,
) -> None:
    """Folds one MLA call into the slots at `lse_offset` and `qkv_offset` of float32 `buffer`. `lse` is
    [heads, rows] or [heads, batch, seq] (the dense flash-attn output viewed head-first), `query` and
    `key` [..., heads, nope + rope] with as many rows as `lse`. `rows_before` is what the step has put in
    the slots already. `tile_rows`, `num_warps` and `target_programs` exist only for tuning sweeps."""
    heads = lse.shape[0]
    query, key, value = (_rows(tensor, heads) for tensor in (query, key, value))
    if lse.ndim == 2:
        lse = lse.unsqueeze(1)
    rope_dim = query.shape[-1] - nope_dim
    rows = query.shape[0]
    blocks, rows_per_program, blocks_pad = _grid(rows, heads, tile_rows, target_programs)
    launch(
        _softmax_stats_kernel, (blocks, heads, 1),
        query, key, value, lse, buffer, _scratch(query.device),
        query.stride(0), query.stride(1), key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        lse.stride(0), lse.stride(1), lse.stride(2), lse.shape[2],
        rows, rows_before, rows_per_program, blocks, lse_offset, qkv_offset,
        NOPE=nope_dim, ROPE=rope_dim, VDIM=value.shape[-1],
        NOPE_PAD=_pad(nope_dim), ROPE_PAD=_pad(rope_dim), V_PAD=_pad(value.shape[-1]),
        TILE_ROWS=tile_rows, BLOCKS_PAD=blocks_pad, COUNTER_WORDS=_COUNTER_WORDS, num_warps=num_warps,
    )


def launch_gate_stats_kernel(
    gate: torch.Tensor,
    buffer: torch.Tensor,
    offset: int,
    rows_before: int,
    tile_rows: int = _TILE_ROWS,
    num_warps: int = _NUM_WARPS,
    target_programs: int = _TARGET_PROGRAMS,
) -> None:
    """Folds the gate logits [..., heads, dim] of one MLA call into the slot at `offset`."""
    heads = gate.shape[-2]
    gate = _rows(gate, heads)
    rows = gate.shape[0]
    blocks, rows_per_program, blocks_pad = _grid(rows, heads, tile_rows, target_programs)
    launch(
        _gate_stats_kernel, (blocks, heads, 1),
        gate, buffer, _scratch(gate.device), gate.stride(0), gate.stride(1),
        rows, rows_before, rows_per_program, blocks, offset,
        DIM=gate.shape[-1], DIM_PAD=_pad(gate.shape[-1]), TILE_ROWS=tile_rows, BLOCKS_PAD=blocks_pad,
        COUNTER_WORDS=_COUNTER_WORDS, num_warps=num_warps,
    )


def launch_linear_stats_kernel(
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    buffer: torch.Tensor,
    gate_offset: int,
    norm_offset: int,
    decay_offset: int,
    rows_before: int,
    tile_rows: int = _TILE_ROWS,
    num_warps: int = _NUM_WARPS,
    target_programs: int = _TARGET_PROGRAMS,
) -> None:
    """Folds one GDN call into the slots at `gate_offset` and `norm_offset`: `beta` and `log_decay` are
    [..., heads], `value` and `output` [..., heads, dim]. A `decay_offset` of -1 leaves the decay parameters
    alone, any other writes them there."""
    heads = beta.shape[-1]
    beta, log_decay = (tensor.reshape(-1, heads) for tensor in (beta, log_decay))
    value, output = (_rows(tensor, heads) for tensor in (value, output))
    rows = beta.shape[0]
    blocks, rows_per_program, blocks_pad = _grid(rows, heads, tile_rows, target_programs)
    launch(
        _linear_stats_kernel, (blocks, heads, 1),
        beta, log_decay, value, output, a_log, dt_bias, buffer, _scratch(beta.device),
        beta.stride(0), beta.stride(1), log_decay.stride(0), log_decay.stride(1),
        value.stride(0), value.stride(1), output.stride(0), output.stride(1),
        rows, rows_before, rows_per_program, blocks, gate_offset, norm_offset, decay_offset,
        DIM=value.shape[-1], DIM_PAD=_pad(value.shape[-1]), TILE_ROWS=tile_rows, BLOCKS_PAD=blocks_pad,
        COUNTER_WORDS=_COUNTER_WORDS, num_warps=num_warps,
    )
