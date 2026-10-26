from __future__ import annotations

import torch
import triton
import triton.language as tl

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.contract.metrics import INT32_MAX, QUANTILE_COUNTS_WIDTH, QUANTILE_LEVELS

# A 16/32-bit value is widened exactly to float32 and mapped to a key whose unsigned order is the float order. Each
# pass histograms one 8-bit digit of the key for every target rank of q1..q99 and narrows each target to one bin;
# q0 and q100 are the minimum and maximum keys. A bfloat16 key has 16 significant bits (two passes), float16 19
# (three) and float32 32 (four). A float64 population is sorted completely instead.
_KEY_BITS = 32
_DIGIT_BITS = 8
_INT32_MIN = -(2**31)
_TARGETS = 7
_SLOTS = 8
_BINS = 2**_DIGIT_BITS
_BLOCK = 4096
_PROGRAMS = 264
_PASSES = {torch.bfloat16: 2, torch.float16: 3, torch.float32: 4}
_SORT_BLOCK = 1024
_SORT_PROGRAMS = 256
_SORT_PATH_WARPS = 4
_SELECT_WARPS = 4
_SCAN_WARPS = 8


@triton.jit
def _prepare(values, valid, sortable, partials, elements, HAS_MASK: tl.constexpr,
             PROGRAMS: tl.constexpr, BLOCK: tl.constexpr):
    program = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    finite_count = tl.full((), 0, tl.int32)
    nan_count = tl.full((), 0, tl.int32)
    positive_count = tl.full((), 0, tl.int32)
    negative_count = tl.full((), 0, tl.int32)
    masked_count = tl.full((), 0, tl.int32)
    for start in range(program.to(tl.int64) * BLOCK, elements, PROGRAMS * BLOCK):
        indices = start + offsets
        present = indices < elements
        value = tl.load(values + indices, present, other=0)
        if HAS_MASK:
            included = tl.load(valid + indices, present, other=0) & present
        else:
            included = present
        nan = value != value
        positive = value == float("inf")
        negative = value == -float("inf")
        finite = included & ~nan & ~positive & ~negative
        tl.store(sortable + indices, tl.where(finite, value, float("inf")), present)
        finite_count += tl.sum(finite.to(tl.int32), 0)
        nan_count += tl.sum((included & nan).to(tl.int32), 0)
        positive_count += tl.sum((included & positive).to(tl.int32), 0)
        negative_count += tl.sum((included & negative).to(tl.int32), 0)
        masked_count += tl.sum((present & ~included).to(tl.int32), 0)
    tl.store(partials + program, finite_count)
    tl.store(partials + PROGRAMS + program, nan_count)
    tl.store(partials + 2 * PROGRAMS + program, positive_count)
    tl.store(partials + 3 * PROGRAMS + program, negative_count)
    tl.store(partials + 4 * PROGRAMS + program, masked_count)


@triton.jit
def _select(ordered, partials, quantiles, counts, PROGRAMS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    finite = tl.sum(tl.load(partials + offsets, offsets < PROGRAMS, other=0), 0)
    tl.store(counts, finite)
    for category in tl.static_range(1, 5):
        count = tl.sum(tl.load(partials + category * PROGRAMS + offsets, offsets < PROGRAMS, other=0), 0)
        tl.store(counts + category, count)
    slots = tl.arange(0, 16)
    percentiles = tl.where(slots == 0, 0, tl.where(slots == 1, 1, tl.where(slots == 2, 5,
                  tl.where(slots == 3, 25, tl.where(slots == 4, 50, tl.where(slots == 5, 75,
                  tl.where(slots == 6, 95, tl.where(slots == 7, 99, 100))))))))
    indices = tl.maximum((finite.to(tl.int64) * percentiles + 99) // 100 - 1, 0)
    selected = tl.load(ordered + indices, slots < 9, other=0)
    tl.store(quantiles + slots, tl.where(finite > 0, selected, float("nan")), slots < 9)


@triton.jit
def _load_keys(values, valid, indices, elements, HAS_MASK: tl.constexpr):
    present = indices < elements
    value = tl.load(values + indices, present, other=0).to(tl.float32)
    if HAS_MASK:
        included = tl.load(valid + indices, present, other=0) & present
    else:
        included = present
    finite = included & (value == value) & (value != float("inf")) & (value != -float("inf"))
    bits = value.to(tl.int32, bitcast=True)
    # Signed order of `ordered` is the float order; flipping its sign bit gives the unsigned key.
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    key = ordered ^ (tl.full((), 1, tl.int32) << 31)
    return value, present, included, finite, ordered, key


@triton.jit
def _radix_count(values, valid, counts, hist, extremes, elements, HAS_MASK: tl.constexpr,
                 PROGRAMS: tl.constexpr, BLOCK: tl.constexpr, BINS: tl.constexpr):
    program = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    local = tl.zeros((BINS,), tl.int32)
    dumped = tl.full((), 0, tl.int32)
    finite_count = tl.full((), 0, tl.int32)
    nan_count = tl.full((), 0, tl.int32)
    positive_count = tl.full((), 0, tl.int32)
    negative_count = tl.full((), 0, tl.int32)
    masked_count = tl.full((), 0, tl.int32)
    smallest = tl.full((), 2147483647, tl.int32)
    largest = tl.full((), -2147483647 - 1, tl.int32)
    for start in range(program.to(tl.int64) * BLOCK, elements, PROGRAMS * BLOCK):
        value, present, included, finite, ordered, key = _load_keys(values, valid, start + offsets, elements, HAS_MASK)
        # Lanes outside the population land in bin 0 and are subtracted again below.
        local += tl.histogram(tl.where(finite, (key >> 24) & (BINS - 1), 0), BINS)
        dumped += tl.sum((~finite).to(tl.int32), 0)
        finite_count += tl.sum(finite.to(tl.int32), 0)
        nan_count += tl.sum((included & (value != value)).to(tl.int32), 0)
        positive_count += tl.sum((included & (value == float("inf"))).to(tl.int32), 0)
        negative_count += tl.sum((included & (value == -float("inf"))).to(tl.int32), 0)
        masked_count += tl.sum((present & ~included).to(tl.int32), 0)
        smallest = tl.minimum(smallest, tl.min(tl.where(finite, ordered, 2147483647), 0))
        largest = tl.maximum(largest, tl.max(tl.where(finite, ordered, -2147483647 - 1), 0))
    bins = tl.arange(0, BINS)
    tl.atomic_add(hist + bins, local - tl.where(bins == 0, dumped, 0))
    tl.atomic_add(counts, finite_count)
    tl.atomic_add(counts + 1, nan_count)
    tl.atomic_add(counts + 2, positive_count)
    tl.atomic_add(counts + 3, negative_count)
    tl.atomic_add(counts + 4, masked_count)
    tl.atomic_min(extremes, smallest)
    tl.atomic_max(extremes + 1, largest)


@triton.jit
def _radix_refine(values, valid, prefixes, hist, elements, HAS_MASK: tl.constexpr, PREFIX_SHIFT: tl.constexpr,
                  PREFIX_MASK: tl.constexpr, SHIFT: tl.constexpr, PROGRAMS: tl.constexpr, BLOCK: tl.constexpr,
                  BINS: tl.constexpr, TARGETS: tl.constexpr, SLOTS: tl.constexpr):
    # One read serves every target; row t of the histogram counts the digits under t's prefix.
    program = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    rows = tl.arange(0, SLOTS)
    bins = tl.arange(0, BINS)
    local = tl.zeros((SLOTS, BINS), tl.int32)
    for start in range(program.to(tl.int64) * BLOCK, elements, PROGRAMS * BLOCK):
        _, _, _, finite, _, key = _load_keys(values, valid, start + offsets, elements, HAS_MASK)
        high = (key >> PREFIX_SHIFT) & PREFIX_MASK
        digit = (key >> SHIFT) & (BINS - 1)
        for target in tl.static_range(TARGETS):
            match = finite & (high == tl.load(prefixes + target).to(tl.int32))
            # Non-matching lanes were counted in bin 0.
            counted = tl.histogram(tl.where(match, digit, 0), BINS) - tl.where(
                bins == 0, tl.sum((~match).to(tl.int32), 0), 0)
            local += tl.where(rows[:, None] == target, counted[None, :], 0)
    tl.atomic_add(hist + rows[:, None] * BINS + bins[None, :], local)


@triton.jit
def _radix_select(hist, counts, prefixes, ranks, extremes, quantiles, PASS: tl.constexpr, FINAL_SHIFT: tl.constexpr,
                  BINS: tl.constexpr, TARGETS: tl.constexpr, FINAL: tl.constexpr):
    bins = tl.arange(0, BINS)
    finite = tl.load(counts).to(tl.int64)
    for target in tl.static_range(TARGETS):
        if PASS == 0:
            percentile = tl.where(target == 0, 1, tl.where(target == 1, 5, tl.where(target == 2, 25,
                         tl.where(target == 3, 50, tl.where(target == 4, 75, tl.where(target == 5, 95, 99))))))
            rank = tl.maximum((finite * percentile + 99) // 100 - 1, 0)
            counted = tl.load(hist + bins).to(tl.int64)
            prefix = tl.full((), 0, tl.int64)
        else:
            rank = tl.load(ranks + target)
            counted = tl.load(hist + target * BINS + bins).to(tl.int64)
            prefix = tl.load(prefixes + target)
        cumulative = tl.cumsum(counted, 0)
        selected = tl.sum((cumulative <= rank).to(tl.int64), 0)
        rank -= tl.sum(tl.where(bins < selected, counted, 0), 0)
        prefix = (prefix << 8) | selected
        tl.store(prefixes + target, prefix)
        tl.store(ranks + target, rank)
        if FINAL:
            # Key bits below a 16-bit source's precision are zeros for a positive value and ones
            # for a negative one, whose key has a clear top bit.
            negative = ((prefix >> (31 - FINAL_SHIFT)) & 1) == 0
            key = (prefix << FINAL_SHIFT) | tl.where(negative, (1 << FINAL_SHIFT) - 1, 0)
            ordered = key.to(tl.int32) ^ (tl.full((), 1, tl.int32) << 31)
            value = (ordered ^ ((ordered >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)
            tl.store(quantiles + 1 + target, tl.where(finite > 0, value, float("nan")))
    if FINAL:
        for index in tl.static_range(2):
            ordered = tl.load(extremes + index)
            value = (ordered ^ ((ordered >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)
            tl.store(quantiles + index * (TARGETS + 1), tl.where(finite > 0, value, float("nan")))


def _launch_sorted(flat: torch.Tensor, valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    # Without a mask the kernels are given the values as a placeholder pointer.
    mask = flat if valid is None else valid
    programs = min(triton.cdiv(flat.numel(), _SORT_BLOCK), _SORT_PROGRAMS)
    sortable = torch.empty_like(flat)
    partials = torch.empty(QUANTILE_COUNTS_WIDTH * programs, dtype=torch.int32, device=flat.device)
    quantiles = torch.empty(len(QUANTILE_LEVELS), dtype=torch.float64, device=flat.device)
    counts = torch.empty(QUANTILE_COUNTS_WIDTH, dtype=torch.int32, device=flat.device)
    _prepare[(programs,)](flat, mask, sortable, partials, flat.numel(), valid is not None,
                          programs, _SORT_BLOCK, num_warps=_SORT_PATH_WARPS)
    ordered = torch.sort(sortable).values
    _select[(1,)](ordered, partials, quantiles, counts, programs, triton.next_power_of_2(programs),
                  num_warps=_SORT_PATH_WARPS)
    return quantiles, counts


def launch_exact_quantiles(values: torch.Tensor, valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """The quantiles (`QUANTILE_LEVELS`) and the counts (finite, NaN, +inf, -inf and masked elements) of the
    elements of `values` that `valid` selects, or of all of them without `valid`."""
    flat = values.detach().reshape(-1)
    mask = None if valid is None else valid.reshape(-1)
    if values.dtype == torch.float64:
        # The extension selects an unmasked population that is small enough for one block; any other is sorted.
        extension = activation_ext.prebuilt() if mask is None else None
        selected = None if extension is None else extension.exact_quantiles_float64(flat.contiguous())
        return _launch_sorted(flat, mask) if selected is None else selected
    quantiles = torch.empty(len(QUANTILE_LEVELS), dtype=torch.float32, device=flat.device)
    counts = torch.empty(QUANTILE_COUNTS_WIDTH, dtype=torch.int32, device=flat.device)
    if launch_exact_quantiles_into(values, valid, quantiles, counts):
        return quantiles, counts
    return _launch_triton(flat, mask)


def launch_exact_quantiles_into(
    values: torch.Tensor,
    valid: torch.Tensor | None,
    quantiles: torch.Tensor,
    counts: torch.Tensor,
    first_pass: torch.Tensor | None = None,
) -> bool:
    """The prebuilt extension's quantiles of a 16- or 32-bit population written into nine float32
    `quantiles` and five int32 `counts`, or False without the extension. `first_pass` is the
    extension's activation statistics scratch for the same population, whose first pass it holds."""
    extension = activation_ext.prebuilt()
    if extension is None:
        return False
    extension.exact_quantiles(values, valid, quantiles, counts, first_pass)
    return True


def launch_order_statistics(flat: torch.Tensor, ranks: list[int]) -> torch.Tensor | None:
    """The prebuilt extension's radix selection of `ranks`, or None without the extension."""
    extension = activation_ext.prebuilt()
    return None if extension is None else extension.order_statistics(flat, ranks)


def launch_row_top_values(values: torch.Tensor, k: int) -> torch.Tensor | None:
    """The prebuilt extension's per-row top-k values, or None without the extension."""
    extension = activation_ext.prebuilt()
    return None if extension is None else extension.row_top_values(values, k)


def launch_row_entropy(values: torch.Tensor) -> torch.Tensor | None:
    """The prebuilt extension's per-row normalized entropy, or None without the extension."""
    extension = activation_ext.prebuilt()
    return None if extension is None else extension.row_entropy(values)


def launch_row_margins(values: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor | None:
    """The prebuilt extension's per-row margins below a boundary, or None without the extension."""
    extension = activation_ext.prebuilt()
    return None if extension is None else extension.row_margins(values, boundary)


def _launch_triton(flat: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    device = flat.device
    elements = flat.numel()
    passes = _PASSES[flat.dtype]
    programs = min(triton.cdiv(elements, _BLOCK), _PROGRAMS)
    refined_words = (passes - 1) * _SLOTS * _BINS
    scratch = torch.zeros(QUANTILE_COUNTS_WIDTH + _BINS + refined_words, dtype=torch.int32, device=device)
    counts, first_histogram, refined_histograms = scratch.split((QUANTILE_COUNTS_WIDTH, _BINS, refined_words))
    extremes = torch.tensor([INT32_MAX, _INT32_MIN], dtype=torch.int32, device=device)
    selection = torch.empty(2 * _TARGETS, dtype=torch.int64, device=device)
    prefixes, ranks = selection.split(_TARGETS)
    quantiles = torch.empty(len(QUANTILE_LEVELS), dtype=torch.float32, device=device)
    has_mask = mask is not None
    # Without a mask the kernels are given the values as a placeholder pointer.
    source_mask = flat if mask is None else mask
    final_shift = _KEY_BITS - _DIGIT_BITS * passes
    _radix_count[(programs,)](flat, source_mask, counts, first_histogram, extremes, elements, has_mask,
                              programs, _BLOCK, _BINS, num_warps=_SCAN_WARPS)
    _radix_select[(1,)](first_histogram, counts, prefixes, ranks, extremes, quantiles, 0, final_shift, _BINS,
                        _TARGETS, passes == 1, num_warps=_SELECT_WARPS)
    for level in range(1, passes):
        histogram = refined_histograms[(level - 1) * _SLOTS * _BINS:level * _SLOTS * _BINS]
        prefix_bits = _DIGIT_BITS * level
        _radix_refine[(programs,)](flat, source_mask, prefixes, histogram, elements, has_mask,
                                   _KEY_BITS - prefix_bits, 2**prefix_bits - 1, _KEY_BITS - _DIGIT_BITS - prefix_bits,
                                   programs, _BLOCK, _BINS, _TARGETS, _SLOTS, num_warps=_SCAN_WARPS)
        _radix_select[(1,)](histogram, counts, prefixes, ranks, extremes, quantiles, level, final_shift, _BINS,
                            _TARGETS, level == passes - 1, num_warps=_SELECT_WARPS)
    return quantiles, counts
