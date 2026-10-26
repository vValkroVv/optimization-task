from __future__ import annotations

import math

import torch

from metric_sampler.capture.stats.helpers.quantiles import QUANTILE_PERCENTILES

# A bf16/fp16 population the statistics kernels count in full. Its scratch holds the counts of its raw
# values (`raw_pair_counts`), from which key_finalize_kernel selects the quantiles as from the counts of
# its finite keys (`key_histogram`): CPU reference of both. A population of at most this many elements
# is selected in one block and has no such scratch.
SELECTION_DTYPES = (torch.bfloat16, torch.float16)
ONE_BLOCK_SELECTION = 1 << 13
RAW_VALUES = 1 << 16
KEY_BINS = RAW_VALUES
KEY_GROUP = 256
GROUP_SHIFT = 6
GROUP_RAW_VALUES = 1 << GROUP_SHIFT
KEY_WORD_GROUPS = KEY_BINS // GROUP_RAW_VALUES
PAIR_COUNT_WORDS = KEY_BINS // 2
KEY_SCRATCH_WORDS = PAIR_COUNT_WORDS + KEY_WORD_GROUPS // 2
SIGN_BIT_POSITION = 15
SIGN_BIT = 1 << SIGN_BIT_POSITION
MAGNITUDE_MASK = SIGN_BIT - 1
LOW_HALF_MASK = 0xFFFF
COUNT_HIGH_HALF_SHIFT = 32
PERCENT = 100


def from_raw_bits(raw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The bf16/fp16 elements whose bits are `raw`, int32 in [0, 65536)."""
    return torch.where(raw >= SIGN_BIT, raw - RAW_VALUES, raw).to(torch.int16).view(dtype)


def raw_bits(values: torch.Tensor) -> torch.Tensor:
    return values.reshape(-1).view(torch.int16).to(torch.int32) & LOW_HALF_MASK


def key16(raw: torch.Tensor) -> torch.Tensor:
    """The key of an element's bits: its unsigned order is the float order, negative zero first."""
    return raw ^ (SIGN_BIT | ((raw >> SIGN_BIT_POSITION) * MAGNITUDE_MASK))


def key_values(keys: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The float32 value of each key of bf16/fp16 elements."""
    return from_raw_bits(keys ^ torch.where(keys >= SIGN_BIT, SIGN_BIT, LOW_HALF_MASK), dtype).float()


def key_histogram(values: torch.Tensor) -> torch.Tensor:
    """What the statistics kernels count in the scratch of a bf16/fp16 population: every element's key,
    then the keys of NaN and infinities dropped."""
    counted = torch.bincount(key16(raw_bits(values)), minlength=KEY_BINS)
    finite = torch.isfinite(key_values(torch.arange(KEY_BINS, device=values.device), values.dtype))
    return torch.where(finite, counted, 0)


def raw_pair_counts(values: torch.Tensor) -> torch.Tensor:
    """What the statistics kernels leave in the scratch of a bf16/fp16 population: for each pair of raw
    values 2w and 2w + 1 a 64-bit word with the count of 2w in its low half and that of 2w + 1 in its
    high half. Every element is counted, NaN and infinities too."""
    counted = torch.bincount(raw_bits(values), minlength=RAW_VALUES).reshape(-1, 2)
    return counted[:, 0] | (counted[:, 1] << COUNT_HIGH_HALF_SHIFT)


def group_totals(values: torch.Tensor) -> torch.Tensor:
    """What the statistics kernels leave in the scratch of a bf16/fp16 population after the raw counts: for
    each group of 32 words, the 64 raw values from 64 g, the count of all their elements as an int32."""
    return torch.bincount(raw_bits(values) >> GROUP_SHIFT, minlength=KEY_WORD_GROUPS).to(torch.int32)


def select_from_key_histogram(histogram: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """What key_finalize_kernel selects: for each percentile q, the key at rank
    max(ceil(q N / 100) - 1, 0) of the N finite elements, found here as the group of 256 keys whose cumulative
    total first exceeds the rank and then the key of that group; NaN for an empty population."""
    population = int(histogram.sum())
    if population == 0:
        return torch.full((len(QUANTILE_PERCENTILES),), math.nan)
    groups = histogram.reshape(-1, KEY_GROUP)
    group_ends = groups.sum(dim=1).cumsum(0)
    keys = []
    for percentile in QUANTILE_PERCENTILES:
        rank = max((population * percentile + (PERCENT - 1)) // PERCENT - 1, 0)
        group = int(torch.searchsorted(group_ends, rank, right=True))
        below = int(group_ends[group - 1]) if group else 0
        key = int(torch.searchsorted(groups[group].cumsum(0), rank - below, right=True))
        keys.append(group * KEY_GROUP + key)
    return key_values(torch.tensor(keys), dtype)


def selected_bits(quantiles: torch.Tensor) -> torch.Tensor:
    """The float32 bits of selected quantiles, every NaN as one NaN, so that signed zeros differ."""
    return torch.where(quantiles.isnan(), torch.tensor(math.nan, device=quantiles.device), quantiles).view(torch.int32)
