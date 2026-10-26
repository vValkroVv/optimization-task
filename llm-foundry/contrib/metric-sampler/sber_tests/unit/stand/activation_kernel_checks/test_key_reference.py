from __future__ import annotations

import math

import pytest
import torch

from metric_sampler.capture.stats.helpers.quantiles import QUANTILE_PERCENTILES, exact_quantiles_aten
from stand.activation_kernel_checks.key_reference import (
    SELECTION_DTYPES,
    from_raw_bits,
    group_totals,
    key16,
    key_histogram,
    key_values,
    raw_bits,
    raw_pair_counts,
    select_from_key_histogram,
)
from stand.activation_kernel_checks.populations import selection_populations

pytestmark = pytest.mark.unit

DTYPES = SELECTION_DTYPES
CPU = torch.device("cpu")


def every_bit_pattern(dtype: torch.dtype) -> torch.Tensor:
    return from_raw_bits(torch.arange(1 << 16), dtype)


def radix_key32(floats: torch.Tensor) -> torch.Tensor:
    """The 32-bit key the radix passes give a float32 value."""
    bits = floats.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(bits >= 0x80000000, bits ^ 0xFFFFFFFF, bits | 0x80000000)


def selected(values: torch.Tensor) -> torch.Tensor:
    return select_from_key_histogram(key_histogram(values), values.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_keys_of_all_bit_patterns_are_a_permutation_ordered_like_the_floats(dtype):
    values = every_bit_pattern(dtype)
    keys = key16(raw_bits(values))
    by_key = key_values(torch.arange(1 << 16), dtype)
    finite = by_key[torch.isfinite(by_key)]

    assert sorted(keys.tolist()) == list(range(1 << 16))
    assert torch.equal(key_values(keys, dtype).view(torch.int32), values.float().view(torch.int32))
    assert torch.all(finite[1:] >= finite[:-1])
    assert key16(torch.tensor([0x8000, 0x0000])).tolist() == [0x7FFF, 0x8000]


def test_bf16_keys_are_the_top_half_of_the_radix_keys_of_the_widened_values():
    values = every_bit_pattern(torch.bfloat16)

    assert torch.equal(radix_key32(values.float()) >> 16, key16(raw_bits(values)).to(torch.int64))


def test_fp16_keys_order_the_finite_values_like_the_radix_keys_of_the_widened_values():
    values = every_bit_pattern(torch.float16)
    finite = torch.isfinite(values)

    assert torch.equal(torch.argsort(radix_key32(values.float()[finite])),
                       torch.argsort(key16(raw_bits(values))[finite]))


@pytest.mark.parametrize("dtype,nonfinite", [(torch.bfloat16, 2 * 128), (torch.float16, 2 * 1024)])
def test_histogram_drops_exactly_the_keys_of_nan_and_infinities(dtype, nonfinite):
    histogram = key_histogram(every_bit_pattern(dtype))
    finite = torch.isfinite(key_values(torch.arange(1 << 16), dtype))

    assert (histogram == 0).sum().item() == nonfinite
    assert torch.equal(histogram == 1, finite)
    assert histogram.sum().item() == (1 << 16) - nonfinite


@pytest.mark.parametrize("name", [name for name, _ in selection_populations(CPU)])
def test_selection_from_the_histogram_equals_the_sorted_reference_on_adversarial_populations(populations, name):
    values = populations[name].reshape(-1)
    expected, expected_counts = exact_quantiles_aten(values)
    histogram = key_histogram(values)

    actual = select_from_key_histogram(histogram, values.dtype)

    assert torch.equal(actual.nan_to_num(7.0), expected.nan_to_num(7.0))
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert histogram.sum().item() == expected_counts[0].item()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", [1, 2, 3, 99, 100, 101, 199, 201, 256])
def test_selected_ranks_are_integer_ceilings_for_every_population_size(dtype, size):
    values = torch.arange(size, dtype=dtype).flip(0)

    expected = exact_quantiles_aten(values)[0]

    assert torch.equal(selected(values), expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_signed_zeros_are_selected_by_key_order_with_negative_zero_first(dtype):
    values = torch.tensor([0.0] * 5 + [-0.0] * 3, dtype=dtype)

    actual = selected(values)

    assert torch.signbit(actual).tolist() == [True, True, True, True, False, False, False, False, False]
    assert torch.equal(actual, torch.zeros(9))


@pytest.mark.parametrize("dtype", DTYPES)
def test_population_without_a_finite_value_selects_nan(dtype):
    values = torch.tensor([math.nan, math.inf, -math.inf], dtype=dtype)

    assert torch.isnan(selected(values)).all()
    assert key_histogram(values).sum().item() == 0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("dtype", DTYPES)
def test_selection_by_groups_equals_a_cumulative_search_of_all_keys(seed, dtype):
    generator = torch.Generator().manual_seed(seed)
    finite = torch.isfinite(key_values(torch.arange(1 << 16), dtype))
    histogram = torch.zeros(1 << 16, dtype=torch.int64)
    hot = torch.randint(0, 1 << 16, (3 + 40 * seed,), generator=generator)
    histogram[hot] = torch.randint(1, 1000, hot.shape, generator=generator)
    histogram[[0, 255, 256, 257, 65279, 65280, 65535]] += torch.randint(0, 3, (7,), generator=generator)
    histogram = torch.where(finite, histogram, 0)
    population = int(histogram.sum())
    ranks = torch.tensor([max((population * q + 99) // 100 - 1, 0) for q in QUANTILE_PERCENTILES])
    expected_keys = torch.searchsorted(histogram.cumsum(0), ranks, right=True)

    actual = select_from_key_histogram(histogram, dtype)

    assert torch.equal(actual.view(torch.int32), key_values(expected_keys, dtype).view(torch.int32))


def wrapped_counts(keys: list[int], start: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """count_raw, counter_wrapped and give_away in one thread's order, on a table whose word 2 holds `start`
    counts of the keys 4 and 5: the counts of every key and the totals of the groups of 32 words that the wraps
    and the table's words add up to."""
    table = {2: start[0] | start[1] << 16}
    histogram = torch.zeros(1 << 16, dtype=torch.int64)
    totals = torch.zeros(1 << 10, dtype=torch.int64)
    for key in keys:
        word, odd = key >> 1, key & 1
        before = table.get(word, 0)
        table[word] = (before + (0x10000 if odd else 1)) & 0xFFFFFFFF
        counter = 0xFFFF0000 if odd else 0xFFFF
        if before & counter == counter:
            carried = 1 - odd
            to_high = (0x10000 if before >> 16 == 0xFFFF else 0) - carried
            histogram[2 * word] += carried << 16
            histogram[2 * word + 1] += to_high
            totals[word // 32] += (carried << 16) + to_high
    for word, value in table.items():
        histogram[2 * word] += value & 0xFFFF
        histogram[2 * word + 1] += value >> 16
        totals[word // 32] += (value & 0xFFFF) + (value >> 16)
    return histogram, totals


@pytest.mark.parametrize("start", [(0, 0), (0xFFFF, 0xFFFF), (0xFFFF, 0), (0, 0xFFFF)])
@pytest.mark.parametrize("keys", [
    [5] * 150000,
    [4] * 150000,
    [4, 5] * 75000,
    [(step * 7919) % 8 for step in range(150000)],
    [65534, 65535, 0, 1] * 70000,
], ids=["one odd key", "one even key", "two keys of a word", "eight keys", "ends of the table"])
def test_counters_of_16_bits_count_exactly_beyond_their_range_by_giving_counts_away(keys, start):
    expected = torch.bincount(torch.tensor(keys), minlength=1 << 16)
    expected[4] += start[0]
    expected[5] += start[1]

    histogram, totals = wrapped_counts(keys, start)

    assert torch.equal(histogram, expected)
    assert torch.equal(
        totals, torch.bincount(torch.arange(1 << 16).repeat_interleave(expected) // 64, minlength=1 << 10)
    )


@pytest.mark.parametrize("dtype,nan,infinity", [(torch.bfloat16, 0x7FC0, 0x7F80), (torch.float16, 0x7E00, 0x7C00)])
def test_raw_pair_counts_put_the_even_value_in_the_low_half_and_count_every_element(dtype, nan, infinity):
    raw = torch.tensor([6, 7, 7, 0, 65535, 65534, 65535, 8, 6, nan, infinity, nan])

    counts = raw_pair_counts(from_raw_bits(raw, dtype))

    assert counts.shape == (1 << 15,)
    assert counts.dtype == torch.int64
    expected = {3: 2 | (2 << 32), 0: 1, 4: 1, 32767: 1 | (2 << 32), nan // 2: 2, infinity // 2: 1}
    assert {word: counts[word].item() for word in expected} == expected
    assert ((counts & 0xFFFFFFFF).sum() + (counts >> 32).sum()).item() == raw.numel()


@pytest.mark.parametrize("dtype,nan", [(torch.bfloat16, 0x7FC0), (torch.float16, 0x7E00)])
def test_group_totals_count_the_64_raw_values_of_each_group_of_32_words(dtype, nan):
    raw = torch.tensor([0, 63, 64, 64, 65535, 65472, 65471, nan, nan])

    totals = group_totals(from_raw_bits(raw, dtype))

    assert totals.shape == (1 << 10,)
    assert totals.dtype == torch.int32
    expected = {0: 2, 1: 2, 1023: 2, 1022: 1, nan >> 6: 2}
    assert {group: totals[group].item() for group in expected} == expected
    assert totals.sum().item() == raw.numel()
