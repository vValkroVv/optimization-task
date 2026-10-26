from __future__ import annotations

import math

import pytest
import torch

from metric_sampler.capture.recorder import CaptureCapacityError
from metric_sampler.capture.stats.helpers.quantiles import (
    QUANTILE_PERCENTILES,
    exact_quantiles,
    exact_quantiles_into,
    order_statistics,
    quantile_slots,
    row_entropy,
    row_margins,
    row_top_values,
)
from metric_sampler.contract.metrics import QUANTILE_COUNTS_WIDTH, QUANTILE_LEVELS

pytestmark = pytest.mark.unit


def reference(values, valid=None):
    included = values.flatten().tolist()
    mask = [True] * len(included) if valid is None else valid.flatten().tolist()
    finite = sorted(value for value, use in zip(included, mask) if use and math.isfinite(value))
    counts = [len(finite), 0, 0, 0, mask.count(False)]
    for value, use in zip(included, mask):
        if use and not math.isfinite(value):
            counts[1 if math.isnan(value) else 2 if value > 0 else 3] += 1
    selected = [finite[max((len(finite) * percentile + 99) // 100 - 1, 0)] if finite else float('nan')
                for percentile in QUANTILE_PERCENTILES]
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    return torch.tensor(selected, dtype=dtype), torch.tensor(counts, dtype=torch.int32)


def test_quantile_percentiles_are_the_catalog_levels_in_percent():
    assert QUANTILE_PERCENTILES == (0, 1, 5, 25, 50, 75, 95, 99, 100)


@pytest.mark.parametrize('float64,quantile_width', [(False, len(QUANTILE_LEVELS)), (True, 2 * len(QUANTILE_LEVELS))])
def test_quantile_slots_are_the_quantiles_then_their_population_counts(float64, quantile_width):
    assert quantile_slots('x/quantiles', 'x/quantile_counts', float64=float64) == (
        ('x/quantiles', quantile_width), ('x/quantile_counts', QUANTILE_COUNTS_WIDTH),
    )


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('masked', [False, True])
def test_quantiles_cover_all_elements_and_nonfinite_categories(dtype, masked):
    values = torch.tensor([[7, -4, float('nan'), float('inf')],
                           [0, 3, -float('inf'), 8], [2, 3, 99, -99]], dtype=dtype).T
    valid = (torch.arange(12).reshape(3, 4).T % 3 != 0) if masked else None
    actual = exact_quantiles(values, valid)
    expected = reference(values, valid)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert actual[1].sum() == values.numel()


@pytest.mark.parametrize('values,valid', [
    (torch.empty(0), None),
    (torch.empty(2, 0), torch.empty(2, 0, dtype=torch.bool)),
    (torch.tensor([float('nan'), float('inf'), -float('inf')]), None),
    (torch.tensor([1., 2.]), torch.tensor([False, False])),
    (torch.tensor(7.), None),
])
def test_empty_all_nonfinite_masked_and_scalar_populations(values, valid):
    actual = exact_quantiles(values, valid)
    expected = reference(values, valid)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize('count', [2, 99, 100, 101, 10001])
def test_inverse_cdf_uses_integer_ranks_without_interpolation_or_sampling(count):
    values = torch.arange(count, dtype=torch.float64)
    values[1] = -17
    values[-2] = count + 100
    actual, counts = exact_quantiles(values)
    expected, expected_counts = reference(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)


def test_float64_order_statistics_remain_lossless_and_detached():
    values = torch.tensor([1 + 2**-25, 1 + 2**-27, -1e100, 1e100], dtype=torch.float64, requires_grad=True)
    actual, counts = exact_quantiles(values)
    expected, _ = reference(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert counts.tolist() == [4, 0, 0, 0, 0]
    assert actual.dtype == torch.float64
    assert not actual.requires_grad


@pytest.mark.parametrize('dtype', [torch.int32, torch.bool, torch.complex64])
def test_nonfloating_population_is_rejected(dtype):
    with pytest.raises(ValueError, match='dtype'):
        exact_quantiles(torch.ones(3, dtype=dtype))


@pytest.mark.parametrize('valid', [torch.ones(3), torch.ones(2, dtype=torch.bool),
                                  torch.ones(3, dtype=torch.bool, device='meta')])
def test_mask_contract_is_enforced(valid):
    with pytest.raises(ValueError, match='same shape and device'):
        exact_quantiles(torch.ones(3), valid)


def test_individual_population_capacity_checked_without_allocating_data():
    with pytest.raises(CaptureCapacityError, match='int32 counts'):
        exact_quantiles(torch.empty(2**31, device='meta'))


@pytest.mark.parametrize('values,valid', [
    (torch.tensor([[3., float('nan')], [-1., float('inf')]]), None),
    (torch.tensor([4., 1., 2., 8.], dtype=torch.bfloat16), torch.tensor([True, False, True, True])),
    (torch.empty(0), None),
])
def test_exact_quantiles_into_writes_what_exact_quantiles_returns(values, valid):
    quantiles = torch.zeros(len(QUANTILE_PERCENTILES))
    counts = torch.full((QUANTILE_COUNTS_WIDTH,), -1, dtype=torch.int32)

    exact_quantiles_into(values, valid, quantiles, counts)

    expected_quantiles, expected_counts = exact_quantiles(values, valid)
    torch.testing.assert_close(quantiles, expected_quantiles, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(counts, expected_counts)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_order_statistics_follow_sort_order_with_nan_last(dtype):
    values = torch.tensor([[3., float('nan'), -1.], [float('inf'), 0.5, -float('inf')]], dtype=dtype)
    selected = order_statistics(values, [0, 2, 3, 4, 1, 5])
    assert selected.dtype == torch.float32
    assert selected[:5].tolist() == [-math.inf, 0.5, 3., math.inf, -1.]
    assert math.isnan(selected[5].item())


def test_order_statistics_of_float64_stay_float64():
    values = torch.tensor([1 + 2**-40, 1., 1 + 2**-42], dtype=torch.float64)
    assert order_statistics(values, [2, 1]).tolist() == [1 + 2**-40, 1 + 2**-42]


@pytest.mark.parametrize('ranks', [[-1], [3], [0, 3]])
def test_order_statistics_reject_ranks_outside_the_population(ranks):
    with pytest.raises(ValueError):
        order_statistics(torch.ones(3), ranks)


def test_row_top_values_are_the_descending_top_k_of_each_row():
    values = torch.tensor([[1., 5., 3., 5.], [float('nan'), -1., 2., 0.]])
    top = row_top_values(values, 3)
    assert top[0].tolist() == [5., 5., 3.]
    assert math.isnan(top[1, 0].item()) and top[1, 1:].tolist() == [2., 0.]


def test_row_entropy_is_the_float64_entropy_of_each_normalized_row_over_the_log_of_its_length():
    values = torch.tensor([[2., 2., 2., 2.], [1., 3., 0., 0.], [0., 5., 0., 0.], [0., 0., 0., 0.],
                           [float('nan'), 1., 1., 1.]])
    entropy = row_entropy(values)
    assert entropy.dtype == torch.float64
    expected = [1.0, -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)) / math.log(4)]
    assert entropy[:2].tolist() == pytest.approx(expected, rel=1e-15)
    assert entropy[2].item() == 0.0
    assert math.isnan(entropy[3].item()) and math.isnan(entropy[4].item())


def test_row_margins_subtract_each_value_from_its_rows_boundary_with_nan_as_infinity():
    values = torch.tensor([[3., 1., float('nan')], [-1., float('inf'), 2.]])
    margins = row_margins(values, torch.tensor([2., 0.5]))
    assert margins[0].tolist() == [-1., 1., -math.inf]
    assert margins[1].tolist() == [1.5, -math.inf, -1.5]
