from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from stand.activation_kernel_checks.key_reference import KEY_BINS, SELECTION_DTYPES, group_totals, raw_pair_counts
from stand.activation_kernel_checks.quantile_checks import check_raw_counts

pytestmark = pytest.mark.unit


def kernel_scratch(values: torch.Tensor) -> torch.Tensor:
    """The scratch the statistics kernels leave for a bf16/fp16 population: the raw pair counts, then the totals
    of the groups of 32 words, two int32 to a word."""
    return torch.cat((raw_pair_counts(values), group_totals(values).view(torch.int64)))


def raw_count_problems(values: torch.Tensor, scratch: torch.Tensor) -> list[str]:
    def observe(tensor, *arguments):
        refining = len(arguments) == 9
        return torch.zeros(10, dtype=torch.int32) if refining else scratch

    return check_raw_counts([("population", values)], SimpleNamespace(observe=observe))


@pytest.mark.parametrize("dtype", SELECTION_DTYPES)
def test_raw_count_check_accepts_the_counts_and_the_group_totals_of_the_population(dtype):
    values = torch.randn(100, 100).to(dtype)

    assert raw_count_problems(values, kernel_scratch(values)) == []


@pytest.mark.parametrize("dtype", SELECTION_DTYPES)
def test_raw_count_check_reports_a_wrong_group_total(dtype):
    values = torch.randn(100, 100).to(dtype)
    scratch = kernel_scratch(values)
    scratch[KEY_BINS // 2] += 1

    problems = raw_count_problems(values, scratch)

    assert len(problems) == 1
    assert "totals of the groups of 32 words differ at [0]" in problems[0]


@pytest.mark.parametrize("dtype", SELECTION_DTYPES)
def test_raw_count_check_reports_a_wrong_word_count_and_a_scratch_without_totals(dtype):
    values = torch.randn(100, 100).to(dtype)
    scratch = kernel_scratch(values)
    scratch[0] += 1

    assert "raw counts population: differ at words [0]" in raw_count_problems(values, scratch)[0]
    assert "expected 33280 of int64" in raw_count_problems(values, scratch[: KEY_BINS // 2])[0]
