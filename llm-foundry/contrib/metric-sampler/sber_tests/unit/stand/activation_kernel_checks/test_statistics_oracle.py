import pytest
import torch

from stand.activation_kernel_checks.statistics_oracle import (
    compare,
    compare_histogram,
    equal_with_nans,
    run_reference,
    sum_rounding_factor,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("detailed", [False, True])
def test_activation_oracle_checks_exponent_histogram_on_every_step(detailed):
    values = torch.tensor([[1.0, -2.0], [0.0, 4.0]])
    reference = run_reference(values, detailed)
    actual = run_reference(values, detailed)
    assert compare("point", reference, actual) == []
    assert reference.tensor("exp_hist").sum() == 4
    actual.tensor("exp_hist").zero_()
    problems = compare("point", reference, actual)
    assert len(problems) == 1
    assert problems == ["point: exp_hist differs at bins [0, 40, 41, 42]: expected [1, 1, 1, 1], got [0, 0, 0, 0]"]


def test_activation_sum_oracle_handles_cancellation_without_relative_to_small_sum():
    values = torch.tensor([[1e7, 1., -1e7]])
    reference = run_reference(values, False)
    actual = run_reference(values, False)
    assert reference.sum_reference == (1., 200000000000001.)
    actual.tensor("moments")[0] = 0
    assert compare("cancellation", reference, actual) == []
    actual.tensor("moments")[0] = 1000
    problems = compare("cancellation", reference, actual)
    assert len(problems) == 1
    assert "sum FP64=1.0, got=1000.0" in problems[0]


@pytest.mark.parametrize("values", [[[1e30, -1e30]], [[3e38, 3e38]], [[-3e38, -3e38]],
                                    [[1e30, -1e30, float("nan"), float("inf")]]])
def test_activation_fp32_overflow_keeps_exact_counts_and_fp64_reference(values):
    tensor = torch.tensor(values)
    reference = run_reference(tensor, False)
    actual = run_reference(tensor, False)
    assert reference.sum_reference[1] > torch.finfo(torch.float32).max
    assert reference.tensor("counts")[:2].tolist() == [tensor.numel() - 2, 2]
    assert reference.tensor("exp_hist").sum().item() == 2
    assert reference.tensor("token_rms_hist")[-1].item() == 1
    assert compare("overflow", reference, actual) == []
    for wrong_sum in (-float("inf"), 0.):
        actual.tensor("moments")[1] = wrong_sum
        problems = compare("overflow", reference, actual)
        assert len(problems) == 1
        assert "sumsq FP64=" in problems[0] and f"got={wrong_sum}" in problems[0]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_activation_oracle_rejects_unexpected_nonfinite_sum(invalid):
    reference = run_reference(torch.tensor([[1., -1.]]), False)
    actual = run_reference(torch.tensor([[1., -1.]]), False)
    actual.tensor("moments")[0] = invalid
    problems = compare("finite", reference, actual)
    assert len(problems) == 1 and "sum FP64=0.0" in problems[0]


def test_activation_oracle_reports_a_changed_count_amax_and_unused_word():
    reference = run_reference(torch.tensor([[1., float("nan")], [3., 2.]]), False)
    actual = run_reference(torch.tensor([[1., float("nan")], [3., 2.]]), False)
    actual.tensor("counts")[0] += 1
    actual.tensor("counts")[1] -= 1
    actual.tensor("amax")[0] = 5.
    actual.tensor("moments")[3] = 1.

    assert compare("point", reference, actual) == [
        "point: nonfinite count 1 vs 2",
        "point: finite count 3 vs 2",
        "point: unused float32 count words must remain zero",
        "point: amax 3.0 vs 5.0",
    ]


def test_activation_rounding_bound_has_finite_domain_and_uses_unit_roundoff():
    unit32 = 2. ** -24
    unit64 = 2. ** -53
    assert sum_rounding_factor(1, 1, 1) == pytest.approx(
        22 * unit32 / (1 - 22 * unit32) + unit64 / (1 - unit64), rel=1e-15)
    assert 0 < sum_rounding_factor(65536, 7168, 132) < 1e-3
    with pytest.raises(ValueError, match="rounding-bound domain"):
        sum_rounding_factor(2 ** 31, 1, 1)


def test_histogram_comparison_names_the_differing_bins_and_refuses_floating_counts():
    reference = torch.tensor([1, 2, 3])

    assert compare_histogram("point", "exp_hist", reference, reference.clone()) == []
    assert compare_histogram("point", "exp_hist", reference, torch.tensor([1, 5, 3]).to(torch.int32)) == [
        "point: exp_hist differs at bins [1]: expected [2], got [5]"
    ]
    with pytest.raises(TypeError, match="retain integer counts"):
        compare_histogram("point", "exp_hist", reference.float(), reference)


def test_nan_aware_equality_treats_every_nan_alike_and_still_tells_values_apart():
    first = torch.tensor([1.0, float("nan"), 3.0])

    assert equal_with_nans(first, torch.tensor([1.0, -float("nan"), 3.0]))
    assert not equal_with_nans(first, torch.tensor([1.0, float("nan"), 4.0]))
    assert not equal_with_nans(first, torch.tensor([1.0, 2.0, 3.0]))
