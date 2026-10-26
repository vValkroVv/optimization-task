from __future__ import annotations

import pytest
import torch

from stand.activation_kernel_checks.key_reference import ONE_BLOCK_SELECTION, SELECTION_DTYPES
from stand.activation_kernel_checks.populations import make_tensors, special_population

pytestmark = pytest.mark.unit

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tensors() -> list[tuple[str, torch.Tensor]]:
    return make_tensors(CPU)


def test_populations_cover_unaligned_bases_and_strided_rows_beyond_the_one_block_selection(populations):
    misaligned = populations["misaligned base pointer bfloat16"]
    strided = populations["rows of 896 in a stride of 897 float16"]

    assert misaligned.data_ptr() % 16 != 0
    assert strided.stride() == (897, 1) and strided.shape == (4096, 896)
    assert all(tensor.numel() > ONE_BLOCK_SELECTION for tensor in populations.values())
    assert {tensor.dtype for tensor in populations.values()} == set(SELECTION_DTYPES)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_special_population_holds_the_extremes_the_selection_ranks_of_the_edge_check_rely_on(dtype):
    values = special_population(3 * 65536, dtype, CPU)
    largest = torch.finfo(dtype).max
    nan = values.isnan()
    negative_nan = nan & torch.signbit(values)

    assert (values == float("inf")).sum().item() == 25
    assert (values == -float("inf")).sum().item() == 25
    assert (values == largest).sum().item() == 50
    assert (values == -largest).sum().item() == 50
    assert nan.sum().item() == 50
    assert negative_nan.sum().item() == 25


def test_standard_tensors_are_drawn_from_one_seed_whatever_was_drawn_before(tensors):
    torch.randn(7)

    again = make_tensors(CPU)

    assert [name for name, _ in again] == [name for name, _ in tensors]
    for (name, tensor), (_, repeated) in zip(tensors, again):
        assert torch.equal(repeated.float().nan_to_num(7.0), tensor.float().nan_to_num(7.0)), name


def test_standard_tensors_have_distinct_names(tensors):
    names = [name for name, _ in tensors]

    assert len(names) == len(set(names))


def test_standard_tensors_include_unaligned_strided_and_overflowing_cases(tensors):
    by_name = dict(tensors)
    unaligned = by_name["unaligned rows h=300 dtype=torch.bfloat16"]
    noncontiguous = by_name["noncontiguous h=896"]

    assert unaligned.shape == (4096, 300) and unaligned.stride() == (301, 1) and unaligned.data_ptr() % 4 != 0
    assert noncontiguous.shape == (4096, 896) and noncontiguous.stride() == (1792, 2)
    assert torch.equal(by_name["finite FP32 square overflow"], torch.tensor([[1e30, -1e30]]))
    assert torch.isinf(by_name["nonfinite h=896 dtype=torch.float32"][3, :3]).all()
    assert (by_name["zeros h=896 dtype=torch.float32"][::7] == 0).all()
