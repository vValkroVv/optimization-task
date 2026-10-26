from __future__ import annotations

import math

import pytest
import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import RankRecorder
from metric_sampler.capture.stats.helpers.norms import (
    accumulate_row_norm_stats,
    accumulate_row_norms,
    accumulate_row_norms_batch,
    amax_and_row_norm_sum,
    record_weight_norm_rows,
    weight_norm_rows,
)
from metric_sampler.contract.catalog import metric_by_name

pytestmark = pytest.mark.unit

_WEIGHTS = metric_by_name("monitor/materialized_weight_local")
_WEIGHT_ROW_WIDTH = 3


@pytest.fixture
def make_recorder(shm_rings):
    """Makes a recorder of the weight norm rows of `layers`, in a step that has begun."""
    recorders = []

    def make(layers: list[int]) -> RankRecorder:
        layout = Layout([(_WEIGHTS, layer, -1, _WEIGHT_ROW_WIDTH) for layer in layers])
        recorder = RankRecorder(layout, shm_rings(layout, slot_count=1), torch.device("cpu"))
        recorder.begin_step(False)
        recorders.append(recorder)
        return recorder

    yield make
    for recorder in recorders:
        del recorder._host_slots


def _read_row(recorder: RankRecorder, layer: int) -> torch.Tensor:
    slot = recorder._layout.slot(_WEIGHTS.name, layer, -1)
    return recorder.buffer[slot.offset : slot.offset + slot.width].view(torch.float32)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_amax_and_row_norm_sum_reduce_last_axis_rows(dtype) -> None:
    value = torch.tensor([[[3.0, -4.0], [0.0, 1.0]], [[-8.0, 6.0], [0.5, 0.0]]], dtype=dtype)

    absolute_max, norm_sum, rows = amax_and_row_norm_sum(value)

    assert (absolute_max.dtype, norm_sum.dtype, rows) == (torch.float32, torch.float64, 4)
    assert (absolute_max.item(), norm_sum.item()) == (8.0, 5.0 + 1.0 + 10.0 + 0.5)


def test_amax_and_row_norm_sum_of_a_scalar_is_one_row() -> None:
    assert [
        element.item() if isinstance(element, torch.Tensor) else element
        for element in amax_and_row_norm_sum(torch.tensor(-2.5))
    ] == [2.5, 2.5, 1]


def test_amax_and_row_norm_sum_propagate_nan() -> None:
    absolute_max, norm_sum, rows = amax_and_row_norm_sum(torch.tensor([[1.0, float("nan")], [2.0, 0.0]]))

    assert torch.isnan(absolute_max) and torch.isnan(norm_sum) and rows == 2


def test_accumulate_row_norms_takes_the_maximum_and_sum_of_every_observation_into_its_slot() -> None:
    accumulator = torch.tensor([[-math.inf, 0.0]] * 3, dtype=torch.float64)
    first = torch.tensor([[3.0, -4.0], [0.0, 1.0]])
    second = torch.tensor([[-7.0, 0.5]], dtype=torch.bfloat16)
    first_stats = accumulate_row_norms(first, accumulator, 1)
    accumulate_row_norms(second, accumulator, 1)
    accumulate_row_norm_stats(first_stats, accumulator, 2)

    absolute_max, norm_sum, _ = amax_and_row_norm_sum(first)
    assert first_stats.dtype == torch.float64 and first_stats.tolist() == [absolute_max.item(), norm_sum.item()]
    _, second_sum, _ = amax_and_row_norm_sum(second)
    assert accumulator[1].tolist() == [7.0, second_sum.item() + norm_sum.item()]
    assert accumulator[2].tolist() == [absolute_max.item(), norm_sum.item()]
    assert accumulator[0].tolist() == [-math.inf, 0.0]


def test_accumulate_row_norms_propagates_nan_and_takes_a_scalar_as_one_row() -> None:
    accumulator = torch.tensor([[-math.inf, 0.0]], dtype=torch.float64)
    accumulate_row_norms(torch.tensor(-2.5), accumulator, 0)
    assert accumulator[0].tolist() == [2.5, 2.5]
    accumulate_row_norms(torch.tensor([[1.0, float("nan")]]), accumulator, 0)
    accumulate_row_norms(torch.tensor([[9.0]]), accumulator, 0)
    assert accumulator[0].isnan().all()


@pytest.mark.parametrize("value", [
    torch.tensor([[3.0, -4.0], [0.0, 1.0]]),
    torch.tensor([[-7.0, 0.5]], dtype=torch.bfloat16),
    torch.tensor(-2.5),
    torch.tensor([[1.0, float("nan")]]),
])
def test_accumulate_row_norms_takes_the_same_statistics_whether_or_not_it_keeps_them(value) -> None:
    kept_into = torch.tensor([[-math.inf, 0.0]], dtype=torch.float64)
    dropped_into = kept_into.clone()

    kept = accumulate_row_norms(value, kept_into, 0)
    dropped = accumulate_row_norms(value, dropped_into, 0, keep_stats=False)

    assert dropped is None
    assert kept.shape == (2,)
    torch.testing.assert_close(dropped_into, kept_into, rtol=0, atol=0, equal_nan=True)


def test_accumulate_row_norms_batch_takes_each_tensor_into_all_of_its_slots() -> None:
    values = [
        torch.tensor([[3.0, -4.0], [0.0, 1.0]]),
        torch.tensor([[-7.0, 0.5]], dtype=torch.bfloat16),
        torch.tensor(-2.5),
        torch.tensor([[1.0, float("nan")]]),
    ]
    slots = [[4, 1, 2], [0], [3, 5], [6]]
    accumulator = torch.tensor([[-math.inf, 0.0]] * 7, dtype=torch.float64)
    accumulator[0] = accumulator[1] = 1.5  # slots that already hold observations

    accumulate_row_norms_batch(values, accumulator, slots)

    expected = torch.tensor([[-math.inf, 0.0]] * 7, dtype=torch.float64)
    expected[0] = expected[1] = 1.5
    for value, value_slots in zip(values, slots):
        absolute_max, norm_sum, _ = amax_and_row_norm_sum(value)
        for slot in value_slots:
            expected[slot] = torch.tensor(
                [
                    torch.maximum(expected[slot, 0], absolute_max.double()).item(),
                    norm_sum.item() + expected[slot, 1].item(),
                ],
                dtype=torch.float64,
            )
    torch.testing.assert_close(accumulator, expected, rtol=0, atol=0, equal_nan=True)
    assert accumulator[4].tolist() == accumulator[2].tolist() and accumulator[6].isnan().all()
    assert accumulator[1, 1].item() == 1.5 + amax_and_row_norm_sum(values[0])[1].item()


def test_weight_norm_rows_are_each_weights_norms_in_its_dtype() -> None:
    weights = [torch.tensor([[3.0, -4.0]], dtype=torch.bfloat16), torch.tensor([0.5, -1.5, 1e-3])]
    rows = weight_norm_rows(weights)
    expected = [[4.0, 5.0, 1.0],
                [1.5, torch.linalg.vector_norm(weights[1]).item(), 1.0]]
    assert rows.dtype == torch.float32
    torch.testing.assert_close(rows, torch.tensor(expected), rtol=0, atol=0)


def test_record_weight_norm_rows_records_each_weights_norms_into_its_row(make_recorder) -> None:
    recorder = make_recorder([4, 1])
    weights = [torch.tensor([[3.0, -4.0]], dtype=torch.bfloat16), torch.tensor([0.5, -1.5, 1e-3])]
    rows = recorder.resolve_rows(_WEIGHTS.name, [1, 4], -1)

    record_weight_norm_rows(recorder, rows, weights, [0, 0], [3, 3])

    expected = weight_norm_rows(weights)
    for index, layer in enumerate((1, 4)):
        assert _read_row(recorder, layer).tolist() == expected[index].tolist()
    assert recorder._recorded.all()


def test_record_weight_norm_rows_records_each_expert_of_a_stack_as_a_weight_of_its_own(make_recorder) -> None:
    generator = torch.Generator().manual_seed(6)
    stack = torch.randn(3, 4, 5, generator=generator).bfloat16()
    dense = torch.randn(6, generator=generator)
    recorder = make_recorder(list(range(4)))
    rows = recorder.resolve_rows(_WEIGHTS.name, [2, 0, 3, 1], -1)

    record_weight_norm_rows(recorder, rows, [dense, stack], [0, 3], [6, 20, 20, 20])

    for layer, weight in zip([2, 0, 3, 1], [dense, *stack]):
        assert _read_row(recorder, layer).tolist() == weight_norm_rows([weight.reshape(-1)])[0].tolist()
    assert recorder._recorded.all()


def test_record_weight_norm_rows_of_several_batches_records_each_weights_norms(make_recorder) -> None:
    generator = torch.Generator().manual_seed(5)
    batches = [
        [torch.randn(64, 48, generator=generator).bfloat16(), torch.randn(48, generator=generator)],
        [torch.randn(7, 5, generator=generator).half()],
        [torch.randn(1, generator=generator), torch.tensor([float("nan"), 1.0])],
    ]
    batches[0][1][3] = float("inf")
    weights = [weight for batch in batches for weight in batch]
    launch_numels = [max(weight.numel() for weight in batch) for batch in batches for _ in batch]
    recorder = make_recorder(list(range(5)))
    rows = recorder.resolve_rows(_WEIGHTS.name, [4, 0, 3, 1, 2], -1)

    record_weight_norm_rows(recorder, rows, weights, [0] * len(weights), launch_numels)

    for layer, weight in zip([4, 0, 3, 1, 2], weights):
        torch.testing.assert_close(
            _read_row(recorder, layer), weight_norm_rows([weight])[0], rtol=0, atol=0, equal_nan=True
        )
    assert recorder._recorded.all()
