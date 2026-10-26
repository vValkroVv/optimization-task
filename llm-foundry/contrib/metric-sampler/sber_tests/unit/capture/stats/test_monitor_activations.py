from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder
from metric_sampler.capture.stats import monitor_activations
from metric_sampler.capture.stats.helpers.norms import amax_and_row_norm_sum
from metric_sampler.capture.stats.monitor_activations import ROW_NORM_BATCH, MonitorActivationStats
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import INT32_MAX

pytestmark = pytest.mark.unit

_NAMES = ["a:input.0", "b:input.0", "a:grad_output.0"]
_METRICS = ("monitor/activation_local", "monitor/activation_local_observations")


class _Quantized(torch.Tensor):
    """Stands in for a quantized tensor type the monitor skips."""


@pytest.fixture
def make_stats(shm_rings):
    """Makes the stats of a catalog over a recorder of its rows for microbatches 0 and 1."""
    recorders = []

    def make(
        names: list[str], log_output: bool = True, log_gradients: bool = True, log_grad_input: bool = True,
    ) -> MonitorActivationStats:
        layout = Layout([
            (metric_by_name(metric), entry, microbatch, 2)
            for metric in _METRICS for entry in range(len(names)) for microbatch in (0, 1)
        ])
        recorder = RankRecorder(layout, shm_rings(layout, slot_count=1), torch.device("cpu"))
        recorder.begin_step(False)
        recorders.append(recorder)
        return MonitorActivationStats(recorder, names, log_output, log_gradients, log_grad_input, _Quantized)

    yield make
    for recorder in recorders:
        del recorder._host_slots


@pytest.fixture
def stats(make_stats):
    return make_stats(_NAMES)


@pytest.fixture
def batches(monkeypatch):
    """What each call of the batched reduction was given: its tensors and the slots they go into."""
    calls = []
    reduce_batch = monitor_activations.accumulate_row_norms_batch

    def record(values, accumulator, slots):
        calls.append((list(values), [list(value_slots) for value_slots in slots]))
        reduce_batch(values, accumulator, slots)

    monkeypatch.setattr(monitor_activations, "accumulate_row_norms_batch", record)
    return calls


def _read(recorder: RankRecorder, microbatch: int) -> tuple[list[list[float]], list[list[int]]]:
    """The [absolute max, mean row norm] and [observations, rows] of each catalog entry in the step the recorder
    handed to its ring."""
    ring = recorder._ring
    payload = ring.payload(ring.ready_slots()[0])
    values, counts = [], []
    for entry in range(len(_NAMES)):
        offset = recorder._layout.slot(_METRICS[0], entry, microbatch).offset
        values.append(payload[offset:offset + 2].view(np.float32).tolist())
        offset = recorder._layout.slot(_METRICS[1], entry, microbatch).offset
        counts.append(payload[offset:offset + 2].tolist())
    return values, counts


def test_forward_and_gradient_observations_accumulate_the_absolute_max_and_row_norm_sum(stats):
    value = torch.randn(3, 5, 4)
    absolute_max, norm_sum, rows = amax_and_row_norm_sum(value)

    stats._observe_forward_tensor("a:input.0", 0, value)
    stats.end_forward(0)
    stats.observe_gradient(["a:grad_output.0"], 0, value)

    accumulator = stats._pending[0].accumulator
    assert accumulator[0].tolist() == accumulator[2].tolist() == [absolute_max.item(), norm_sum.item()]
    assert stats._pending[0].counts == [[1, rows], [0, 0], [1, rows]]


def test_a_forward_tensor_observed_by_two_modules_is_accumulated_into_both_by_one_reduction(stats, batches):
    value = torch.randn(7, 4)
    absolute_max, norm_sum, rows = amax_and_row_norm_sum(value)

    stats._observe_forward_tensor("a:input.0", 1, value)
    stats._observe_forward_tensor("b:input.0", 1, value)
    stats.end_forward(1)

    accumulator = stats._pending[1].accumulator
    assert accumulator[0].tolist() == accumulator[1].tolist() == [absolute_max.item(), norm_sum.item()]
    assert stats._pending[1].counts[:2] == [[1, rows], [1, rows]]
    assert batches == [([value], [[0, 1]])]


def test_observed_tensors_are_not_reduced_until_the_forward_ends(stats, batches):
    stats._observe_forward_tensor("a:input.0", 0, torch.randn(3, 4))

    unreduced = [-math.inf, 0.0]
    assert batches == [] and stats._pending[0].accumulator[0].tolist() == unreduced
    stats.end_forward(0)
    assert len(batches) == 1 and stats._pending[0].accumulator[0].tolist() != unreduced
    stats.end_forward(0)
    assert len(batches) == 1


def test_a_slot_observed_again_flushes_the_batch_first_so_its_observations_keep_their_order(stats, batches):
    first, second, other = torch.randn(2, 4), torch.randn(3, 4), torch.randn(5, 4)

    stats._observe_forward_tensor("a:input.0", 0, first)
    stats._observe_forward_tensor("b:input.0", 0, other)
    stats._observe_forward_tensor("a:input.0", 0, second)
    stats.end_forward(0)

    assert batches == [([first, other], [[0], [1]]), ([second], [[0]])]
    accumulator = stats._pending[0].accumulator
    assert accumulator[0, 1].item() == amax_and_row_norm_sum(second)[1].item() + amax_and_row_norm_sum(first)[1].item()


def test_the_batch_is_flushed_when_a_tensor_past_the_bound_arrives(make_stats, batches):
    names = [f"m{index}:input.0" for index in range(ROW_NORM_BATCH + 2)]
    stats = make_stats(names)
    values = [torch.randn(2, 3) for _ in names]

    for name, value in zip(names, values[:ROW_NORM_BATCH]):
        stats._observe_forward_tensor(name, 0, value)
    # Another name for a tensor already queued does not make the batch longer.
    stats._observe_forward_tensor(names[ROW_NORM_BATCH], 0, values[0])
    assert batches == []

    stats._observe_forward_tensor(names[ROW_NORM_BATCH + 1], 0, values[ROW_NORM_BATCH])

    assert len(batches) == 1 and batches[0][0] == values[:ROW_NORM_BATCH]
    assert batches[0][1] == [[0, ROW_NORM_BATCH], *([index] for index in range(1, ROW_NORM_BATCH))]
    stats.end_forward(0)
    assert len(batches) == 2 and batches[1] == ([values[ROW_NORM_BATCH]], [[ROW_NORM_BATCH + 1]])


def test_a_tensor_modified_in_place_after_it_was_observed_is_reported_instead_of_reduced(stats, batches):
    value = torch.randn(3, 4)
    stats._observe_forward_tensor("b:input.0", 0, value)
    value.mul_(2)

    with pytest.raises(RuntimeError, match=r"b:input\.0 was modified in place"):
        stats.end_forward(0)

    assert batches == []
    stats.end_forward(0)


@pytest.mark.parametrize("skipped", [torch.zeros(0, 4), torch.randn(2, 4).as_subclass(_Quantized)])
def test_an_empty_or_quantized_forward_tensor_is_not_observed(stats, batches, skipped):
    stats._observe_forward_tensor("a:input.0", 0, skipped)
    stats.end_forward(0)

    assert batches == [] and 0 not in stats._pending


def test_a_gradient_observed_under_two_names_is_accumulated_into_both(stats):
    value = torch.randn(7, 4)
    absolute_max, norm_sum, rows = amax_and_row_norm_sum(value)

    stats.observe_gradient(["a:grad_output.0", "b:input.0"], 1, value)

    accumulator = stats._pending[1].accumulator
    assert accumulator[1].tolist() == accumulator[2].tolist() == [absolute_max.item(), norm_sum.item()]
    assert stats._pending[1].counts == [[0, 0], [1, rows], [1, rows]]


@pytest.mark.parametrize("skipped", [torch.zeros(0, 4), torch.randn(2, 4).as_subclass(_Quantized)])
def test_an_empty_or_quantized_gradient_is_not_observed(stats, skipped):
    stats.observe_gradient(["a:grad_output.0"], 0, skipped)

    assert 0 not in stats._pending


def test_a_scalar_counts_as_one_row(stats):
    stats.observe_gradient(["a:grad_output.0"], 0, torch.tensor(-3.0))

    assert stats._pending[0].counts[2] == [1, 1]
    assert stats._pending[0].accumulator[2].tolist() == [3.0, 3.0]


def test_counts_past_int32_stop_capture(stats):
    stats.observe_gradient(["a:grad_output.0"], 0, torch.randn(2, 4))
    stats._pending[0].counts[2][1] = INT32_MAX - 1

    with pytest.raises(CaptureCapacityError, match="exceed Int32"):
        stats.observe_gradient(["a:grad_output.0"], 0, torch.randn(2, 4))


def test_modules_that_share_an_input_watch_its_gradient_with_one_hook_for_all_of_their_names(make_stats):
    points = ("input", "grad_input", "output", "grad_output")
    stats = make_stats(sorted(f"{module}:{point}.0" for module in "ab" for point in points))
    shared = torch.randn(5, 3, requires_grad=True) * 2
    modules = {"a": torch.nn.Linear(3, 4), "b": torch.nn.Linear(3, 4)}
    outputs = {name: module(shared) for name, module in modules.items()}

    for name in modules:
        assert stats.observe_module(name, 0, (shared,), outputs[name]) is None
    stats._flush_row_norms(0)
    observers = stats._gradient_observers
    hooks = sum(len(tensor._backward_hooks) for tensor in (shared, *outputs.values()))
    (outputs["a"].sum() + 2 * outputs["b"].sum()).backward()

    assert hooks == 3 and len(observers) == 3
    assert observers[id(shared)][1].names == ["a:grad_input.0", "b:grad_input.0"]
    pending = stats._pending[0]
    index = stats._indices
    # Each output's gradient is a constant, each row has four of them; the shared input's gradient is
    # observed under both names.
    assert pending.accumulator[index["a:grad_output.0"]].tolist() == [1.0, 5 * 2.0]
    assert pending.accumulator[index["b:grad_output.0"]].tolist() == [2.0, 5 * 4.0]
    grad_inputs = pending.accumulator[[index["a:grad_input.0"], index["b:grad_input.0"]]].tolist()
    assert grad_inputs[0] == grad_inputs[1]
    assert pending.counts[index["a:grad_input.0"]] == pending.counts[index["b:grad_input.0"]] == [1, 5]
    assert pending.counts[index["a:grad_output.0"]] == pending.counts[index["b:grad_output.0"]] == [1, 5]
    # The shared input was reduced once for the forward observations of both modules.
    assert pending.accumulator[index["a:input.0"]].tolist() == pending.accumulator[index["b:input.0"]].tolist()


def test_the_end_of_the_forward_forgets_its_watched_tensors_and_a_later_tensor_gets_a_hook_of_its_own(stats):
    first = torch.randn(2, 4, requires_grad=True) * 1
    stats._watch_gradient("a:grad_output.0", 0, first)
    stats.end_forward(0)
    assert stats._gradient_observers == {}

    second = torch.randn(2, 4, requires_grad=True) * 1
    stats._watch_gradient("a:grad_output.0", 0, second)
    # A dead tensor whose id a new tensor took is not the new tensor.
    stats._gradient_observers[id(second)] = (lambda: None, stats._gradient_observers[id(second)][1])
    stats._watch_gradient("b:input.0", 0, second)
    second.sum().backward()

    assert len(second._backward_hooks) == 2
    assert stats._pending[0].counts == [[0, 0], [1, 2], [1, 2]]


@pytest.mark.parametrize(("log_output", "log_gradients", "log_grad_input", "observed"), [
    (True, True, True, ["m:grad_input.0", "m:grad_output.0", "m:input.0", "m:output.0", "m:output.1"]),
    (True, True, False, ["m:grad_output.0", "m:input.0", "m:output.0", "m:output.1"]),
    (False, True, True, ["m:grad_input.0", "m:grad_output.0", "m:input.0"]),
    (True, False, True, ["m:input.0", "m:output.0", "m:output.1"]),
])
def test_a_module_call_observes_the_floating_tensors_its_flags_select(
    make_stats, log_output, log_gradients, log_grad_input, observed,
):
    points = ("input", "grad_input", "output", "grad_output")
    names = sorted(f"m:{point}.{index}" for point in points for index in range(3))
    stats = make_stats(names, log_output, log_gradients, log_grad_input)
    source = torch.randn(4, 3, requires_grad=True) * 1
    outputs = (source * 2, (source * 3).detach(), torch.ones(2, dtype=torch.int64))

    stats.observe_module("m", 0, (source, torch.ones(3, dtype=torch.int64), "flag"), outputs)
    stats.end_forward(0)
    (outputs[0].sum() + source.sum()).backward()

    counts = stats._pending[0].counts
    assert [name for name, (calls, _) in zip(names, counts) if calls] == observed


def test_the_flush_records_each_entry_and_nan_for_one_without_observations(stats):
    value = torch.randn(3, 4)
    gradient = torch.randn(5, 4)
    stats._observe_forward_tensor("a:input.0", 0, value)
    stats.end_forward(0)
    stats.observe_gradient(["a:grad_output.0"], 0, gradient)

    stats.flush(0)
    stats.flush(1)

    recorder = stats._recorder
    assert recorder.step_end(step=1)
    values, counts = _read(recorder, 0)
    expected = [
        [np.float32(amax_and_row_norm_sum(tensor)[0].item()).item(),
         np.float32(amax_and_row_norm_sum(tensor)[1].item() / rows).item()]
        for tensor, rows in ((value, 3), (gradient, 5))
    ]
    assert values[0] == expected[0] and values[2] == expected[1]
    assert all(math.isnan(statistic) for statistic in values[1])
    assert counts == [[1, 3], [0, 0], [1, 5]]
    assert 0 not in stats._pending
    unobserved_values, unobserved_counts = _read(recorder, 1)
    assert all(math.isnan(statistic) for row in unobserved_values for statistic in row)
    assert unobserved_counts == [[0, 0]] * len(_NAMES)


def test_rows_are_resolved_once_per_metric_and_microbatch(stats):
    stats.flush(0)
    resolved = dict(stats._rows)
    stats.flush(0)

    assert stats._rows == resolved and len(resolved) == 2
