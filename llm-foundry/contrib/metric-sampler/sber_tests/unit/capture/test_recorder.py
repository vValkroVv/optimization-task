from __future__ import annotations

import numpy as np
import pytest
import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder
from metric_sampler.capture.ring import ShmRing, mask_word_count
from metric_sampler.contract.catalog import METRICS, metric_by_name
from metric_sampler.contract.metrics import ValueType

pytestmark = pytest.mark.unit

_LOSS = metric_by_name("loss/total")
_COUNTS = metric_by_name("router/tokens_per_expert")
_HIST = metric_by_name("expert_loss/hist")
_GPU_DURATION_METRIC = metric_by_name("time/forward_gpu_s")


@pytest.fixture
def layout() -> Layout:
    return Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4)])


@pytest.fixture
def detailed_layout() -> Layout:
    return Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4), (_HIST, 0, -1, 2)])


@pytest.fixture
def recorder_and_ring(shm_rings, layout: Layout):
    ring = shm_rings(layout, slot_count=2)
    recorder = RankRecorder(layout, ring, torch.device("cpu"))
    yield recorder, ring
    del recorder._host_slots


class FakeGpuEvent:
    """A stand-in for `torch.cuda.Event`: `start.elapsed_time(end)` returns the millisecond gap
    between the two timestamps, matching `torch.cuda.Event.elapsed_time`'s semantics."""

    def __init__(self, timestamp_ms: float) -> None:
        self._timestamp_ms = timestamp_ms

    def elapsed_time(self, other: FakeGpuEvent) -> float:
        return other._timestamp_ms - self._timestamp_ms


def test_begin_step_zeroes_the_active_buffer(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring
    recorder.record("loss/total", -1, -1, torch.tensor(3.5))
    recorder.record("router/tokens_per_expert", 0, -1, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    assert recorder.buffer.any()

    recorder.begin_step(False)

    assert not recorder.buffer.any()


def test_begin_step_clears_the_recorded_mask(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring
    recorder.record("loss/total", -1, -1, torch.tensor(3.5))
    assert recorder._recorded.any()

    recorder.begin_step(False)

    assert not recorder._recorded.any()


def test_step_end_mask_reflects_recorded_slots_and_survives_the_overlay(recorder_and_ring) -> None:
    recorder, ring = recorder_and_ring
    recorder.record_host("loss/total", -1, -1, 2.5)

    recorder.step_end(step=1)

    slot_index = ring.ready_slots()[0]
    payload = ring.payload(slot_index)
    mask = payload[recorder._layout.words : recorder._layout.words + mask_word_count(len(recorder._layout.slots))]
    mask = mask.view(np.uint32)

    loss_index = recorder._layout.slot_index("loss/total", -1, -1)
    counts_index = recorder._layout.slot_index("router/tokens_per_expert", 0, -1)
    assert bool((mask[loss_index // 32] >> (loss_index % 32)) & 1) is True
    assert bool((mask[counts_index // 32] >> (counts_index % 32)) & 1) is False

    loss_slot = recorder._layout.slot("loss/total", -1, -1)
    value = payload[loss_slot.offset : loss_slot.offset + loss_slot.width].view(np.float32)[0]
    assert value == pytest.approx(2.5)


def test_step_end_applies_a_mask_spanning_more_than_one_word(shm_rings) -> None:
    wide_metrics = METRICS[:33]
    wide_layout = Layout([(metric, -1, -1, 1) for metric in wide_metrics])
    ring = shm_rings(wide_layout, slot_count=1)
    recorder = RankRecorder(wide_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)
    for metric in wide_metrics:
        recorder.record_host(metric.name, -1, -1, 2.5 if metric.value_type is ValueType.FLOAT32 else 7)

    assert recorder.step_end(step=1) is True

    payload = ring.payload(ring.ready_slots()[0])
    mask = payload[wide_layout.words : wide_layout.words + mask_word_count(33)].view(np.uint32)
    assert mask_word_count(33) == 2
    assert int(mask[0]) == 0xFFFFFFFF
    assert int(mask[1]) == 0b1

    for metric in (wide_metrics[0], wide_metrics[-1]):
        slot = wide_layout.slot(metric.name, -1, -1)
        word = payload[slot.offset : slot.offset + 1]
        if metric.value_type is ValueType.FLOAT32:
            assert word.view(np.float32)[0] == pytest.approx(2.5)
        else:
            assert int(word[0]) == 7


def test_step_end_with_no_staged_scalars_skips_the_scalar_overlay_batch(shm_rings) -> None:
    wide_metrics = METRICS[:33]
    wide_layout = Layout([(metric, -1, -1, 4) for metric in wide_metrics])
    ring = shm_rings(wide_layout, slot_count=1)
    recorder = RankRecorder(wide_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)
    for metric in wide_metrics:
        tensor = (
            torch.tensor([1.0, 2.0, 3.0, 4.0]) if metric.value_type is ValueType.FLOAT32
            else torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        )
        recorder.record(metric.name, -1, -1, tensor)

    assert recorder.step_end(step=1) is True

    payload = ring.payload(ring.ready_slots()[0])
    mask = payload[wide_layout.words : wide_layout.words + mask_word_count(33)].view(np.uint32)
    assert mask_word_count(33) == 2
    assert int(mask[0]) == 0xFFFFFFFF
    assert int(mask[1]) == 0b1


def test_record_writes_float_metric_into_the_buffer(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    recorder.record("loss/total", -1, -1, torch.tensor(3.5))

    assert recorder.buffer[0:1].view(torch.float32).item() == pytest.approx(3.5)


def test_record_writes_int_metric_into_the_buffer(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    recorder.record("router/tokens_per_expert", 0, -1, torch.tensor([1, 2, 3, 4], dtype=torch.int32))

    assert recorder.buffer[1:5].tolist() == [1, 2, 3, 4]


def test_record_raises_value_error_on_width_mismatch(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    with pytest.raises(ValueError):
        recorder.record("loss/total", -1, -1, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor(3.5),
        torch.tensor([3.5]),
        torch.tensor([[3.5]]),
        torch.tensor(3.5, dtype=torch.float64),
        torch.tensor(3.5, dtype=torch.bfloat16),
        torch.tensor(3.5, requires_grad=True),
        torch.tensor(3.5, requires_grad=True) * 1.0,
    ],
)
def test_record_writes_a_scalar_of_any_shape_dtype_and_grad_mode(recorder_and_ring, value: torch.Tensor) -> None:
    recorder, _ = recorder_and_ring

    recorder.record("loss/total", -1, -1, value)

    assert recorder.float_buffer()[0:1].tolist() == [3.5]
    assert not recorder.buffer.requires_grad


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.tensor([1, 2, 3, 4], dtype=torch.int32), [1, 2, 3, 4]),
        (torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), [1, 2, 3, 4]),
        (torch.tensor([[1, 3], [2, 4]], dtype=torch.int32).t(), [1, 2, 3, 4]),
        (torch.tensor([1, 2, 3, 4], dtype=torch.int64), [1, 2, 3, 4]),
        (torch.arange(8, dtype=torch.int32)[::2], [0, 2, 4, 6]),
    ],
)
def test_record_writes_a_vector_in_row_major_order_whatever_its_layout_and_dtype(
    recorder_and_ring, value: torch.Tensor, expected: list[int]
) -> None:
    recorder, _ = recorder_and_ring

    recorder.record("router/tokens_per_expert", 0, -1, value)

    assert recorder.buffer[1:5].tolist() == expected


def test_record_rounds_a_float64_value_to_the_float32_slot_like_a_cast(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring
    value = torch.tensor(0.1, dtype=torch.float64)

    recorder.record("loss/total", -1, -1, value)

    assert recorder.float_buffer()[0].item() == value.float().item()


def test_record_leaves_the_autograd_graph_of_its_source_unchanged(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring
    source = torch.tensor(2.0, requires_grad=True)
    value = source * 3.0

    recorder.record("loss/total", -1, -1, value)
    value.backward()

    assert source.grad.item() == 3.0
    assert recorder.float_buffer()[0:1].tolist() == [6.0]


def test_record_and_accumulator_after_a_buffer_flip_write_only_the_active_buffer(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring
    first = recorder.buffer
    recorder.record("loss/total", -1, -1, torch.tensor(1.0))
    recorder.accumulator("router/tokens_per_expert", 0, -1).add_(1)
    recorder.step_end(step=1)
    second = recorder.buffer
    recorder.begin_step(False)

    recorder.record("loss/total", -1, -1, torch.tensor(2.0))
    recorder.accumulator("router/tokens_per_expert", 0, -1).add_(2)

    assert first[0:1].view(torch.float32).item() == 1.0
    assert first[1:5].tolist() == [1, 1, 1, 1]
    assert second[0:1].view(torch.float32).item() == 2.0
    assert second[1:5].tolist() == [2, 2, 2, 2]
    recorder.step_end(step=2)
    recorder.begin_step(False)
    recorder.record("loss/total", -1, -1, torch.tensor(3.0))
    recorder.accumulator("router/tokens_per_expert", 0, -1).add_(3)
    assert recorder.buffer is first
    assert first[0:1].view(torch.float32).item() == 3.0
    assert first[1:5].tolist() == [3, 3, 3, 3]
    assert second[0:1].view(torch.float32).item() == 2.0


def test_step_end_copies_buffer_into_a_free_ring_slot(recorder_and_ring) -> None:
    recorder, ring = recorder_and_ring
    recorder.record("loss/total", -1, -1, torch.tensor(1.5))
    recorder.record("router/tokens_per_expert", 0, -1, torch.tensor([1, 2, 3, 4], dtype=torch.int32))

    assert recorder.step_end(step=7) is True

    ready = ring.ready_slots()
    assert len(ready) == 1
    header = ring.header(ready[0])
    assert header.step == 7
    assert header.seq == 0
    assert header.dropped_before == 0
    payload = ring.payload(ready[0])
    assert payload[1:5].tolist() == [1, 2, 3, 4]
    assert payload[0:1].view("float32")[0] == pytest.approx(1.5)
    assert int(payload[-1]) == 7


def test_step_end_writes_the_step_number_into_the_trailing_word(recorder_and_ring) -> None:
    recorder, ring = recorder_and_ring
    recorder.record("loss/total", -1, -1, torch.tensor(1.0))

    recorder.step_end(step=42)

    payload = ring.payload(ring.ready_slots()[0])
    assert int(payload[-1]) == 42


def test_step_end_alternates_between_two_device_buffers(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring
    buffer_a = recorder.buffer

    recorder.record("loss/total", -1, -1, torch.tensor(1.0))
    recorder.step_end(step=1)
    buffer_b = recorder.buffer
    assert buffer_b is not buffer_a

    recorder.record("loss/total", -1, -1, torch.tensor(2.0))
    recorder.step_end(step=2)
    assert recorder.buffer is buffer_a


def test_full_ring_raises_and_preserves_the_step_for_retry(recorder_and_ring) -> None:
    recorder, ring = recorder_and_ring

    assert recorder.step_end(step=1) is True
    assert recorder.step_end(step=2) is True
    recorder.begin_step(False)
    recorder.record("loss/total", -1, -1, torch.tensor(7.5))
    with pytest.raises(CaptureCapacityError, match="ring is full at step 3"):
        recorder.step_end(step=3)
    assert recorder.dropped_steps == 0

    ring.release(0)
    assert recorder.step_end(step=3) is True
    header = ring.header(0)
    assert header.dropped_before == 0
    assert header.step == 3
    assert header.seq == 2
    slot = recorder._layout.slot("loss/total", -1, -1)
    assert ring.payload(0)[slot.offset : slot.offset + slot.width].view(np.float32).tolist() == [7.5]


def test_ring_occupancy_includes_writing_and_ready_slots(recorder_and_ring) -> None:
    recorder, ring = recorder_and_ring
    assert recorder.ring_capacity_slots == 2
    assert recorder.ring_used_slots == 0
    writing = ring.claim_free_slot()
    assert recorder.ring_used_slots == 1
    assert recorder.step_end(1)
    assert recorder.ring_used_slots == 2
    ring.release(writing)
    assert recorder.ring_used_slots == 1
    ring.release(ring.ready_slots()[0])
    assert recorder.ring_used_slots == 0


def test_record_host_raises_value_error_for_a_non_scalar_metric(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    with pytest.raises(ValueError):
        recorder.record_host("router/tokens_per_expert", 0, -1, 1.0)


def test_record_host_writes_an_int_scalar_metric(shm_rings) -> None:
    single_count_layout = Layout([(_COUNTS, 0, -1, 1)])
    ring = shm_rings(single_count_layout, slot_count=1)
    recorder = RankRecorder(single_count_layout, ring, torch.device("cpu"))

    recorder.record_host("router/tokens_per_expert", 0, -1, 3.0)
    recorder.step_end(step=1)

    slot = single_count_layout.slot("router/tokens_per_expert", 0, -1)
    payload = ring.payload(ring.ready_slots()[0])
    assert int(payload[slot.offset]) == 3


def test_record_host_overlay_wins_over_a_device_value_for_the_same_slot(recorder_and_ring) -> None:
    recorder, ring = recorder_and_ring
    recorder.record("loss/total", -1, -1, torch.tensor(9.0))
    recorder.record_host("loss/total", -1, -1, 2.5)

    recorder.step_end(step=1)

    slot = recorder._layout.slot("loss/total", -1, -1)
    payload = ring.payload(ring.ready_slots()[0])
    value = payload[slot.offset : slot.offset + slot.width].view(np.float32)[0]
    assert value == pytest.approx(2.5)


def test_record_host_value_does_not_persist_into_a_step_that_never_records_it(recorder_and_ring) -> None:
    recorder, ring = recorder_and_ring
    recorder.record_host("loss/total", -1, -1, 2.5)
    recorder.step_end(step=1)

    recorder.begin_step(False)
    recorder.step_end(step=2)

    slot = recorder._layout.slot("loss/total", -1, -1)
    second_slot_index = next(i for i in ring.ready_slots() if ring.header(i).step == 2)
    payload = ring.payload(second_slot_index)
    value = payload[slot.offset : slot.offset + slot.width].view(np.float32)[0]
    assert value == pytest.approx(0.0)


def test_close_requests_stop_on_the_ring(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=2)
    recorder = RankRecorder(layout, ring, torch.device("cpu"))
    observer = ShmRing.attach(ring._shm.name)

    recorder.close()

    assert observer.stop_requested is True
    observer.close()


def test_verbose_reflects_the_ring_verbose_until_step(recorder_and_ring) -> None:
    recorder, ring = recorder_and_ring

    assert recorder.verbose(10) is False

    ring.verbose_until_step = 10
    assert recorder.verbose(10) is True
    assert recorder.verbose(11) is False


def test_close_releases_host_views_before_closing_the_ring(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=2)
    recorder = RankRecorder(layout, ring, torch.device("cpu"))

    recorder.close()


def test_accumulator_writes_into_the_active_buffer_and_marks_the_slot_recorded(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    slot_view = recorder.accumulator("router/tokens_per_expert", 0, -1)
    slot_view += torch.tensor([1, 2, 3, 4], dtype=torch.int32)

    assert recorder.buffer[1:5].tolist() == [1, 2, 3, 4]
    assert recorder._recorded[recorder._layout.slot_index("router/tokens_per_expert", 0, -1)] == True  # noqa: E712


def test_accumulator_returns_a_float32_view_for_a_float_metric(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    slot_view = recorder.accumulator("loss/total", -1, -1)
    slot_view += 2.5

    assert recorder.buffer[0:1].view(torch.float32).item() == pytest.approx(2.5)


def test_accumulator_is_not_cached_across_a_buffer_flip(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    first = recorder.accumulator("loss/total", -1, -1)
    first += 1.0
    recorder.step_end(step=1)
    recorder.begin_step(False)
    second = recorder.accumulator("loss/total", -1, -1)

    assert second.data_ptr() != first.data_ptr()


def test_accumulator_on_a_detailed_slot_outside_a_detailed_step_raises_value_error(shm_rings, detailed_layout) -> None:
    ring = shm_rings(detailed_layout, slot_count=1)
    recorder = RankRecorder(detailed_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)

    with pytest.raises(ValueError):
        recorder.accumulator("expert_loss/hist", 0, -1)


def test_accumulator_on_a_detailed_slot_inside_a_detailed_step_succeeds(shm_rings, detailed_layout) -> None:
    ring = shm_rings(detailed_layout, slot_count=1)
    recorder = RankRecorder(detailed_layout, ring, torch.device("cpu"))
    recorder.begin_step(True)

    slot_view = recorder.accumulator("expert_loss/hist", 0, -1)
    slot_view += 1

    assert slot_view.tolist() == [1, 1]


def test_resolve_returns_the_slots_offset_width_and_index(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    resolved = recorder.resolve("router/tokens_per_expert", 0, -1)

    slot = recorder._layout.slot("router/tokens_per_expert", 0, -1)
    assert resolved.offset == slot.offset
    assert resolved.width == slot.width
    assert resolved.slot_index == recorder._layout.slot_index("router/tokens_per_expert", 0, -1)
    assert resolved.detailed is False
    assert resolved.value_type is ValueType.INT32


@pytest.mark.parametrize(
    ("metric", "layer", "microbatch"), [("router/tokens_per_expert", 0, -1), ("loss/total", -1, -1)]
)
def test_view_is_the_resolved_slots_typed_words_of_the_active_buffer_made_once_per_buffer(
    recorder_and_ring, metric, layer, microbatch,
) -> None:
    recorder, _ = recorder_and_ring
    resolved = recorder.resolve(metric, layer, microbatch)
    typed = recorder.buffer.view(torch.float32) if resolved.value_type is ValueType.FLOAT32 else recorder.buffer

    first = recorder.view(resolved)
    first += 3

    assert recorder.view(resolved) is first
    assert first.dtype == typed.dtype and first.shape == (resolved.width,)
    assert typed[resolved.offset : resolved.offset + resolved.width].tolist() == [3] * resolved.width
    assert recorder.accumulator(metric, layer, microbatch) is first

    recorder.step_end(step=1)
    recorder.begin_step(False)
    second = recorder.view(resolved)

    assert second.data_ptr() != first.data_ptr()
    assert recorder.view(resolved) is second
    assert recorder.view(resolved).tolist() == [0] * resolved.width


def test_resolve_reports_a_float_metrics_value_type(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    resolved = recorder.resolve("loss/total", -1, -1)

    assert resolved.value_type is ValueType.FLOAT32


def test_resolve_reports_the_metrics_detailed_flag(shm_rings, detailed_layout) -> None:
    ring = shm_rings(detailed_layout, slot_count=1)
    recorder = RankRecorder(detailed_layout, ring, torch.device("cpu"))

    resolved = recorder.resolve("expert_loss/hist", 0, -1)

    assert resolved.detailed is True


def test_float_buffer_is_a_float32_view_of_the_active_buffer(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    recorder.float_buffer()[0:1] = 2.5

    assert recorder.buffer[0:1].view(torch.float32).item() == pytest.approx(2.5)


def test_float_buffer_switches_buffers_after_step_end(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring
    first = recorder.float_buffer()

    recorder.step_end(step=1)
    recorder.begin_step(False)

    assert recorder.float_buffer().data_ptr() != first.data_ptr()


def test_write_through_a_resolved_slot_is_copied_into_the_ring_slot(recorder_and_ring) -> None:
    recorder, ring = recorder_and_ring
    resolved = recorder.resolve("loss/total", -1, -1)
    recorder.float_buffer()[resolved.offset : resolved.offset + resolved.width] = 4.5
    recorder.mark_recorded([resolved.slot_index])

    recorder.step_end(step=1)

    payload = ring.payload(ring.ready_slots()[0])
    value = payload[resolved.offset : resolved.offset + resolved.width].view(np.float32)[0]
    assert value == pytest.approx(4.5)


def test_mark_recorded_marks_every_given_slot_index(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring
    loss_index = recorder._layout.slot_index("loss/total", -1, -1)
    counts_index = recorder._layout.slot_index("router/tokens_per_expert", 0, -1)

    recorder.mark_recorded([loss_index, counts_index])

    assert recorder._recorded[loss_index] == True  # noqa: E712
    assert recorder._recorded[counts_index] == True  # noqa: E712


def test_recorded_flags_share_the_recorded_mask_across_steps(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring
    loss_index = recorder._layout.slot_index("loss/total", -1, -1)
    flags = recorder.recorded_flags()

    flags[loss_index] = True
    assert recorder._recorded[loss_index] == True  # noqa: E712
    recorder.begin_step(False)
    assert not flags.any()
    flags[loss_index] = True
    assert recorder._recorded.tolist() == flags.tolist() and recorder._recorded.sum() == 1


def test_mark_recorded_on_a_detailed_slot_outside_a_detailed_step_raises_value_error_and_marks_nothing(
    shm_rings, detailed_layout
) -> None:
    ring = shm_rings(detailed_layout, slot_count=1)
    recorder = RankRecorder(detailed_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)
    loss_index = recorder._layout.slot_index("loss/total", -1, -1)
    hist_index = recorder._layout.slot_index("expert_loss/hist", 0, -1)

    with pytest.raises(ValueError):
        recorder.mark_recorded([loss_index, hist_index])

    assert recorder._recorded[loss_index] == False  # noqa: E712
    assert recorder._recorded[hist_index] == False  # noqa: E712


def test_mark_recorded_on_a_detailed_slot_inside_a_detailed_step_succeeds(shm_rings, detailed_layout) -> None:
    ring = shm_rings(detailed_layout, slot_count=1)
    recorder = RankRecorder(detailed_layout, ring, torch.device("cpu"))
    recorder.begin_step(True)
    hist_index = recorder._layout.slot_index("expert_loss/hist", 0, -1)

    recorder.mark_recorded([hist_index])

    assert recorder._recorded[hist_index] == True  # noqa: E712


def test_record_on_a_detailed_slot_outside_a_detailed_step_raises_value_error(shm_rings, detailed_layout) -> None:
    ring = shm_rings(detailed_layout, slot_count=1)
    recorder = RankRecorder(detailed_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)

    with pytest.raises(ValueError):
        recorder.record("expert_loss/hist", 0, -1, torch.tensor([1, 2], dtype=torch.int32))


def test_record_host_on_a_detailed_slot_outside_a_detailed_step_raises_value_error(shm_rings) -> None:
    single_layout = Layout([(_HIST, 0, -1, 1)])
    ring = shm_rings(single_layout, slot_count=1)
    recorder = RankRecorder(single_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)

    with pytest.raises(ValueError):
        recorder.record_host("expert_loss/hist", 0, -1, 1.0)


def test_step_end_on_a_non_detailed_step_does_not_touch_the_detailed_region(shm_rings, detailed_layout) -> None:
    ring = shm_rings(detailed_layout, slot_count=1)
    recorder = RankRecorder(detailed_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)
    recorder.record("loss/total", -1, -1, torch.tensor(1.0))

    slot_index = ring.claim_free_slot()
    ring.payload(slot_index)[detailed_layout.non_detailed_words : detailed_layout.words] = 777
    ring.release(slot_index)

    assert recorder.step_end(step=1) is True

    payload = ring.payload(ring.ready_slots()[0])
    assert payload[detailed_layout.non_detailed_words : detailed_layout.words].tolist() == [777, 777]

    mask = payload[detailed_layout.words : detailed_layout.words + mask_word_count(len(detailed_layout.slots))]
    mask = mask.view(np.uint32)
    hist_index = detailed_layout.slot_index("expert_loss/hist", 0, -1)
    assert bool((mask[hist_index // 32] >> (hist_index % 32)) & 1) is False


def test_step_end_on_a_detailed_step_copies_the_detailed_region_too(shm_rings, detailed_layout) -> None:
    ring = shm_rings(detailed_layout, slot_count=1)
    recorder = RankRecorder(detailed_layout, ring, torch.device("cpu"))
    recorder.begin_step(True)
    recorder.record("loss/total", -1, -1, torch.tensor(1.0))
    slot_view = recorder.accumulator("expert_loss/hist", 0, -1)
    slot_view += torch.tensor([5, 9], dtype=torch.int32)

    recorder.step_end(step=1)

    hist_slot = detailed_layout.slot("expert_loss/hist", 0, -1)
    payload = ring.payload(ring.ready_slots()[0])
    assert payload[hist_slot.offset : hist_slot.offset + hist_slot.width].tolist() == [5, 9]
    assert int(payload[-1]) == 1


def test_record_gpu_duration_writes_seconds_computed_from_milliseconds(shm_rings) -> None:
    single_layout = Layout([(_GPU_DURATION_METRIC, -1, 0, 1)])
    ring = shm_rings(single_layout, slot_count=1)
    recorder = RankRecorder(single_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)

    recorder.record_gpu_duration("time/forward_gpu_s", -1, 0, FakeGpuEvent(0.0), FakeGpuEvent(15.0))
    recorder.step_end(step=1)

    slot = single_layout.slot("time/forward_gpu_s", -1, 0)
    payload = ring.payload(ring.ready_slots()[0])
    value = payload[slot.offset : slot.offset + slot.width].view(np.float32)[0]
    assert value == pytest.approx(0.015)


def test_record_gpu_duration_marks_the_slot_recorded(shm_rings) -> None:
    single_layout = Layout([(_GPU_DURATION_METRIC, -1, 0, 1)])
    ring = shm_rings(single_layout, slot_count=1)
    recorder = RankRecorder(single_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)

    recorder.record_gpu_duration("time/forward_gpu_s", -1, 0, FakeGpuEvent(0.0), FakeGpuEvent(5.0))

    assert recorder._recorded[single_layout.slot_index("time/forward_gpu_s", -1, 0)] == True  # noqa: E712


def test_record_gpu_duration_raises_value_error_for_a_non_scalar_slot(recorder_and_ring) -> None:
    recorder, _ = recorder_and_ring

    with pytest.raises(ValueError):
        recorder.record_gpu_duration("router/tokens_per_expert", 0, -1, FakeGpuEvent(0.0), FakeGpuEvent(1.0))


def test_record_gpu_duration_on_a_detailed_slot_outside_a_detailed_step_raises_value_error(shm_rings) -> None:
    single_layout = Layout([(_HIST, 0, -1, 1)])
    ring = shm_rings(single_layout, slot_count=1)
    recorder = RankRecorder(single_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)

    with pytest.raises(ValueError):
        recorder.record_gpu_duration("expert_loss/hist", 0, -1, FakeGpuEvent(0.0), FakeGpuEvent(1.0))


def test_begin_step_clears_a_pending_gpu_duration(shm_rings) -> None:
    single_layout = Layout([(_GPU_DURATION_METRIC, -1, 0, 1)])
    ring = shm_rings(single_layout, slot_count=2)
    recorder = RankRecorder(single_layout, ring, torch.device("cpu"))
    recorder.begin_step(False)
    recorder.record_gpu_duration("time/forward_gpu_s", -1, 0, FakeGpuEvent(0.0), FakeGpuEvent(5.0))
    recorder.step_end(step=1)

    recorder.begin_step(False)
    recorder.step_end(step=2)

    slot = single_layout.slot("time/forward_gpu_s", -1, 0)
    second_index = next(i for i in ring.ready_slots() if ring.header(i).step == 2)
    payload = ring.payload(second_index)
    value = payload[slot.offset : slot.offset + slot.width].view(np.float32)[0]
    assert value == pytest.approx(0.0)


@pytest.fixture
def rows_recorder(shm_rings):
    layout = Layout([(_LOSS, 2, -1, 1), (_COUNTS, 0, -1, 4), (_LOSS, 7, -1, 1), (_COUNTS, 3, -1, 4),
                     (_COUNTS, 5, -1, 2), (_HIST, 0, -1, 2)])
    ring = shm_rings(layout, slot_count=1)
    recorder = RankRecorder(layout, ring, torch.device("cpu"))
    recorder.begin_step(False)
    yield recorder
    del recorder._host_slots


def test_record_rows_writes_each_row_into_its_own_slot_and_marks_them(rows_recorder) -> None:
    rows = rows_recorder.resolve_rows("router/tokens_per_expert", [3, 0], -1)

    rows_recorder.record_rows(rows, torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.int32))

    for layer, expected in ((0, [1, 2, 3, 4]), (3, [5, 6, 7, 8])):
        slot = rows_recorder._layout.slot("router/tokens_per_expert", layer, -1)
        assert rows_recorder.buffer[slot.offset:slot.offset + 4].tolist() == expected
    indices = [rows_recorder._layout.slot_index("router/tokens_per_expert", layer, -1) for layer in (0, 3)]
    assert rows_recorder._recorded.nonzero()[0].tolist() == sorted(indices)


def test_record_rows_writes_float_metrics_through_the_float_view(rows_recorder) -> None:
    rows = rows_recorder.resolve_rows("loss/total", [7, 2], -1)

    rows_recorder.record_rows(rows, torch.tensor([[1.5], [-0.25]], dtype=torch.float64))

    assert rows_recorder.float_buffer()[[
        rows_recorder._layout.slot("loss/total", 7, -1).offset, rows_recorder._layout.slot("loss/total", 2, -1).offset
    ]].tolist() == [1.5, -0.25]


def test_resolve_rows_requires_one_width(rows_recorder) -> None:
    with pytest.raises(ValueError, match="one width"):
        rows_recorder.resolve_rows("router/tokens_per_expert", [0, 5], -1)


def test_resolve_rows_requires_a_row(rows_recorder) -> None:
    with pytest.raises(ValueError, match="no rows"):
        rows_recorder.resolve_rows("router/tokens_per_expert", [], -1)


def test_record_rows_rejects_a_shape_that_does_not_match_the_rows(rows_recorder) -> None:
    rows = rows_recorder.resolve_rows("router/tokens_per_expert", [0, 3], -1)

    with pytest.raises(ValueError, match="2 rows of 4 values"):
        rows_recorder.record_rows(rows, torch.zeros(1, 4, dtype=torch.int32))
    assert not rows_recorder._recorded.any()


def test_record_rows_rejects_detailed_rows_outside_a_detailed_step(rows_recorder) -> None:
    rows = rows_recorder.resolve_rows("expert_loss/hist", [0], -1)

    with pytest.raises(ValueError, match="detailed"):
        rows_recorder.record_rows(rows, torch.zeros(1, 2, dtype=torch.int32))
    rows_recorder.begin_step(True)
    rows_recorder.record_rows(rows, torch.tensor([[4, 9]], dtype=torch.int32))
    offset = rows_recorder._layout.slot("expert_loss/hist", 0, -1).offset
    assert rows_recorder.buffer[offset:offset + 2].tolist() == [4, 9]


def test_mark_rows_recorded_marks_every_row_and_rejects_detailed_rows_outside_a_detailed_step(rows_recorder) -> None:
    rows = rows_recorder.resolve_rows("router/tokens_per_expert", [3, 0], -1)
    detailed = rows_recorder.resolve_rows("expert_loss/hist", [0], -1)

    rows_recorder.mark_rows_recorded(rows)

    indices = [rows_recorder._layout.slot_index("router/tokens_per_expert", layer, -1) for layer in (0, 3)]
    assert rows_recorder._recorded.nonzero()[0].tolist() == sorted(indices)
    with pytest.raises(ValueError, match="detailed"):
        rows_recorder.mark_rows_recorded(detailed)


def test_record_host_words_overlay_the_slot_and_wrap_unsigned_words(rows_recorder) -> None:
    rows_recorder.record("router/tokens_per_expert", 3, -1, torch.tensor([9, 9, 9, 9], dtype=torch.int32))
    rows_recorder.record_host_words("router/tokens_per_expert", 3, -1, [1, 2**32 - 1, -5, 2**31])
    rows_recorder.step_end(step=1)

    slot = rows_recorder._layout.slot("router/tokens_per_expert", 3, -1)
    payload = rows_recorder._ring.payload(rows_recorder._ring.ready_slots()[0])
    assert payload[slot.offset:slot.offset + 4].tolist() == [1, -1, -5, -2**31]


@pytest.mark.parametrize("metric,layer,words,message", [
    ("loss/total", 2, [1], "integer words"),
    ("router/tokens_per_expert", 0, [1, 2], "4 words"),
])
def test_record_host_words_rejects_float_metrics_and_wrong_widths(rows_recorder, metric, layer, words, message) -> None:
    with pytest.raises(ValueError, match=message):
        rows_recorder.record_host_words(metric, layer, -1, words)
    assert not rows_recorder._recorded.any()


def test_record_host_rows_overlay_every_resolved_slot(rows_recorder) -> None:
    rows = rows_recorder.resolve_rows("router/tokens_per_expert", [3, 0], -1)

    rows_recorder.record_host_rows(rows, np.array([[5, 6, 7, 8], [1, 2, 3, 2**32 - 2]]))
    rows_recorder.step_end(step=1)

    payload = rows_recorder._ring.payload(rows_recorder._ring.ready_slots()[0])
    for layer, expected in ((0, [1, 2, 3, -2]), (3, [5, 6, 7, 8])):
        slot = rows_recorder._layout.slot("router/tokens_per_expert", layer, -1)
        assert payload[slot.offset:slot.offset + 4].tolist() == expected


@pytest.mark.parametrize("metric,layers,words,message", [
    ("loss/total", [2], np.zeros((1, 1)), "integer words"),
    ("router/tokens_per_expert", [0, 3], np.zeros((2, 3)), "2 rows of 4 words"),
    ("expert_loss/hist", [0], np.zeros((1, 2)), "detailed"),
])
def test_record_host_rows_rejects_float_wrong_shape_and_detailed_rows(
    rows_recorder, metric, layers, words, message,
) -> None:
    rows = rows_recorder.resolve_rows(metric, layers, -1)

    with pytest.raises(ValueError, match=message):
        rows_recorder.record_host_rows(rows, words)
    assert not rows_recorder._recorded.any()
