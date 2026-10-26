from __future__ import annotations

import pyarrow.parquet as pq
import pytest
import torch
import torch.utils.checkpoint as checkpoint

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder
from metric_sampler.capture.stats.activations import ActivationStats
from metric_sampler.capture.stats.helpers.quantiles import exact_quantiles
from metric_sampler.capture.stats.quantiles import ExactQuantileStats
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import ACTIVATION_LOG2_BINS, ValueType
from metric_sampler.node.aggregator import NodeAggregator, RankRing

pytestmark = pytest.mark.unit

_POINT = "residual"
_LAYER = 0
_BINS = ACTIVATION_LOG2_BINS.count


def _read(recorder: RankRecorder, name: str, layer: int, microbatch: int = 0) -> torch.Tensor:
    slot = recorder._layout.slot(name, layer, microbatch)
    view = recorder.buffer[slot.offset : slot.offset + slot.width]
    return view.view(torch.float32) if slot.metric.value_type is ValueType.FLOAT32 else view


def _point_requests(point: str, layer: int, microbatch: int = 0) -> list[tuple]:
    return [
        (metric_by_name(f"act/{point}/moments"), layer, microbatch, 4),
        (metric_by_name(f"act/{point}/counts"), layer, microbatch, 2 + 2 * _BINS),
        (metric_by_name(f"act/{point}/amax"), layer, microbatch, 1),
        (metric_by_name(f"act/{point}/power_sums"), layer, microbatch, 2),
    ]


def _make_recorder(shm_rings, requests: list[tuple]) -> RankRecorder:
    layout = Layout(requests)
    ring = shm_rings(layout, slot_count=1)
    return RankRecorder(layout, ring, torch.device("cpu"))


@pytest.fixture
def recorder(shm_rings):
    rank_recorder = _make_recorder(shm_rings, _point_requests(_POINT, _LAYER))
    yield rank_recorder
    del rank_recorder._host_slots


@pytest.fixture
def quantile_recorder(shm_rings):
    quantiles = metric_by_name(f"act/{_POINT}/quantiles")
    quantile_counts = metric_by_name(f"act/{_POINT}/quantile_counts")
    rank_recorder = _make_recorder(
        shm_rings, [*_point_requests(_POINT, _LAYER, 1), (quantiles, _LAYER, 1, 9), (quantile_counts, _LAYER, 1, 5)]
    )
    yield rank_recorder
    del rank_recorder._host_slots


def test_observe_computes_moments_and_rms_hist_on_a_manual_tensor(recorder) -> None:
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)

    values = torch.tensor([[1.0, -2.0], [0.0, float("nan")], [4.0, float("inf")]])
    stats.observe(_POINT, _LAYER, values)

    assert _read(recorder, "act/residual/moments", _LAYER).tolist() == pytest.approx([3.0, 21.0, 0.0, 0.0])
    assert _read(recorder, "act/residual/counts", _LAYER)[:2].tolist() == [2, 4]
    assert _read(recorder, "act/residual/amax", _LAYER).item() == pytest.approx(4.0)

    expected_root_mean_squares = torch.zeros(_BINS)
    expected_root_mean_squares[40] = 1
    expected_root_mean_squares[0] = 1
    expected_root_mean_squares[42] = 1
    assert _read(recorder, "act/residual/counts", _LAYER)[2 + _BINS:].tolist() == pytest.approx(
        expected_root_mean_squares.tolist()
    )

    expected_exponents = torch.zeros(_BINS)
    expected_exponents[[0, 40, 41, 42]] = 1
    assert _read(recorder, "act/residual/counts", _LAYER)[2:2 + _BINS].tolist() == expected_exponents.tolist()


def test_observe_sums_cubes_and_fourth_powers_of_finite_elements_across_calls(recorder) -> None:
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)
    stats.observe(_POINT, _LAYER, torch.tensor([[1.0, -2.0], [0.0, float("nan")], [4.0, float("inf")]]))
    stats.observe(_POINT, _LAYER, torch.tensor([[-3.0, 0.5]], dtype=torch.bfloat16))

    cubes = 1 - 8 + 64 - 27 + 0.125
    fourths = 1 + 16 + 256 + 81 + 0.0625
    assert _read(recorder, "act/residual/power_sums", _LAYER).tolist() == pytest.approx([cubes, fourths])


def test_observe_writes_exp_hist_on_a_detailed_step(recorder) -> None:
    recorder.begin_step(True)
    stats = ActivationStats(recorder)
    stats.begin_step(True)

    values = torch.tensor([[1.0, -2.0], [0.0, float("nan")], [4.0, float("inf")]])
    stats.observe(_POINT, _LAYER, values)

    expected_exponents = torch.zeros(_BINS)
    expected_exponents[0] = 1
    expected_exponents[40] = 1
    expected_exponents[41] = 1
    expected_exponents[42] = 1
    assert _read(recorder, "act/residual/counts", _LAYER)[2:2 + _BINS].tolist() == pytest.approx(
        expected_exponents.tolist()
    )


def test_observe_accumulates_moments_and_amax_across_calls_within_a_step(recorder) -> None:
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)

    stats.observe(_POINT, _LAYER, torch.tensor([[1.0, 2.0]]))
    stats.observe(_POINT, _LAYER, torch.tensor([[10.0, -3.0]]))

    assert _read(recorder, "act/residual/moments", _LAYER).tolist() == pytest.approx([10.0, 114.0, 0.0, 0.0])
    assert _read(recorder, "act/residual/amax", _LAYER).item() == pytest.approx(10.0)

    stats.observe(_POINT, _LAYER, torch.tensor([[0.5, -0.5]]))
    assert _read(recorder, "act/residual/amax", _LAYER).item() == pytest.approx(10.0)


def test_observe_resolves_a_points_slots_only_once(recorder) -> None:
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)
    resolved_names: list[str] = []
    original_resolve = recorder.resolve

    def counting_resolve(metric_name: str, layer: int, microbatch: int):
        resolved_names.append(metric_name)
        return original_resolve(metric_name, layer, microbatch)

    recorder.resolve = counting_resolve

    stats.observe(_POINT, _LAYER, torch.tensor([[1.0, 2.0]]))
    stats.observe(_POINT, _LAYER, torch.tensor([[3.0, 4.0]]))

    assert sorted(resolved_names) == sorted(
        [f"act/{_POINT}/moments", f"act/{_POINT}/counts", f"act/{_POINT}/amax", f"act/{_POINT}/power_sums"]
    )


def test_observe_caches_resolved_slots_independently_per_layer(shm_rings) -> None:
    requests = _point_requests(_POINT, 0) + _point_requests(_POINT, 1)
    recorder = _make_recorder(shm_rings, requests)
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)

    stats.observe(_POINT, 0, torch.tensor([[1.0, 2.0]]))
    stats.observe(_POINT, 1, torch.tensor([[10.0, 20.0]]))
    stats.observe(_POINT, 0, torch.tensor([[3.0, 4.0]]))

    assert _read(recorder, "act/residual/moments", 0).tolist() == pytest.approx([10.0, 30.0, 0.0, 0.0])
    assert _read(recorder, "act/residual/moments", 1).tolist() == pytest.approx([30.0, 500.0, 0.0, 0.0])


def test_observe_skips_a_row_with_no_finite_elements_in_the_rms_histogram(recorder) -> None:
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)

    values = torch.tensor([[float("nan"), float("inf")], [1.0, 1.0]])
    stats.observe(_POINT, _LAYER, values)

    assert _read(recorder, "act/residual/counts", _LAYER)[2 + _BINS:].sum().item() == 1
    assert _read(recorder, "act/residual/counts", _LAYER)[0].item() == 2
    assert _read(recorder, "act/residual/counts", _LAYER)[1].item() == 2


def test_observe_accepts_bf16_and_fp16_inputs(recorder) -> None:
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)

    stats.observe(_POINT, _LAYER, torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16))

    assert _read(recorder, "act/residual/moments", _LAYER).tolist() == pytest.approx(
        [-1.0, 5.0, 0.0, 0.0], abs=1e-2
    )


def test_observe_handles_a_noncontiguous_input(recorder) -> None:
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)
    base = torch.tensor([[1.0, 10.0, -2.0, 20.0]])
    noncontiguous = base[:, ::2]
    assert not noncontiguous.is_contiguous()

    stats.observe(_POINT, _LAYER, noncontiguous)

    assert _read(recorder, "act/residual/moments", _LAYER).tolist() == pytest.approx([-1.0, 5.0, 0.0, 0.0])


def test_observe_records_embed_on_an_ordinary_step(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, _point_requests("embed", -1))
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)

    stats.observe("embed", -1, torch.tensor([[1.0, 2.0]]))

    assert _read(recorder, "act/embed/moments", -1).tolist() == [3.0, 5.0, 0.0, 0.0]


def test_observe_on_a_detailed_point_inside_a_detailed_step_records(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, _point_requests("embed", -1))
    recorder.begin_step(True)
    stats = ActivationStats(recorder)
    stats.begin_step(True)

    stats.observe("embed", -1, torch.tensor([[1.0, 2.0]]))

    assert _read(recorder, "act/embed/moments", -1).tolist() == pytest.approx([3.0, 5.0, 0.0, 0.0])


def test_observe_under_no_grad_does_nothing(recorder) -> None:
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)

    with torch.no_grad():
        stats.observe(_POINT, _LAYER, torch.tensor([[1.0, 2.0]]))

    assert not recorder.buffer.any()


@pytest.mark.parametrize("training_forward", [False, True])
def test_explicit_custom_autograd_forward_captures_once_and_backward_reentry_is_ignored(recorder, training_forward):
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)

    class CustomForward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values):
            assert not torch.is_grad_enabled()
            stats.observe(_POINT, _LAYER, values, training_forward=training_forward)
            ctx.save_for_backward(values)
            return values * 2

        @staticmethod
        def backward(ctx, gradient):
            with torch.enable_grad():
                stats.observe(_POINT, _LAYER, ctx.saved_tensors[0] * 100, training_forward=training_forward)
            return gradient * 2

    values = torch.tensor([[1., -2.]], requires_grad=True)
    CustomForward.apply(values).sum().backward()
    assert values.grad.tolist() == [[2., 2.]]
    expected_counts = torch.zeros(2 + 2 * _BINS, dtype=torch.int32)
    if training_forward:
        expected_counts[1] = 2
        expected_counts[2 + 40] = expected_counts[2 + 41] = 1
        expected_counts[2 + _BINS + 40] = 1
    assert torch.equal(_read(recorder, "act/residual/counts", _LAYER), expected_counts)
    assert _read(recorder, "act/residual/moments", _LAYER).tolist() == (
        [-1., 5., 0., 0.] if training_forward else [0., 0., 0., 0.]
    )
    assert _read(recorder, "act/residual/amax", _LAYER).tolist() == ([2.] if training_forward else [0.])


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_observe_raises_value_error_for_an_unsupported_dtype(recorder, dtype) -> None:
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)

    with pytest.raises(ValueError):
        stats.observe(_POINT, _LAYER, torch.zeros(1, 2, dtype=dtype))


def test_observe_saves_no_tensors_for_backward_even_when_input_requires_grad(recorder) -> None:
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)
    values = torch.tensor([[1.0, -2.0]], requires_grad=True)
    saved = []

    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        stats.observe(_POINT, _LAYER, values * 1.0)

    assert saved == []


def _block(stats: ActivationStats, calls: list[int]):
    def block(tensor: torch.Tensor) -> torch.Tensor:
        doubled = tensor * 2.0
        calls.append(1)
        stats.observe(_POINT, _LAYER, doubled)
        return (doubled * doubled).sum()

    return block


def _run_forward_backward(shm_rings, run_block) -> tuple[list[float], int]:
    recorder = _make_recorder(shm_rings, _point_requests(_POINT, _LAYER))
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)
    calls: list[int] = []
    block = _block(stats, calls)
    values = torch.tensor([[1.0, -2.0], [3.0, -4.0]], requires_grad=True)

    out = run_block(block, values)
    out.backward()

    return (_read(recorder, "act/residual/moments", _LAYER).tolist()
            + _read(recorder, "act/residual/counts", _LAYER)[:2].tolist()), len(calls)


def test_observe_is_not_double_counted_under_non_reentrant_checkpointing(shm_rings) -> None:
    plain_moments, plain_calls = _run_forward_backward(shm_rings, lambda block, values: block(values))
    checkpointed_moments, checkpointed_calls = _run_forward_backward(
        shm_rings, lambda block, values: checkpoint.checkpoint(block, values, use_reentrant=False)
    )

    assert plain_calls == 1
    # The block runs once for real, once more to recompute `y` for backward; the graph-task-id
    # guard keeps the second call from being counted.
    assert checkpointed_calls == 2
    assert checkpointed_moments == pytest.approx(plain_moments)


def test_observe_is_not_double_counted_under_selective_activation_checkpointing(shm_rings) -> None:
    plain_moments, _ = _run_forward_backward(shm_rings, lambda block, values: block(values))

    def context_fn():
        return checkpoint.create_selective_checkpoint_contexts(
            lambda ctx, op, *args, **kwargs: checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
        )

    sac_moments, sac_calls = _run_forward_backward(
        shm_rings,
        lambda block, values: checkpoint.checkpoint(block, values, use_reentrant=False, context_fn=context_fn),
    )

    assert sac_calls == 2
    assert sac_moments == pytest.approx(plain_moments)


def test_observe_records_nothing_under_reentrant_checkpointing(shm_rings) -> None:
    # use_reentrant=True runs the forward under torch.no_grad() and the backward recompute inside
    # the backward graph task, so both guards skip it; hence this codebase requires
    # use_reentrant=False (or SAC) for activation checkpointing.
    reentrant_moments, reentrant_calls = _run_forward_backward(
        shm_rings, lambda block, values: checkpoint.checkpoint(block, values, use_reentrant=True)
    )

    assert reentrant_calls == 2
    assert reentrant_moments == pytest.approx([0.0] * 6)


@pytest.mark.parametrize("loaded", [object(), None])
def test_the_extension_is_resolved_once_whether_or_not_it_loads(recorder, monkeypatch, loaded) -> None:
    calls: list[int] = []
    monkeypatch.setattr(activation_ext, "prebuilt", lambda: calls.append(1) or loaded)
    stats = ActivationStats(recorder)

    assert stats._extension is loaded
    assert stats._extension is loaded

    assert len(calls) == 1


def test_activation_statistics_retain_microbatch_values_across_steps(shm_rings) -> None:
    requests = [request for microbatch in range(2) for request in
                _point_requests(_POINT, _LAYER, microbatch)]
    recorder = _make_recorder(shm_rings, requests)
    stats = ActivationStats(recorder)
    for _ in range(2):
        recorder.begin_step(False)
        stats.begin_step(False)
        for microbatch, tensor in enumerate((torch.tensor([[1.0, -2.0]]), torch.tensor([[4.0, 8.0]]))):
            stats.begin_microbatch(microbatch)
            stats.observe(_POINT, _LAYER, tensor)
        for microbatch, moments, absolute_max, bins in (
            (0, [-1.0, 5.0, 0.0, 0.0], 2.0, [40, 41]),
            (1, [12.0, 80.0, 0.0, 0.0], 8.0, [42, 43]),
        ):
            assert _read(recorder, "act/residual/moments", _LAYER, microbatch).tolist() == moments
            assert _read(recorder, "act/residual/amax", _LAYER, microbatch).item() == absolute_max
            expected = torch.zeros(_BINS)
            expected[bins] = 1
            assert _read(recorder, "act/residual/counts", _LAYER, microbatch)[2:2 + _BINS].tolist() == expected.tolist()


def test_empty_expert_tensor_records_zero_statistics(shm_rings, tmp_path) -> None:
    requests = _point_requests(_POINT, _LAYER)
    recorder = _make_recorder(shm_rings, requests)
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)
    stats.begin_microbatch(0)
    stats.observe(_POINT, _LAYER, torch.empty(0, 16))
    for metric, _, _, width in requests:
        assert _read(recorder, metric.name, _LAYER).tolist() == [0.0] * width
    assert recorder.step_end(1)
    aggregator = NodeAggregator(
        [RankRing(0, recorder._ring)], tmp_path, "node", "attempt", window_steps=1,
        rank_timeout_s=5.0, nodes_in_job=1, primary_node=True, clock=lambda: 0.0,
    )
    rows = [row for path in aggregator.poll_once() if path.suffix == ".parquet"
            for row in pq.read_table(path, partitioning=None).to_pylist()]
    expected = {metric.metric_id: [0.0] * width for metric, _, _, width in requests}
    assert {row["metric_id"]: row["values"] for row in rows if row["metric_id"] in expected} == expected


@pytest.mark.parametrize("values", [
    torch.tensor([[1.0, -2.0], [0.0, float("nan")], [4.0, float("inf")]]),
    torch.empty(0, 2),
])
def test_observe_with_quantiles_records_the_exact_quantiles_of_each_observation(quantile_recorder, values) -> None:
    recorder = quantile_recorder
    quantiles = metric_by_name(f"act/{_POINT}/quantiles")
    quantile_counts = metric_by_name(f"act/{_POINT}/quantile_counts")
    recorder.begin_step(False)
    stats = ActivationStats(recorder, ExactQuantileStats(recorder))
    stats.begin_step(False)
    stats.begin_microbatch(1)
    stats.observe(_POINT, _LAYER, values)

    expected, counts = exact_quantiles(values)
    torch.testing.assert_close(_read(recorder, quantiles.name, _LAYER, 1), expected, rtol=0, atol=0, equal_nan=True)
    assert _read(recorder, quantile_counts.name, _LAYER, 1).tolist() == counts.tolist()
    with pytest.raises(ValueError, match="observed twice"):
        stats.observe(_POINT, _LAYER, values)


@pytest.mark.parametrize("initial_elements", [0, 1])
def test_integer_population_overflow_rejected_before_read_or_write(recorder, initial_elements):
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)
    stats.observe(_POINT, _LAYER, torch.ones((initial_elements, 1)))
    before = recorder.buffer.clone()
    huge_view = torch.ones(1, 1).expand(2**31 - initial_elements, 1)
    with pytest.raises(CaptureCapacityError, match="exceeds int32 counts"):
        stats.observe(_POINT, _LAYER, huge_view)
    assert torch.equal(recorder.buffer, before)
    stats.observe(_POINT, _LAYER, torch.ones((1, 1)))
    assert _read(recorder, "act/residual/counts", _LAYER)[:2].tolist() == [0, initial_elements + 1]


def test_integer_population_resets_at_each_training_step(recorder):
    stats = ActivationStats(recorder)
    for size in (3, 2):
        recorder.begin_step(False)
        stats.begin_step(False)
        stats.observe(_POINT, _LAYER, torch.ones((size, 1)))
        assert _read(recorder, "act/residual/counts", _LAYER)[:2].tolist() == [0, size]


def test_integer_counts_are_exact_beyond_float32_integer_precision(recorder):
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)
    for _ in range(257):
        stats.observe(_POINT, _LAYER, torch.ones((257, 256), dtype=torch.bfloat16))
    expected = torch.zeros(2 + 2 * _BINS, dtype=torch.int32)
    expected[1] = 257 * 257 * 256
    expected[2 + 40] = expected[1]
    expected[2 + _BINS + 40] = 257 * 257
    stats.observe(_POINT, _LAYER, torch.ones((1, 1), dtype=torch.bfloat16))
    expected[1] += 1
    expected[2 + 40] += 1
    expected[2 + _BINS + 40] += 1
    assert torch.equal(_read(recorder, "act/residual/counts", _LAYER), expected)
    assert int(expected[1]) > 2**24 and int(expected[1]) % 2 == 1


def test_finite_activation_with_overflowing_rms_uses_last_bin(recorder):
    recorder.begin_step(False)
    stats = ActivationStats(recorder)
    stats.begin_step(False)
    stats.observe(_POINT, _LAYER, torch.tensor([[1e30, -1e30]], dtype=torch.float32))
    expected = torch.zeros(2 + 2 * _BINS, dtype=torch.int32)
    expected[1] = 2
    expected[2 + _BINS - 1] = 2
    expected[-1] = 1
    assert torch.equal(_read(recorder, "act/residual/counts", _LAYER), expected)
    assert _read(recorder, "act/residual/moments", _LAYER).tolist() == [0., float("inf"), 0., 0.]
