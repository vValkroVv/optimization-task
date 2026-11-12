from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from torchtitan.callbacks.base import ObserverState
from torchtitan.callbacks.sources.trainer_source import TrainerSource


def state(step=0, inputs=0, valid=0):
    return ObserverState(
        event=None,
        step=step,
        model_parts=(),
        device="cpu",
        parallel_dims=SimpleNamespace(world_size=4, dp_replicate=1, dp_shard=4),
        config=SimpleNamespace(training=SimpleNamespace(local_batch_size=1)),
        ntokens_seen=valid,
        ninput_tokens_seen=inputs,
    )


def metrics(value):
    return {record.name: record.value for record in value._records}


def test_global_input_throughput_includes_unsupervised_positions():
    source = TrainerSource(
        TrainerSource.Config(log_throughput=True, throughput_window_size=2)
    )
    initial = state()
    with patch(
        "torchtitan.callbacks.sources.trainer_source.time.perf_counter", return_value=0
    ):
        source.fit_start(initial)
    first = state(1, inputs=400, valid=40)
    with patch(
        "torchtitan.callbacks.sources.trainer_source.time.perf_counter", return_value=2
    ):
        source.after_optimizer_step(first)
    assert metrics(first)["throughput/device/step/input_tokens_per_sec"] == 50
    assert "throughput/input_tokens_per_sec" not in metrics(first)
    second = state(2, inputs=1200, valid=40)
    with patch(
        "torchtitan.callbacks.sources.trainer_source.time.perf_counter", return_value=6
    ):
        source.after_optimizer_step(second)
    result = metrics(second)
    assert result["throughput/input_tokens_per_sec"] == 200
    assert result["throughput/device/input_tokens_per_sec"] == 50
    assert result["throughput/tokens_per_sec"] == pytest.approx(40 / 6)
    assert result["throughput/step/tokens_per_sec"] == 0

    # Resume starts a new window even when the input counter is run-local.
    with patch(
        "torchtitan.callbacks.sources.trainer_source.time.perf_counter",
        return_value=100,
    ):
        source.fit_start(state(20, inputs=0, valid=40))
    resumed = state(21, inputs=800, valid=50)
    with patch(
        "torchtitan.callbacks.sources.trainer_source.time.perf_counter",
        return_value=104,
    ):
        source.after_optimizer_step(resumed)
    assert metrics(resumed)["throughput/device/step/input_tokens_per_sec"] == 50
    assert "throughput/input_tokens_per_sec" not in metrics(resumed)


def test_input_metrics_absent_for_trainers_without_input_counter():
    source = TrainerSource(
        TrainerSource.Config(log_throughput=True, throughput_window_size=1)
    )
    with patch(
        "torchtitan.callbacks.sources.trainer_source.time.perf_counter", return_value=0
    ):
        source.fit_start(replace(state(), ninput_tokens_seen=None))
    value = replace(state(1, valid=20), ninput_tokens_seen=None)
    with patch(
        "torchtitan.callbacks.sources.trainer_source.time.perf_counter", return_value=2
    ):
        source.after_optimizer_step(value)
    assert metrics(value)["throughput/tokens_per_sec"] == 10
    assert not any("input_tokens" in key for key in metrics(value))
