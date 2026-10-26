"""CUDA cache release precedes optimizer monitoring and can be disabled."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchtitan.callbacks.builtin.monitoring import MonitoringCallback


@pytest.mark.parametrize(
    "enabled,step,device,has_sources,release",
    [
        (None, 50, "cuda:3", True, True),
        (True, 50, "cuda:3", True, True),
        (True, 100, "cuda:3", True, True),
        (False, 50, "cuda:3", True, False),
        (True, 49, "cuda:3", True, False),
        (True, 51, "cuda:3", True, False),
        (True, 50, "cpu", True, False),
        (True, 50, "cuda:3", False, False),
    ],
)
def test_cache_release_before_optimizer_sources(
    monkeypatch, enabled, step, device, has_sources, release
):
    backend = Mock()
    backend.filter_monitor_sources.side_effect = lambda sources: sources
    config = MonitoringCallback.Config(log_interval=50)
    if enabled is not None:
        config.empty_cuda_cache_before_optimizer = enabled
    callback = MonitoringCallback(config, sharding_backend=backend)
    events = []

    @contextmanager
    def cuda_device(selected):
        assert selected == device
        events.append("enter_device")
        yield
        events.append("exit_device")

    monkeypatch.setattr(torch.cuda, "device", cuda_device)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("release"))
    source = Mock()
    source.before_optimizer_step.side_effect = lambda state: events.append("source")
    callback.sources = [source] if has_sources else []
    state = SimpleNamespace(step=step, device=device)

    callback.before_optimizer_step(state)

    expected = ["enter_device", "release", "exit_device"] if release else []
    if has_sources:
        expected.append("source")
        source.before_optimizer_step.assert_called_once_with(state)
    assert events == expected


def test_cache_release_enabled_by_default():
    assert MonitoringCallback.Config().empty_cuda_cache_before_optimizer
