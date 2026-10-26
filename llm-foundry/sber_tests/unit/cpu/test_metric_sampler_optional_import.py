"""The sampler package is needed only when its callback or logger is configured."""

import importlib
import sys

import pytest


def test_builders_work_without_metric_sampler(monkeypatch):
    pytest.importorskip("composer")

    # Simulate a checkout or image that does not install/init the sampler submodule.
    cached_prefixes = (
        "metric_sampler.",
        "llmfoundry.callbacks.metric_sampler",
    )
    for module_name in tuple(sys.modules):
        if module_name == "metric_sampler" or module_name.startswith(cached_prefixes):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "metric_sampler", None)

    builders = importlib.import_module("llmfoundry.utils.builders")

    callback = builders.build_callback("lr_monitor", {})
    assert callback.__class__.__name__ == "LRMonitor"

    with pytest.raises(ModuleNotFoundError, match="metric_sampler"):
        builders.build_callback("metric_sampler", {})
    with pytest.raises(ModuleNotFoundError, match="metric_sampler"):
        builders.build_logger("metric_sampler", {})
