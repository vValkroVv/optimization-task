"""The callback hands a monitored module's call to the package's MonitorActivationStats only in a training forward
with grad: its tensors are reduced when the forward ends, their gradients when the backward reaches them."""

import types
import uuid

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("composer", reason="Requires the CPU training runtime")

import torch  # noqa: E402
from composer.callbacks.activation_monitor import ActivationMonitor  # noqa: E402

from llmfoundry.callbacks.metric_sampler import MetricSampler  # noqa: E402
from llmfoundry.models.ops.float8.fp8_tensor import Float8BlockwiseQTensor  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.recorder import RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.capture.stats.helpers.norms import amax_and_row_norm_sum  # noqa: E402
from metric_sampler.capture.stats.monitor_activations import MonitorActivationStats  # noqa: E402
from metric_sampler.contract.catalog import metric_by_name  # noqa: E402

_CATALOG = ["m:grad_input.0", "m:grad_output.0", "m:input.0", "m:output.0"]


@pytest.fixture
def sampler(tmp_path):
    sampler = MetricSampler(str(tmp_path), rank_local_monitors=True)
    sampler.monitor_capture._activation_catalog = _CATALOG
    layout = Layout([
        (metric_by_name(name), index, 0, 2)
        for name in ("monitor/activation_local", "monitor/activation_local_observations")
        for index in range(len(_CATALOG))
    ])
    ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    sampler.recorder = RankRecorder(layout, ring, torch.device("cpu"))
    sampler.recorder.begin_step(False)
    sampler.monitor_activations = MonitorActivationStats(
        sampler.recorder, _CATALOG, True, True, True, Float8BlockwiseQTensor)
    yield sampler
    sampler.recorder.close()
    ring.unlink()


@pytest.mark.parametrize(("training", "active", "grad"), [(False, True, True), (True, False, True), (True, True, False)])
def test_a_call_outside_a_training_forward_with_grad_is_not_observed(sampler, training, active, grad):
    module = torch.nn.Linear(3, 4).train(training)
    sampler._capture_forward_active, sampler._grad_forward = active, grad
    value = torch.randn(5, 3)

    sampler._observe_monitor_module("m", module, (value,), module(value))

    assert sampler.monitor_activations._pending == {}


def test_a_training_call_is_observed_and_reduced_when_the_forward_ends(sampler):
    module = torch.nn.Linear(3, 4)
    sampler._capture_forward_active = sampler._grad_forward = True
    value = torch.randn(5, 3, requires_grad=True)
    output = module(value)

    sampler._observe_monitor_module("m", module, (value,), output)
    sampler.monitor_activations.end_forward(0)
    output.sum().backward()

    pending = sampler.monitor_activations._pending[0]
    amax, norm_sum, rows = amax_and_row_norm_sum(value)
    assert pending.accumulator[_CATALOG.index("m:input.0")].tolist() == [amax.item(), norm_sum.item()]
    assert pending.counts == [[1, rows]] * 4


def _prepare_activation_monitor(tmp_path, stat_list: list[str]) -> MetricSampler:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4))
    monitor = ActivationMonitor(stat_list=stat_list)
    monitor.module_names = {model[0]: "model.0"}
    sampler = MetricSampler(str(tmp_path), rank_local_monitors=True)
    sampler.monitor_capture._prepare_activation_monitor(types.SimpleNamespace(callbacks=[monitor, sampler], model=model))
    return sampler


@pytest.mark.parametrize("stat_list", [["amax", "l2norm"], ["amax", "amean", "l2norm"]])
def test_an_activation_monitor_logging_amax_and_l2norm_is_captured_whatever_else_it_logs(tmp_path, stat_list):
    sampler = _prepare_activation_monitor(tmp_path, stat_list)

    assert sampler.monitor_capture._activation_catalog == ["0:input.0"]


def test_an_activation_monitor_without_l2norm_is_refused(tmp_path):
    with pytest.raises(ValueError, match="stat_list with amax and l2norm"):
        _prepare_activation_monitor(tmp_path, ["amax", "amean"])
