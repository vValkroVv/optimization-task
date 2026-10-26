"""The extension's weight monitor observer records every materialized weight the callback's Python observer records:
the same rows, flags and recording calls over steps of several microbatches, with a SuperTensorModule whose weights
are recorded together, and no recording for a module in eval mode or outside a capture forward."""

import sys
import types
import uuid

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

pytest.importorskip("composer", reason="Requires the training runtime")

import torch  # noqa: E402
from composer.callbacks.weight_monitor import WeightMonitor  # noqa: E402

import llmfoundry.callbacks.metric_sampler_weights as weights_module  # noqa: E402
from llmfoundry.callbacks.metric_sampler import MetricSampler  # noqa: E402
from llmfoundry.callbacks.metric_sampler_monitors import RankLocalMonitorCapture  # noqa: E402
from llmfoundry.callbacks.metric_sampler_weights import MaterializedWeights  # noqa: E402
from metric_sampler.capture.kernels import activation_ext  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.recorder import RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.contract.catalog import metric_by_name  # noqa: E402
from sber_tests.unit.cpu.test_metric_sampler_materialized_weights import _expected_rows, _Model, _SuperTensorModule  # noqa: E402

requires_extension = pytest.mark.skipif(not torch.cuda.is_available() or activation_ext.prebuilt() is None,
                                        reason="Requires CUDA and the prebuilt extension")

_METRIC = "monitor/materialized_weight_local"
# Taken before any test patches it: each recording counts its calls and records through it.
_RECORD = weights_module.record_weight_norm_rows


def _record(directory: str, native: bool, monkeypatch) -> tuple[list[object], list[int]]:
    """The recorder's words and flags after each of two steps of two microbatches, and the weight counts of the
    recording calls, with the extension's observer or, without `native`, the Python one."""
    torch.manual_seed(0)
    model = _Model().cuda()
    calls: list[int] = []

    def record(recorder, rows, weights, experts, launch_numels):
        calls.append(sum(count or 1 for count in experts))
        _RECORD(recorder, rows, weights, experts, launch_numels)

    monkeypatch.setattr(weights_module, "record_weight_norm_rows", record)
    monkeypatch.setitem(sys.modules, "giga_fsdp", types.SimpleNamespace(SuperTensorModule=_SuperTensorModule))
    sampler = MetricSampler(directory)
    monitors = sampler.monitor_capture
    monitor = WeightMonitor(interval="10ba", stat_list=["amax", "l2norm"])
    monitor.create_module_names(model)
    names = sorted(_expected_rows(model))
    monitors._parameter_indices = {name: index for index, name in enumerate(names)}
    layout = Layout([(metric_by_name(_METRIC), index, -1, 3) for index in range(len(names))])
    ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    sampler.recorder = RankRecorder(layout, ring, torch.device("cuda"))
    units = RankLocalMonitorCapture._super_tensor_units(model)
    weights = sampler.materialized_weights = MaterializedWeights(
        sampler.recorder, monitor, names, units, monitors._parameter_rows)
    native_observer = weights.native_observer(sampler) if native else None
    monitor.rank_local_observer = monitors._observe_materialized_weights if native_observer is None else native_observer
    monitor.attach_hooks(model)
    for super_tensor_module in dict.fromkeys(units.values()):
        super_tensor_module.register_forward_hook(monitors._record_super_tensor_weights, prepend=True)
    sampler._capture_forward_active = sampler._grad_forward = True
    steps: list[object] = []
    try:
        for step in range(2):
            sampler.recorder.begin_step(False)
            weights.begin_step()
            for microbatch in range(2):
                model(torch.full((2, 8), 0.5 + step + microbatch, device="cuda"))
            torch.cuda.synchronize()
            steps.append((sampler.recorder.buffer.cpu().clone(), sampler.recorder.recorded_flags().clone()))
        sampler.recorder.begin_step(False)
        weights.begin_step()
        model.head.eval()
        model(torch.ones(2, 8, device="cuda"))
        model.head.train()
        sampler._capture_forward_active = False
        model(torch.ones(2, 8, device="cuda"))
        torch.cuda.synchronize()
        steps.append((sampler.recorder.buffer.cpu().clone(), sampler.recorder.recorded_flags().clone()))
    finally:
        sampler.recorder.close()
        ring.unlink()
    return steps, calls


@requires_extension
def test_the_extensions_weight_observer_records_every_row_the_python_one_records(tmp_path, monkeypatch):
    native_steps, native_calls = _record(str(tmp_path / "native"), True, monkeypatch)
    python_steps, python_calls = _record(str(tmp_path / "python"), False, monkeypatch)

    assert native_calls == python_calls and native_calls
    for (native_words, native_flags), (python_words, python_flags) in zip(native_steps, python_steps):
        assert torch.equal(native_words, python_words)
        assert torch.equal(native_flags, python_flags)
    # Every weight in a step; in the last, none of the head's, in eval mode, nor any outside a capture forward.
    assert bool(native_steps[0][1].all()) and not bool(native_steps[-1][1].all())
