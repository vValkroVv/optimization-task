"""The extension's C++ module hooks and observers take a microbatch's monitored modules as the Python ones do: the same
accumulator words and counts for forward tensors and gradients, under every name of a tensor several modules
saw, with batches flushed at the same points, a scalar as one row, and only what they do not reduce goes to
the Python reductions."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

pytest.importorskip("composer", reason="Requires the training runtime")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import llmfoundry.callbacks.metric_sampler as sampler_module  # noqa: E402
from llmfoundry.callbacks.metric_sampler import MetricSampler  # noqa: E402
from llmfoundry.models.ops.float8.fp8_tensor import Float8BlockwiseQTensor  # noqa: E402
from metric_sampler.capture.kernels import activation_ext  # noqa: E402
from metric_sampler.capture.stats import monitor_activations  # noqa: E402
from metric_sampler.capture.stats.monitor_activations import ROW_NORM_BATCH, MonitorActivationStats  # noqa: E402

_POINTS = ("input", "grad_input", "output", "grad_output")
_MODULES = ("a", "b", "c", "d", *(f"m{index}" for index in range(ROW_NORM_BATCH + 4)))
# Modules with a float32 and a bfloat16 scalar output, and one with a scalar input.
_SCALAR_MODULES = ("s", "t", "u")
_CATALOG = sorted(f"{module}:{point}.0" for module in _MODULES + _SCALAR_MODULES for point in _POINTS)


class _ToScalar(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(32, device="cuda"))
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() * self.weight).mean().to(self.dtype)


class _FromScalar(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(48, 32, device="cuda"))

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        return (scale * self.weight).to(torch.bfloat16)


class _NoExtension:
    """Stands in for the extension module in the callback and the monitored activations only: their reductions
    still use the extension."""

    @staticmethod
    def prebuilt():
        return None


def _sampler(tmp_path, name: str) -> MetricSampler:
    sampler = MetricSampler(str(tmp_path / name))
    sampler.monitor_capture._activation_catalog = list(_CATALOG)
    # Observing needs no recorder: only the flush records.
    sampler.monitor_activations = MonitorActivationStats(
        None, sampler.monitor_capture._activation_catalog, True, True, True, Float8BlockwiseQTensor)
    sampler._capture_forward_active = sampler._grad_forward = True
    return sampler


def _observed(tmp_path, monkeypatch, extension: bool):
    """The accumulator and counts after the forward and backward of the monitored modules, and the catalog names
    the Python reductions took for the extension."""
    if not extension:
        monkeypatch.setattr(sampler_module, "activation_ext", _NoExtension)
        monkeypatch.setattr(monitor_activations, "activation_ext", _NoExtension)
    sampler = _sampler(tmp_path, f"extension-{extension}")
    stats = sampler.monitor_activations
    fallbacks = []
    reduce_tensor, observe_gradient = stats._reduce_tensor, stats._observe_watched_gradient

    def reduce_and_note(accumulator, value, slots):
        fallbacks.extend(_CATALOG[slot] for slot in slots)
        reduce_tensor(accumulator, value, slots)

    def observe_and_note(microbatch, slots, gradient):
        fallbacks.extend(_CATALOG[slot] for slot in slots)
        observe_gradient(microbatch, slots, gradient)

    monkeypatch.setattr(stats, "_reduce_tensor", reduce_and_note)
    monkeypatch.setattr(stats, "_observe_watched_gradient", observe_and_note)
    torch.manual_seed(0)
    shared = (torch.randn(64, 32, device="cuda", requires_grad=True) * 2).to(torch.bfloat16)
    wide = torch.randn(16, 8, device="cuda", dtype=torch.float64, requires_grad=True) * 3
    modules = {name: torch.nn.Linear(32, 32).cuda().to(torch.bfloat16) for name in _MODULES}
    # A float64 module: neither its tensors nor its gradients are ones the extension reduces.
    modules["c"] = torch.nn.Linear(8, 4).cuda().to(torch.float64)
    loss = 0
    for name in _MODULES:
        # a and b share their input; d is called twice, so its slots are seen again within a batch.
        source = wide if name == "c" else shared if name in "ab" else torch.randn(
            48, 32, device="cuda", requires_grad=True).to(torch.bfloat16)
        modules[name].register_forward_hook(sampler.monitor_hook(name))
        for _ in range(2 if name == "d" else 1):
            loss = loss + modules[name](source).float().square().sum()
    scalar = torch.randn((), device="cuda", requires_grad=True) * 2
    for name, module in (("s", _ToScalar(torch.float32)), ("t", _ToScalar(torch.bfloat16)), ("u", _FromScalar())):
        source = scalar if name == "u" else shared
        module.register_forward_hook(sampler.monitor_hook(name))
        loss = loss + module(source).float().square().sum()
    stats.end_forward(0)
    loss.backward()
    torch.cuda.synchronize()
    pending = stats._pending[0]
    counts = np.zeros((len(_CATALOG), 2), dtype=np.int64)
    if pending.counts is not None:
        counts = np.array(pending.counts, dtype=np.int64)
    if pending.extension_counts is not None:
        counts += pending.extension_counts.numpy()
    return pending.accumulator.cpu(), counts, pending.observations is not None, sorted(set(fallbacks))


@pytest.mark.skipif(not torch.cuda.is_available() or activation_ext.prebuilt() is None,
                    reason="Requires CUDA and the prebuilt extension")
def test_the_extension_observes_monitored_modules_as_the_python_observers_do(tmp_path, monkeypatch):
    accumulator, counts, extension, fallbacks = _observed(tmp_path, monkeypatch, extension=True)
    python_accumulator, python_counts, python_extension, _ = _observed(tmp_path, monkeypatch, extension=False)

    assert extension and not python_extension
    # Only the float64 module's tensors and gradients are left to Python: scalars are the extension's.
    assert fallbacks == ["c:grad_input.0", "c:grad_output.0", "c:input.0", "c:output.0"]
    assert torch.equal(accumulator.view(torch.int64), python_accumulator.view(torch.int64))
    assert counts.tolist() == python_counts.tolist()
    index = {name: position for position, name in enumerate(_CATALOG)}
    assert counts[index["d:input.0"]].tolist() == [2, 96] and counts[index["d:grad_output.0"]].tolist() == [2, 96]
    assert counts[index["a:grad_input.0"]].tolist() == counts[index["b:grad_input.0"]].tolist() == [1, 64]
    assert counts[index["c:input.0"]].tolist() == counts[index["c:grad_output.0"]].tolist() == [1, 16]
    for name in ("s:output.0", "s:grad_output.0", "t:output.0", "t:grad_output.0", "u:input.0", "u:grad_input.0"):
        assert counts[index[name]].tolist() == [1, 1], name
    assert (counts[:, 0] > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available() or activation_ext.prebuilt() is None,
                    reason="Requires CUDA and the prebuilt extension")
def test_hooks_another_monitor_registered_on_a_watched_tensor_still_see_its_gradient():
    source = torch.randn(8, 4, device="cuda", requires_grad=True)
    value = source * 2
    seen = []
    value.register_hook(lambda grad: seen.append(("before", grad.clone())))
    accumulator = torch.zeros(1, 2, dtype=torch.float64, device="cuda")
    counts = torch.zeros(1, 2, dtype=torch.int64)
    observations = activation_ext.prebuilt().MonitorObservations(
        accumulator, counts, ROW_NORM_BATCH, lambda value, slots: None, lambda slots, grad: None, None)
    observations.observe_module((value,), (), [-1, 0])
    value.register_hook(lambda grad: seen.append(("after", grad.clone())))

    (value * 3).sum().backward()

    assert [name for name, _ in seen] == ["before", "after"]
    assert all(torch.equal(grad, torch.full_like(grad, 3.0)) for _, grad in seen)
    assert counts.tolist() == [[1, 8]]
    assert torch.equal(source.grad, torch.full_like(source, 6.0))


@pytest.mark.skipif(not torch.cuda.is_available() or activation_ext.prebuilt() is None,
                    reason="Requires CUDA and the prebuilt extension")
def test_a_tensor_modified_in_place_after_the_extension_observed_it_is_reported_instead_of_reduced(tmp_path):
    sampler = _sampler(tmp_path, "modified")
    module = torch.nn.Linear(32, 32).cuda().to(torch.bfloat16)
    value = torch.randn(4, 32, device="cuda").to(torch.bfloat16)
    sampler._observe_monitor_module("b", module, (value,), module(value))
    value.mul_(2)

    with pytest.raises(RuntimeError, match="b:input.0 was modified in place"):
        sampler.monitor_activations.end_forward(0)

    observations = sampler.monitor_activations._pending[0].observations
    assert observations is not None and sampler._monitor_observations is observations
    assert observations.flush() == -1
