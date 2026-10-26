"""The extension's actlog observer records actlog calls as the Python observer does: the same recorder words and
recorded slots for every point, layer and microbatch, what the extension does not take recorded by Python, and the
same failures for a repeated population, a microbatch whose quantile populations were not flushed and a dtype
that is not supported."""

import uuid

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

pytest.importorskip("composer", reason="Requires the training runtime")

import torch  # noqa: E402

from llmfoundry.callbacks.metric_sampler import ActlogActivationObserver  # noqa: E402
from metric_sampler.capture.kernels import activation_ext  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.capture.stats.activations import ActivationStats  # noqa: E402
from metric_sampler.capture.stats.quantiles import ExactQuantileStats  # noqa: E402
from metric_sampler.contract.catalog import metric_by_name  # noqa: E402
from metric_sampler.contract.metrics import ACTIVATION_LOG2_BINS  # noqa: E402

_LAYERS = 2
_MICROBATCHES = 2
_POINT_LAYERS = {"residual": range(_LAYERS), "attn_out": range(_LAYERS), "final_norm": [-1]}
_WIDTHS = {"moments": 4, "counts": 2 + 2 * ACTIVATION_LOG2_BINS.count, "amax": 1, "power_sums": 2, "quantiles": 9,
           "quantile_counts": 5}

requires_extension = pytest.mark.skipif(not torch.cuda.is_available() or activation_ext.prebuilt() is None,
                                        reason="Requires CUDA and the prebuilt extension")


def _layout() -> Layout:
    return Layout([(metric_by_name(f"act/{point}/{name}"), layer, microbatch, width)
                   for point, layers in _POINT_LAYERS.items() for layer in layers
                   for microbatch in range(_MICROBATCHES) for name, width in _WIDTHS.items()])


def _calls(microbatch: int) -> list[tuple]:
    """A microbatch's actlog calls: every point of every layer once, a name that is not a point, a non-contiguous
    tensor, and in the second microbatch an empty final norm. Half precision only: float32 statistics are summed
    with float atomics, so two runs of either observer differ in their last bits."""
    generator = torch.Generator(device="cuda").manual_seed(microbatch)
    calls = []
    for layer in range(_LAYERS):
        residual = torch.randn(512, 256, device="cuda", generator=generator).to(torch.bfloat16) * (layer + 1)
        attention = torch.randn(256, 512, device="cuda", generator=generator).to(torch.bfloat16).t()
        calls.append((residual, "model.model.layers.{}", "_output.0", True, layer))
        calls.append((attention, "model.model.layers.{}.self_attn._attn", "_output.0", True, layer))
        calls.append((residual, "model.model.layers.{}.not_a_point", "_output.0", True, layer))
    final = torch.randn(0 if microbatch else 300, 256, device="cuda", generator=generator).to(torch.float16)
    calls.append((final, "model.norm", "_output.0", False, 0))
    return calls


class _Observed:
    """A recorder, its activation statistics and the actlog observer under test: the extension's or Python's."""

    def __init__(self, native: bool) -> None:
        layout = _layout()
        self.ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
        self.recorder = RankRecorder(layout, self.ring, torch.device("cuda"))
        self.quantiles = ExactQuantileStats(self.recorder)
        self.stats = ActivationStats(self.recorder, self.quantiles)
        python_observer = ActlogActivationObserver(self.stats, _LAYERS)
        self.observer = self.stats.native_observer(python_observer.resolve) if native else python_observer
        assert self.observer is not None and isinstance(self.observer, ActlogActivationObserver) != native
        self.recorder.begin_step(False)
        self.stats.begin_step(False)

    def microbatch(self, microbatch: int, calls: list[tuple]) -> None:
        self.stats.begin_microbatch(microbatch)
        for call in calls:
            self.observer(*call)
        self.quantiles.flush()
        self.stats.flush()

    def result(self) -> tuple[torch.Tensor, list[bool]]:
        torch.cuda.synchronize()
        return self.recorder.buffer.cpu().clone(), self.recorder._recorded.tolist()

    def close(self) -> None:
        self.recorder.close()
        self.ring.unlink()


@requires_extension
def test_the_extension_records_actlog_calls_as_the_python_observer_does():
    results = []
    for native in (True, False):
        observed = _Observed(native)
        try:
            for microbatch in range(_MICROBATCHES):
                observed.microbatch(microbatch, _calls(microbatch))
            results.append(observed.result())
        finally:
            observed.close()

    (words, recorded), (python_words, python_recorded) = results
    assert torch.equal(words, python_words)
    assert recorded == python_recorded and all(recorded)


@requires_extension
@pytest.mark.parametrize("native", [True, False])
def test_a_repeated_population_an_unflushed_microbatch_and_another_dtype_fail_as_in_python(native):
    observed = _Observed(native)
    try:
        call = _calls(0)[0]
        observed.stats.begin_microbatch(0)
        observed.observer(*call)
        with pytest.raises(ValueError, match=r"act/residual/quantiles layer 0 was observed twice in microbatch 0"):
            observed.observer(*call)
        observed.stats.begin_microbatch(1)
        with pytest.raises(CaptureCapacityError, match="quantile populations must be flushed before the next microbatch"):
            observed.observer(*call)
        with pytest.raises(ValueError, match="activation dtype torch.float64 is not supported"):
            observed.observer(call[0].double(), *call[1:])
    finally:
        observed.close()
