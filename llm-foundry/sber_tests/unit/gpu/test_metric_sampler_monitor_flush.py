"""The extension writes a microbatch's monitored activation statistics as the torch operations do: the same
float32 words for every entry, NaN for an entry without observations, and one row for an observation without
any."""

import uuid

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

pytest.importorskip("composer", reason="Requires the training runtime")

import math  # noqa: E402

import torch  # noqa: E402

from llmfoundry.models.ops.float8.fp8_tensor import Float8BlockwiseQTensor  # noqa: E402
from metric_sampler.capture.kernels import activation_ext  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.recorder import RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.capture.stats import monitor_activations  # noqa: E402
from metric_sampler.capture.stats.monitor_activations import MonitorActivationStats, _MonitorActivations  # noqa: E402
from metric_sampler.contract.catalog import metric_by_name  # noqa: E402

_ENTRIES = 300


class _NoExtension:
    """Stands in for the extension module in the monitored activations only."""

    @staticmethod
    def prebuilt():
        return None


def _counts_and_accumulator():
    generator = torch.Generator().manual_seed(5)
    observations = torch.randint(0, 4, (_ENTRIES,), generator=generator)
    rows = torch.randint(0, 3000, (_ENTRIES,), generator=generator) * (observations > 0)
    observations[:3] = torch.tensor([0, 1, 2])
    rows[:3] = torch.tensor([0, 0, 1])
    accumulator = torch.stack((torch.rand(_ENTRIES, generator=generator, dtype=torch.float64) * 7,
                               torch.rand(_ENTRIES, generator=generator, dtype=torch.float64) * 1e6), dim=1)
    accumulator[observations == 0] = torch.tensor([-math.inf, 0.0], dtype=torch.float64)
    return torch.stack((observations, rows), dim=1).tolist(), accumulator.cuda()


def _flushed_words(tmp_path, monkeypatch, extension: bool):
    """The recorder's words and recorded-slot flags after one microbatch's monitored activations are flushed."""
    if not extension:
        monkeypatch.setattr(monitor_activations, "activation_ext", _NoExtension)
    names = ("monitor/activation_local_observations", "monitor/activation_local")
    layout = Layout([(metric_by_name(name), entry, 0, 2) for name in names for entry in range(_ENTRIES)])
    ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    recorder = RankRecorder(layout, ring, torch.device("cuda"))
    catalog = [f"m{entry}:output.0" for entry in range(_ENTRIES)]
    stats = MonitorActivationStats(recorder, catalog, True, True, True, Float8BlockwiseQTensor)
    counts, accumulator = _counts_and_accumulator()
    try:
        recorder.begin_step(False)
        stats._pending[0] = _MonitorActivations(accumulator, counts)
        stats.flush(0)
        torch.cuda.synchronize()
        values = recorder.float_buffer()[stats._rows_for(names[1], 0).word_positions].cpu()
        words = recorder.buffer.cpu().clone()
        recorded = recorder._recorded.copy()
    finally:
        recorder.close()
        ring.unlink()
    return words, recorded, values.reshape(_ENTRIES, 2), counts, accumulator.cpu()


@pytest.mark.skipif(not torch.cuda.is_available() or activation_ext.prebuilt() is None,
                    reason="Requires CUDA and the prebuilt extension")
def test_the_extension_writes_the_monitored_activations_as_the_torch_operations_do(tmp_path, monkeypatch):
    words, recorded, values, counts, accumulator = _flushed_words(tmp_path, monkeypatch, True)
    torch_words, torch_recorded, _, _, _ = _flushed_words(tmp_path, monkeypatch, False)

    assert torch.equal(words, torch_words)
    assert (recorded == torch_recorded).all() and recorded.sum() == 2 * _ENTRIES
    assert values[0].isnan().all()
    # An observation without rows counts as one row.
    assert values[1].tolist() == [float(accumulator[1, 0].float()), float(accumulator[1, 1].float())]
    assert values[2].tolist() == [float(accumulator[2, 0].float()), float(accumulator[2, 1].float())]
    observed = torch.tensor([observations > 0 for observations, _ in counts])
    assert values[~observed].isnan().all() and not values[observed].isnan().any()
