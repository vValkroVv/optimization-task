"""The dispatcher's assignment counts written by one extension launch leave the recorder the words four copies
leave: int64 and int32 counts, and counts after a capacity drop that differ from those before it."""

import uuid

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

pytest.importorskip("composer", reason="Requires the training runtime")

import torch  # noqa: E402

import llmfoundry.callbacks.metric_sampler as sampler_module  # noqa: E402
from llmfoundry.callbacks.metric_sampler import MetricSampler  # noqa: E402
from metric_sampler.capture.kernels import activation_ext  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.layout_plan import DISPATCHER_METRICS  # noqa: E402
from metric_sampler.capture.recorder import RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.contract.catalog import metric_by_name  # noqa: E402

_EXPERTS = 48


class _NoExtension:
    """Stands in for the extension module in the callback only."""

    @staticmethod
    def prebuilt():
        return None


def _recorded_words(tmp_path, monkeypatch, extension: bool, counts):
    """The recorder's words and recorded-slot flags after the counts of each layer are observed."""
    if not extension:
        monkeypatch.setattr(sampler_module, "activation_ext", _NoExtension)
    sampler = MetricSampler(str(tmp_path / f"extension-{extension}"))
    layout = Layout([(metric_by_name(name), layer, 0, _EXPERTS) for layer in (0, 1) for name in DISPATCHER_METRICS])
    ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    sampler.recorder = RankRecorder(layout, ring, torch.device("cuda"))
    try:
        sampler.recorder.begin_step(False)
        for layer, layer_counts in enumerate(counts):
            sampler._observe_dispatcher(layer, *layer_counts)
        torch.cuda.synchronize()
        words = sampler.recorder.buffer.cpu().clone()
        recorded = sampler.recorder._recorded.copy()
    finally:
        sampler.recorder.close()
        ring.unlink()
    return words, recorded


@pytest.mark.skipif(not torch.cuda.is_available() or activation_ext.prebuilt() is None,
                    reason="Requires CUDA and the prebuilt extension")
def test_the_extension_writes_the_dispatcher_counts_as_four_copies_do(tmp_path, monkeypatch):
    generator = torch.Generator(device="cuda").manual_seed(3)
    before = torch.randint(0, 2**20, (_EXPERTS,), device="cuda", generator=generator)
    valid = before - torch.randint(0, 5, (_EXPERTS,), device="cuda", generator=generator)
    after = torch.minimum(before, torch.full_like(before, 2**19)).to(torch.int32)
    counts = [(before, valid, before, valid), (before, valid, after, valid.to(torch.int32))]

    words, recorded = _recorded_words(tmp_path, monkeypatch, True, counts)
    copied, copied_recorded = _recorded_words(tmp_path, monkeypatch, False, counts)

    assert torch.equal(words, copied)
    assert (recorded == copied_recorded).all() and recorded.sum() == 8
