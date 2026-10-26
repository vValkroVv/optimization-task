from __future__ import annotations

import pytest
import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder
from metric_sampler.capture.stats.helpers.quantiles import exact_quantiles
from metric_sampler.capture.stats.quantiles import ExactQuantileStats
from metric_sampler.contract.metrics import Family, MetricSpec, NodeMerge, ValueType

pytestmark = pytest.mark.unit


@pytest.fixture
def recorder(shm_rings):
    quantiles = MetricSpec(114, 'test/quantiles', Family.ROUTER_VECTORS, ValueType.FLOAT32,
                          NodeMerge.PER_RANK, 'value', 'Exact test quantiles')
    counts = MetricSpec(115, 'test/quantile_counts', Family.EXPERT_COUNTS, ValueType.INT32,
                       NodeMerge.PER_RANK, 'count', 'Exact test counts')
    entropy = MetricSpec(138, 'test/entropy_quantiles', Family.EXPERT_COUNTS, ValueType.INT32,
                         NodeMerge.PER_RANK, 'bits', 'Exact float64 test quantiles')
    entropy_counts = MetricSpec(139, 'test/entropy_quantile_counts', Family.EXPERT_COUNTS, ValueType.INT32,
                                NodeMerge.PER_RANK, 'count', 'Exact entropy test counts')
    layout = Layout([(metric, layer, microbatch, width) for layer, microbatch in [(0, 0), (1, 0), (0, 1)]
                     for metric, width in [(quantiles, 9), (counts, 5), (entropy, 18), (entropy_counts, 5)]])
    ring = shm_rings(layout, slot_count=1)
    rank_recorder = RankRecorder(layout, ring, torch.device('cpu'))
    rank_recorder.begin_step(False)
    yield rank_recorder
    del rank_recorder._host_slots


def read(recorder, name, layer=0, microbatch=0):
    slot = recorder._layout.slot(name, layer, microbatch)
    words = recorder.buffer[slot.offset:slot.offset + slot.width]
    return words.view(torch.float32) if slot.metric.value_type is ValueType.FLOAT32 else words


def observe(stats, values, valid=None, layer=0, microbatch=0):
    stats.observe('test/quantiles', 'test/quantile_counts', layer, microbatch, values, valid)


def test_repeated_observation_in_one_microbatch_is_rejected_and_keeps_the_first(recorder):
    stats = ExactQuantileStats(recorder)
    observe(stats, torch.tensor([100.]))
    with pytest.raises(ValueError, match='observed twice'):
        observe(stats, torch.arange(10, dtype=torch.float32))
    assert read(recorder, 'test/quantiles').tolist() == [100.] * 9
    assert read(recorder, 'test/quantile_counts').tolist() == [1, 0, 0, 0, 0]
    stats.flush()
    observe(stats, torch.tensor([21.]), microbatch=1)
    stats.flush()
    assert read(recorder, 'test/quantiles', microbatch=1).tolist() == [21.] * 9


def test_claimed_population_is_not_observed_again_and_holds_the_microbatch(recorder):
    stats = ExactQuantileStats(recorder)
    stats.claim('test/quantiles', 0, 0)

    with pytest.raises(ValueError, match='observed twice'):
        observe(stats, torch.tensor([1.]))
    with pytest.raises(CaptureCapacityError, match='next microbatch'):
        observe(stats, torch.tensor([1.]), layer=1, microbatch=1)
    observe(stats, torch.tensor([7.]), layer=1)
    assert read(recorder, 'test/quantiles', layer=1).tolist() == [7.] * 9


def test_observation_is_snapshot_and_layer_populations_are_separate(recorder):
    stats = ExactQuantileStats(recorder)
    values = torch.tensor([7., 8., 9.], requires_grad=True)
    valid = torch.tensor([True, False, True])
    observe(stats, values, valid)
    observe(stats, torch.tensor([-5.]), layer=1)
    with torch.no_grad():
        values.fill_(100)
    valid.fill_(False)
    stats.flush()
    assert read(recorder, 'test/quantiles').tolist() == [7., 7., 7., 7., 7., 9., 9., 9., 9.]
    assert read(recorder, 'test/quantile_counts').tolist() == [2, 0, 0, 0, 1]
    assert read(recorder, 'test/quantiles', layer=1).tolist() == [-5.] * 9


def test_next_microbatch_requires_explicit_flush(recorder):
    stats = ExactQuantileStats(recorder)
    observe(stats, torch.tensor([3.]))
    with pytest.raises(CaptureCapacityError, match='next microbatch'):
        observe(stats, torch.tensor([5.]), microbatch=1)
    stats.flush()
    assert read(recorder, 'test/quantiles').tolist() == [3.] * 9


def test_empty_observation_is_recorded_not_silently_missing(recorder):
    stats = ExactQuantileStats(recorder)
    observe(stats, torch.empty(0))
    stats.flush()
    assert torch.isnan(read(recorder, 'test/quantiles')).all()
    assert read(recorder, 'test/quantile_counts').tolist() == [0] * 5


@pytest.mark.parametrize('dtype,pack', [(torch.float64, False), (torch.float32, True)])
def test_float64_must_be_explicitly_packed_without_loss(recorder, dtype, pack):
    stats = ExactQuantileStats(recorder)
    with pytest.raises(ValueError, match='explicit lossless word packing'):
        stats.observe('test/entropy_quantiles', 'test/entropy_quantile_counts', 0, 0,
                      torch.ones(2, dtype=dtype), pack_float64=pack)


@pytest.mark.parametrize('empty', [False, True])
def test_float64_quantiles_bitpack_into_integer_words_exactly(recorder, empty):
    values = torch.empty(0, dtype=torch.float64) if empty else torch.tensor(
        [1 + 2**-40, 1 + 2**-42, -1e100, 1e100, float('nan')], dtype=torch.float64)
    stats = ExactQuantileStats(recorder)
    stats.observe('test/entropy_quantiles', 'test/entropy_quantile_counts', 0, 0, values, pack_float64=True)
    stats.flush()
    expected, counts = exact_quantiles(values)
    words = read(recorder, 'test/entropy_quantiles')
    assert words.dtype == torch.int32
    torch.testing.assert_close(words.view(torch.float64), expected, rtol=0, atol=0, equal_nan=True)
    if not empty:
        assert torch.equal(words, expected.view(torch.int32))
    torch.testing.assert_close(read(recorder, 'test/entropy_quantile_counts'), counts, rtol=0, atol=0)


def test_float64_selection_recorded_by_the_caller_matches_observing_the_population(recorder):
    values = torch.tensor([1 + 2**-40, -3.5, float('nan'), 2.25], dtype=torch.float64)
    stats = ExactQuantileStats(recorder)
    stats.record_float64('test/entropy_quantiles', 'test/entropy_quantile_counts', 0, 0, *exact_quantiles(values))
    stats.observe('test/entropy_quantiles', 'test/entropy_quantile_counts', 1, 0, values, pack_float64=True)

    assert torch.equal(read(recorder, 'test/entropy_quantiles'), read(recorder, 'test/entropy_quantiles', layer=1))
    assert torch.equal(
        read(recorder, 'test/entropy_quantile_counts'), read(recorder, 'test/entropy_quantile_counts', layer=1)
    )
    with pytest.raises(ValueError, match='observed twice'):
        stats.record_float64('test/entropy_quantiles', 'test/entropy_quantile_counts', 0, 0, *exact_quantiles(values))
    with pytest.raises(CaptureCapacityError, match='next microbatch'):
        stats.record_float64('test/entropy_quantiles', 'test/entropy_quantile_counts', 0, 1, *exact_quantiles(values))
