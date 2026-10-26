from __future__ import annotations

import math
from decimal import Decimal, localcontext

import pyarrow.parquet as pq
import pytest
import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import RankRecorder
from metric_sampler.capture.stats.token_loss import TokenLossStats
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import (
    ENTROPY_TOKEN_STRIDE,
    LOSS_POSITION_BUCKETS,
    SAMPLE_LOSS_BINS,
    WORST_SAMPLE_COUNT,
    WORST_SAMPLE_FIELDS,
    ValueType,
)
from metric_sampler.node.aggregator import NodeAggregator, RankRing

pytestmark = pytest.mark.unit

_NONFINITE = metric_by_name("loss/nonfinite_tokens")
_TOKEN_MAX = metric_by_name("loss/token_max")
_TOKEN_HIST = metric_by_name("loss/token_hist_by_stream")
_MTP_HIST = metric_by_name("loss/mtp_token_hist")
_STREAM_SUM = metric_by_name("loss/stream_sum")
_STREAM_SUMSQ = metric_by_name("loss/stream_sumsq")
_STREAM_COUNT = metric_by_name("loss/stream_count")
_STREAM_SAMPLES = metric_by_name("loss/stream_samples")
_POSITION_SUM = metric_by_name("loss/position_sum")
_POSITION_COUNT = metric_by_name("loss/position_count")
_SAMPLE_HIST = metric_by_name("loss/sample_hist_by_stream")
_EXPERT_SUM = metric_by_name("expert_loss/sum")
_EXPERT_SUMSQ = metric_by_name("expert_loss/sumsq")
_EXPERT_COUNT = metric_by_name("expert_loss/count")
_EXPERT_HIST = metric_by_name("expert_loss/hist")
_EXPERT_STREAM_COUNT = metric_by_name("expert_loss/stream_count")
_EXPERT_STREAM_SUM = metric_by_name("expert_loss/stream_sum")
_ENTROPY_HIST = metric_by_name("router/entropy_hist")
_ROUTER_TOKENS = metric_by_name("router/tokens_per_expert")
_ROUTER_STREAM_TOKENS = metric_by_name("router/stream_tokens_per_expert")
_WORST_INFO = metric_by_name("loss/worst_sample_info")
_WORST_STATS = metric_by_name("loss/worst_sample_stats")

_EXPERTS = 3
_TOP_K = 2
_STREAMS = 2
_ROUTED_LAYERS = {0: 0, 5: 1}
_MTP_LAYERS = [5]


def _read(recorder: RankRecorder, name: str, layer: int, microbatch: int = 0) -> torch.Tensor:
    slot = recorder._layout.slot(name, layer, microbatch)
    view = recorder.buffer[slot.offset : slot.offset + slot.width]
    if slot.metric.value_type is ValueType.FLOAT32:
        view = view.view(torch.float32)
    return view


def _make_recorder(shm_rings, requests: list[tuple]) -> RankRecorder:
    layout = Layout(requests)
    ring = shm_rings(layout, slot_count=1)
    return RankRecorder(layout, ring, torch.device("cpu"))


def _single_stream_requests() -> list[tuple]:
    return [
        (_NONFINITE, -1, 0, 1),
        (_TOKEN_MAX, -1, 0, 1),
        (_TOKEN_HIST, -1, 0, 256),
        (_STREAM_SUM, -1, 0, 1),
        (_STREAM_SUMSQ, -1, 0, 1),
        (_STREAM_COUNT, -1, 0, 1),
        (_STREAM_SAMPLES, -1, 0, 1),
        (_POSITION_SUM, -1, 0, LOSS_POSITION_BUCKETS),
        (_POSITION_COUNT, -1, 0, LOSS_POSITION_BUCKETS),
    ]


@pytest.fixture
def main_layout_requests() -> list[tuple]:
    return [
        (_NONFINITE, -1, 0, 1),
        (_TOKEN_MAX, -1, 0, 1),
        (_TOKEN_HIST, -1, 0, _STREAMS * 256),
        (_MTP_HIST, 5, 0, 256),
        (_STREAM_SUM, -1, 0, _STREAMS),
        (_STREAM_SUMSQ, -1, 0, _STREAMS),
        (_STREAM_COUNT, -1, 0, _STREAMS),
        (_STREAM_SAMPLES, -1, 0, _STREAMS),
        (_POSITION_SUM, -1, 0, LOSS_POSITION_BUCKETS),
        (_POSITION_COUNT, -1, 0, LOSS_POSITION_BUCKETS),
        (_EXPERT_SUM, 0, 0, _EXPERTS),
        (_EXPERT_SUMSQ, 0, 0, _EXPERTS),
        (_EXPERT_COUNT, 0, 0, _EXPERTS),
        (_ENTROPY_HIST, 0, 0, 32),
        (_ROUTER_TOKENS, 0, 0, _EXPERTS),
        (_ROUTER_STREAM_TOKENS, 0, 0, _EXPERTS * _STREAMS),
        (_EXPERT_SUM, 5, 0, _EXPERTS),
        (_EXPERT_SUMSQ, 5, 0, _EXPERTS),
        (_EXPERT_COUNT, 5, 0, _EXPERTS),
        (_ENTROPY_HIST, 5, 0, 32),
        (_ROUTER_TOKENS, 5, 0, _EXPERTS),
        (_ROUTER_STREAM_TOKENS, 5, 0, _EXPERTS * _STREAMS),
    ]


@pytest.fixture
def sample_recorder(shm_rings, main_layout_requests):
    requests = [request for request in main_layout_requests if request[1] == -1]
    requests.extend([
        (_SAMPLE_HIST, -1, 0, _STREAMS * SAMPLE_LOSS_BINS.count),
        (_WORST_INFO, -1, 0, WORST_SAMPLE_COUNT * WORST_SAMPLE_FIELDS),
        (_WORST_STATS, -1, 0, WORST_SAMPLE_COUNT * 2),
    ])
    def create():
        return _make_recorder(shm_rings, requests)

    return create


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_worst_samples_prioritize_nonfinite_and_keep_identifiers(sample_recorder, dtype) -> None:
    recorder = sample_recorder()
    recorder.begin_step(True)
    stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=2)
    stats.begin_step(True)
    stats.begin_microbatch(0, torch.tensor([0, 1, 0, 1, 1, 0]),
                          torch.tensor([[10, i] for i in range(6)], dtype=dtype))
    loss = torch.tensor([[[1., 3., 100.], [9., 9., 9.], [float("inf"), 5., float("nan")],
                          [0., 0., 0.], [7., 11., 6.], [100., 100., 100.]]], requires_grad=True)
    valid = torch.tensor([[[True, True, False], [True] * 3, [True] * 3,
                           [True] * 3, [True] * 3, [False] * 3]])
    stats.observe_loss(loss, valid)
    assert _read(recorder, _WORST_INFO.name, -1, 0).view(4, 6).tolist() == [
        [0, 10, 2, 0, 1, 2], [1, 10, 1, 0, 3, 0],
        [1, 10, 4, 1, 3, 0], [0, 10, 0, 1, 2, 0],
    ]
    result = _read(recorder, _WORST_STATS.name, -1, 0)
    assert result.tolist() == [5., 5., 9., 9., 8., 11., 2., 3.]
    assert not result.requires_grad
    assert not recorder.buffer.requires_grad


@pytest.mark.parametrize("loss, valid, counts, expected_stats, position", [
    ([0., 0.], [False, True], [1, 0], [0., 0.], 1),
    ([float("nan"), float("inf")], [True, True], [0, 2], [0., -float("inf")], 0),
    ([float("nan"), 100.], [False, False], [0, 0], [0., -float("inf")], 0),
])
def test_worst_samples_padding_and_validity(sample_recorder, loss, valid, counts, expected_stats, position) -> None:
    recorder = sample_recorder()
    recorder.begin_step(True)
    stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=2)
    stats.begin_step(True)
    stats.begin_microbatch(0, torch.tensor([1]), torch.tensor([[7, 9]]))
    stats.observe_loss(torch.tensor([[loss]]), torch.tensor([[valid]]))
    assert _read(recorder, _WORST_INFO.name, -1, 0).tolist() == [1, 7, 9, position, *counts] + [0] * 18
    assert _read(recorder, _WORST_STATS.name, -1, 0).tolist() == expected_stats + [0.] * 6


@pytest.mark.parametrize("sample_ids, error", [
    (torch.tensor([[1, 2, 3]]), "for each sample"),
    (torch.tensor([[1., 2.]]), "integer tensors"),
])
def test_worst_samples_reject_invalid_identifiers(sample_recorder, sample_ids, error) -> None:
    recorder = sample_recorder()
    recorder.begin_step(True)
    stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=2)
    stats.begin_step(True)
    stats.begin_microbatch(0, torch.tensor([0]), sample_ids)
    with pytest.raises(ValueError, match=error):
        stats.observe_loss(torch.ones(1, 1, 2), torch.ones(1, 1, 2, dtype=torch.bool))


@pytest.mark.parametrize("detailed, sample_ids", [(False, torch.tensor([[1, 2]])), (True, None)])
def test_worst_samples_are_absent_without_detailed_step_or_identifiers(
    shm_rings, main_layout_requests, detailed, sample_ids,
) -> None:
    requests = [request for request in main_layout_requests if request[1] == -1]
    requests.append((_SAMPLE_HIST, -1, 0, _STREAMS * SAMPLE_LOSS_BINS.count))
    recorder = _make_recorder(shm_rings, requests)
    recorder.begin_step(detailed)
    stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=2)
    stats.begin_step(detailed)
    stats.begin_microbatch(0, torch.tensor([0]), sample_ids)
    stats.observe_loss(torch.ones(1, 1, 2), torch.ones(1, 1, 2, dtype=torch.bool))
    assert _read(recorder, _STREAM_COUNT.name, -1).tolist() == [2, 0]


def test_worst_samples_survive_ring_and_parquet_per_rank_and_microbatch(
    shm_rings, main_layout_requests, tmp_path,
) -> None:
    requests = [request for request in main_layout_requests if request[1] == -1]
    requests = [(metric, layer, microbatch, width) for microbatch in range(2) for metric, layer, _, width in requests]
    requests.extend((_SAMPLE_HIST, -1, microbatch, _STREAMS * SAMPLE_LOSS_BINS.count) for microbatch in range(2))
    requests.extend((metric, -1, microbatch, width) for microbatch in (0, 1) for metric, width in (
        (_WORST_INFO, WORST_SAMPLE_COUNT * WORST_SAMPLE_FIELDS), (_WORST_STATS, WORST_SAMPLE_COUNT * 2),
    ))
    layout = Layout(requests)
    rings = [shm_rings(layout, slot_count=1) for _ in range(2)]
    for rank, ring in enumerate(rings):
        recorder = RankRecorder(layout, ring, torch.device("cpu"))
        recorder.begin_step(True)
        stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=2)
        stats.begin_step(True)
        for microbatch in (0, 1):
            stats.begin_microbatch(microbatch, torch.tensor([rank]), torch.tensor([[2**32 - 1, microbatch + 2**31]]))
            stats.observe_loss(
                torch.tensor([[[float(10 * rank + microbatch), 2.]]]), torch.ones(1, 1, 2, dtype=torch.bool)
            )
        assert recorder.step_end(100)
    aggregator = NodeAggregator(
        [RankRing(rank, ring) for rank, ring in enumerate(rings)], tmp_path, "n0", "0",
        window_steps=1, rank_timeout_s=5., nodes_in_job=1, primary_node=True, clock=lambda: 0.,
    )
    paths = aggregator.poll_once()
    info_rows = pq.read_table(
        next(path for path in paths if path.name == "expert_counts.parquet"), partitioning=None
    ).to_pylist()
    value_rows = pq.read_table(
        next(path for path in paths if path.name == "router_vectors.parquet"), partitioning=None
    ).to_pylist()
    info = {(row["rank"], row["microbatch"]): row["values"]
            for row in info_rows if row["metric_id"] == _WORST_INFO.metric_id}
    values = {(row["rank"], row["microbatch"]): row["values"]
              for row in value_rows if row["metric_id"] == _WORST_STATS.metric_id}
    assert info == {
        (rank, microbatch): [rank, 2**32 - 1, microbatch + 2**31, int(rank == 0), 2, 0] + [0] * 18
        for rank in (0, 1) for microbatch in (0, 1)
    }
    assert values == {
        (rank, microbatch): [(10 * rank + microbatch + 2.) / 2, max(10 * rank + microbatch, 2.)] + [0.] * 6
        for rank in (0, 1) for microbatch in (0, 1)
    }


def test_observe_loss_computes_token_and_expert_statistics(shm_rings, main_layout_requests) -> None:
    recorder = _make_recorder(shm_rings, main_layout_requests)
    recorder.begin_step(False)
    stats = TokenLossStats(recorder, _ROUTED_LAYERS, _MTP_LAYERS, experts=_EXPERTS, top_k=_TOP_K, streams=_STREAMS)
    stats.begin_step(False)
    stats.begin_microbatch(0, torch.tensor([0, 1]))

    topk_layer0 = torch.tensor([[0, 1], [1, 2], [0, 2], [1, 1]], dtype=torch.int64)
    topk_layer5 = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 2]], dtype=torch.int64)
    stats.observe_routing(0, topk_layer0, torch.rand(4, _EXPERTS))
    stats.observe_routing(5, topk_layer5, torch.rand(4, _EXPERTS))

    loss = torch.tensor(
        [
            [[0.1, 0.3], [float("nan"), 0.5]],
            [[0.2, float("inf")], [0.05, 0.4]],
        ]
    )
    labels_valid = torch.tensor(
        [
            [[True, True], [True, False]],
            [[True, True], [True, True]],
        ]
    )

    stats.observe_loss(loss, labels_valid)

    assert _read(recorder, "loss/nonfinite_tokens", -1, 0).item() == 2
    assert _read(recorder, "loss/token_max", -1, 0).item() == pytest.approx(0.3)

    expected_hist = torch.zeros(_STREAMS * 256, dtype=torch.int32)
    expected_hist[0] = 1
    expected_hist[2] = 1
    assert _read(recorder, "loss/token_hist_by_stream", -1, 0).tolist() == expected_hist.tolist()

    assert _read(recorder, "loss/stream_sum", -1, 0).tolist() == pytest.approx([0.4, 0.0])
    assert _read(recorder, "loss/stream_sumsq", -1, 0).tolist() == pytest.approx([0.1, 0.0])
    assert _read(recorder, "loss/stream_count", -1, 0).tolist() == [2, 0]
    assert _read(recorder, "loss/stream_samples", -1, 0).tolist() == [1, 1]

    # sequence length 2 buckets position 0 into bucket 0 and position 1 into bucket 32 (64 // 2 == 32);
    # the second sample's tokens are both invalid there and contribute nothing.
    expected_position_sum = torch.zeros(LOSS_POSITION_BUCKETS)
    expected_position_sum[0] = 0.1
    expected_position_sum[32] = 0.3
    assert _read(recorder, "loss/position_sum", -1, 0).tolist() == pytest.approx(expected_position_sum.tolist())
    expected_position_count = torch.zeros(LOSS_POSITION_BUCKETS, dtype=torch.int32)
    expected_position_count[0] = 1
    expected_position_count[32] = 1
    assert _read(recorder, "loss/position_count", -1, 0).tolist() == expected_position_count.tolist()

    expected_mtp_hist = torch.zeros(256, dtype=torch.int32)
    expected_mtp_hist[0] = 1
    expected_mtp_hist[1] = 1
    expected_mtp_hist[3] = 1
    assert _read(recorder, "loss/mtp_token_hist", 5, 0).tolist() == expected_mtp_hist.tolist()

    assert _read(recorder, "expert_loss/sum", 0, 0).tolist() == pytest.approx([0.1, 0.4, 0.3])
    assert _read(recorder, "expert_loss/sumsq", 0, 0).tolist() == pytest.approx([0.01, 0.10, 0.09])
    assert _read(recorder, "expert_loss/count", 0, 0).tolist() == [1, 2, 1]

    assert _read(recorder, "expert_loss/sum", 5, 0).tolist() == pytest.approx([0.25, 0.05, 1.0])
    assert _read(recorder, "expert_loss/count", 5, 0).tolist() == [2, 1, 3]


def test_router_stream_counts_cover_all_selected_rows_even_when_labels_are_ignored(
    shm_rings, main_layout_requests,
) -> None:
    recorder = _make_recorder(shm_rings, main_layout_requests)
    recorder.begin_step(False)
    stats = TokenLossStats(recorder, _ROUTED_LAYERS, _MTP_LAYERS, _EXPERTS, _TOP_K, _STREAMS)
    stats.begin_step(False)
    stats.begin_microbatch(0, torch.tensor([0, 1]))

    main = torch.tensor([[0, 1], [1, 2], [0, 2], [1, 1]])
    mtp = torch.tensor([[2, 0], [2, 2], [0, 1], [0, 1]])
    stats.observe_routing(0, main, torch.ones(4, _EXPERTS))
    stats.observe_routing(5, mtp, torch.ones(4, _EXPERTS))
    # All selected rows still count even though every label is ignored and losses are non-finite.
    labels_valid = torch.tensor([[[True, False], [True, True]], [[False, True], [True, False]]])
    stats.observe_loss(torch.full((2, 2, 2), float("nan")), labels_valid)

    for layer, selected in ((0, main), (5, mtp)):
        by_stream = _read(recorder, _ROUTER_STREAM_TOKENS.name, layer).view(_EXPERTS, _STREAMS)
        aggregate = _read(recorder, _ROUTER_TOKENS.name, layer)
        assert torch.equal(by_stream.sum(dim=1), aggregate)
        expected = torch.zeros((_EXPERTS, _STREAMS), dtype=torch.int32)
        expected[:, 0].scatter_add_(0, selected[:2].flatten(), torch.ones(4, dtype=torch.int32))
        expected[:, 1].scatter_add_(0, selected[2:].flatten(), torch.ones(4, dtype=torch.int32))
        assert torch.equal(by_stream, expected)
        assert int(by_stream.sum()) == selected.shape[0] * _TOP_K
        assert _read(recorder, _EXPERT_COUNT.name, layer).sum() == 0


def test_routing_rejects_token_count_not_divisible_by_stream_samples(shm_rings) -> None:
    requests = [(_ENTROPY_HIST, 0, 0, 32), (_ROUTER_TOKENS, 0, 0, 2),
                (_ROUTER_STREAM_TOKENS, 0, 0, 4)]
    recorder = _make_recorder(shm_rings, requests)
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=2)
    stats.begin_microbatch(0, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="must be divisible"):
        stats.observe_routing(0, torch.zeros((3, 1), dtype=torch.int64), torch.ones((3, 2)))


def test_begin_microbatch_validates_and_compacts_stream_ids(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, [])
    stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=2)
    stats.begin_microbatch(0, torch.tensor([0, 9, 1, 9], dtype=torch.int16)[::2])
    assert stats._stream_ids.dtype is torch.int32
    assert stats._stream_ids.is_contiguous()
    assert stats._stream_ids.tolist() == [0, 1]

    with pytest.raises(RuntimeError, match="outside the configured stream range"):
        stats.begin_microbatch(0, torch.tensor([0, 2], dtype=torch.int32))
    with pytest.raises(ValueError, match="one-dimensional"):
        stats.begin_microbatch(0, torch.tensor([[0, 1]], dtype=torch.int32))
    with pytest.raises(ValueError, match="integer dtype"):
        stats.begin_microbatch(0, torch.tensor([0.0, 1.0]))


def test_begin_routing_rejects_missing_stream_ids_for_multiple_streams(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, [])
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=2)

    with pytest.raises(ValueError, match="stream_ids is None but streams != 1"):
        stats.begin_routing(0, tokens=2, device=torch.device("cpu"))


def test_begin_routing_rejects_stream_ids_on_a_different_device(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, [])
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=2)
    stats.begin_microbatch(0, torch.tensor([0, 1]))

    with pytest.raises(ValueError, match="same device"):
        stats.begin_routing(0, tokens=2, device=torch.device("meta"))


def test_without_token_loss_only_the_expert_loss_statistics_are_recorded(shm_rings) -> None:
    expert_requests = [
        (metric, layer, 0, 3) for metric in (_EXPERT_SUM, _EXPERT_SUMSQ, _EXPERT_COUNT) for layer in (0, 5)
    ]
    expert_requests += [(_ROUTER_TOKENS, layer, 0, 3) for layer in (0, 5)] + [
        (_ROUTER_STREAM_TOKENS, layer, 0, 3 * _STREAMS) for layer in (0, 5)
    ] + [
        (_ENTROPY_HIST, layer, 0, 32) for layer in (0, 5)
    ]
    recorder = _make_recorder(shm_rings, expert_requests)
    recorder.begin_step(False)
    stats = TokenLossStats(
        recorder, _ROUTED_LAYERS, _MTP_LAYERS, experts=_EXPERTS, top_k=_TOP_K, streams=_STREAMS, token_loss=False
    )
    stats.begin_step(False)
    stats.begin_microbatch(0, torch.tensor([0, 1]))
    stats.observe_routing(0, torch.tensor([[0, 1], [1, 2], [0, 2], [1, 1]]), torch.rand(4, _EXPERTS))
    stats.observe_routing(5, torch.tensor([[2, 0], [1, 2], [0, 1], [2, 2]]), torch.rand(4, _EXPERTS))

    stats.observe_loss(torch.tensor([[[0.1, 0.3], [float("nan"), 0.5]], [[0.2, float("inf")], [0.05, 0.4]]]),
                       torch.tensor([[[True, True], [True, False]], [[True, True], [True, True]]]))

    assert _read(recorder, "expert_loss/sum", 0, 0).tolist() == pytest.approx([0.1, 0.4, 0.3])
    assert _read(recorder, "expert_loss/count", 0, 0).tolist() == [1, 2, 1]
    assert _read(recorder, "expert_loss/count", 5, 0).tolist() == [2, 1, 3]


def test_observe_routing_builds_entropy_hist_over_every_token(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, [(_ENTROPY_HIST, 0, 0, 32), (_ROUTER_TOKENS, 0, 0, 2),
                                          (_ROUTER_STREAM_TOKENS, 0, 0, 2)])
    recorder.begin_step(False)
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=1)
    stats.begin_step(False)
    stats.begin_microbatch(0, None)

    scores = torch.tensor([[1.0, 1.0]] + [[1000.0, 1e-3]] * 7 + [[3.0, 1.0]])
    stats.observe_routing(0, torch.zeros(9, 1, dtype=torch.int64), scores)

    entropy_0 = 1.0
    normalized_8 = torch.tensor([0.75, 0.25])
    entropy_8 = float(-(normalized_8 * normalized_8.log()).sum() / math.log(2))
    expected_entropy_hist = torch.zeros(32, dtype=torch.int32)
    expected_entropy_hist[0] = 7
    expected_entropy_hist[min(int(entropy_0 / (1 / 32)), 31)] += 1
    expected_entropy_hist[int(entropy_8 / (1 / 32))] += 1
    assert ENTROPY_TOKEN_STRIDE == 1
    assert _read(recorder, "router/entropy_hist", 0, 0).tolist() == expected_entropy_hist.tolist()


@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
def test_entropy_bins_match_decimal_reference_at_all_boundaries(shm_rings, scale) -> None:
    probabilities = []
    for boundary in range(1, 32):
        low, high = 0.0, 0.5
        for _ in range(64):
            middle = (low + high) / 2
            entropy = -(middle * math.log(middle) + (1 - middle) * math.log1p(-middle)) / math.log(2)
            if entropy < boundary / 32:
                low = middle
            else:
                high = middle
        probabilities.append(middle)
    center = torch.tensor(probabilities, dtype=torch.float32)
    neighbors = torch.stack((torch.nextafter(center, torch.zeros_like(center)), center,
                             torch.nextafter(center, torch.ones_like(center)))).flatten()
    scores = torch.stack((neighbors, 1 - neighbors), dim=1) * scale
    scores = torch.cat((scores, torch.tensor([[1., 1.], [0., 1.], [1., 0.]]) * scale))
    expected = [0] * 32
    with localcontext() as context:
        context.prec = 80
        for row in scores.tolist():
            weights = [Decimal.from_float(value) for value in row]
            total = sum(weights)
            terms = [value / total for value in weights if value > 0]
            entropy = -sum(value * value.ln() for value in terms) / Decimal(2).ln()
            expected[min(int(entropy * 32), 31)] += 1

    recorder = _make_recorder(shm_rings, [(_ENTROPY_HIST, 0, 0, 32), (_ROUTER_TOKENS, 0, 0, 2),
                                          (_ROUTER_STREAM_TOKENS, 0, 0, 2)])
    recorder.begin_step(False)
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=1)
    stats.begin_step(False)
    stats.begin_microbatch(0, None)
    stats.observe_routing(0, scores.argmax(dim=1, keepdim=True), scores)
    assert _read(recorder, "router/entropy_hist", 0).tolist() == expected
    assert sum(expected) == scores.shape[0] == 96


def test_stream_stats_retain_separate_microbatches_within_a_step(shm_rings) -> None:
    requests = [(metric, layer, microbatch, width) for microbatch in range(2)
                for metric, layer, _, width in _single_stream_requests()]
    recorder = _make_recorder(shm_rings, requests)
    recorder.begin_step(False)
    stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=1)
    stats.begin_step(False)

    stats.begin_microbatch(0, None)
    stats.observe_loss(torch.tensor([[[0.1, 0.2]]]), torch.ones(1, 1, 2, dtype=torch.bool))
    stats.begin_microbatch(1, None)
    stats.observe_loss(torch.tensor([[[0.3, 0.4]]]), torch.ones(1, 1, 2, dtype=torch.bool))

    for microbatch, total, sum_of_squares in ((0, 0.3, 0.05), (1, 0.7, 0.25)):
        assert _read(recorder, "loss/stream_sum", -1, microbatch).tolist() == pytest.approx([total])
        assert _read(recorder, "loss/stream_sumsq", -1, microbatch).tolist() == pytest.approx([sum_of_squares])
        assert _read(recorder, "loss/stream_count", -1, microbatch).tolist() == [2]
        assert _read(recorder, "loss/stream_samples", -1, microbatch).tolist() == [1]


def test_observe_loss_with_no_stream_ids_uses_a_single_stream(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, _single_stream_requests())
    recorder.begin_step(False)
    stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=1)
    stats.begin_step(False)
    stats.begin_microbatch(0, None)

    stats.observe_loss(torch.tensor([[[0.2, 0.4]]]), torch.ones(1, 1, 2, dtype=torch.bool))

    assert _read(recorder, "loss/stream_sum", -1, 0).tolist() == pytest.approx([0.6])
    assert _read(recorder, "loss/stream_count", -1, 0).tolist() == [2]
    assert _read(recorder, "loss/stream_samples", -1, 0).tolist() == [1]


def test_observe_loss_with_no_valid_tokens_reports_zero_token_max(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, _single_stream_requests())
    recorder.begin_step(False)
    stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=1)
    stats.begin_step(False)
    stats.begin_microbatch(0, None)

    stats.observe_loss(torch.tensor([[[0.2, 0.4]]]), torch.zeros(1, 1, 2, dtype=torch.bool))

    assert _read(recorder, "loss/token_max", -1, 0).item() == 0.0
    assert _read(recorder, "loss/nonfinite_tokens", -1, 0).item() == 0
    assert not _read(recorder, "loss/position_sum", -1, 0).any()
    assert not _read(recorder, "loss/position_count", -1, 0).any()


def _detailed_requests() -> list[tuple]:
    return [
        (_NONFINITE, -1, 0, 1),
        (_TOKEN_MAX, -1, 0, 1),
        (_TOKEN_HIST, -1, 0, 2 * 256),
        (_STREAM_SUM, -1, 0, 2),
        (_STREAM_SUMSQ, -1, 0, 2),
        (_STREAM_COUNT, -1, 0, 2),
        (_STREAM_SAMPLES, -1, 0, 2),
        (_POSITION_SUM, -1, 0, LOSS_POSITION_BUCKETS),
        (_POSITION_COUNT, -1, 0, LOSS_POSITION_BUCKETS),
        (_EXPERT_SUM, 0, 0, 2),
        (_EXPERT_SUMSQ, 0, 0, 2),
        (_EXPERT_COUNT, 0, 0, 2),
        (_ENTROPY_HIST, 0, 0, 32),
        (_ROUTER_TOKENS, 0, 0, 2),
        (_ROUTER_STREAM_TOKENS, 0, 0, 4),
        (_EXPERT_HIST, 0, 0, 2 * 64),
        (_EXPERT_STREAM_COUNT, 0, 0, 2 * 2),
        (_EXPERT_STREAM_SUM, 0, 0, 2 * 2),
        (_SAMPLE_HIST, -1, 0, 2 * SAMPLE_LOSS_BINS.count),
    ]


def _detailed_stats(shm_rings) -> tuple[RankRecorder, TokenLossStats]:
    recorder = _make_recorder(shm_rings, _detailed_requests())
    recorder.begin_step(True)
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=2)
    stats.begin_step(True)
    stats.begin_microbatch(0, torch.tensor([0, 1]))
    return recorder, stats


def test_detailed_mode_builds_entropy_and_expert_loss_histograms(shm_rings) -> None:
    recorder, stats = _detailed_stats(shm_rings)

    scores = torch.tensor([[1.0, 1.0], [3.0, 1.0]])
    topk_indices = torch.tensor([[1], [0]], dtype=torch.int64)
    stats.observe_routing(0, topk_indices, scores)

    loss = torch.tensor([[[0.3], [0.6]]])
    labels_valid = torch.ones(1, 2, 1, dtype=torch.bool)
    stats.observe_loss(loss, labels_valid)

    assert _read(recorder, "expert_loss/sum", 0, 0).tolist() == pytest.approx([0.6, 0.3])
    assert _read(recorder, "expert_loss/count", 0, 0).tolist() == [1, 1]

    expected_hist = torch.zeros(2 * 64, dtype=torch.int32)
    expected_hist[2] = 1
    expected_hist[64 + 1] = 1
    assert _read(recorder, "expert_loss/hist", 0, 0).tolist() == expected_hist.tolist()

    assert _read(recorder, "expert_loss/stream_count", 0, 0).tolist() == [0, 1, 1, 0]
    assert _read(recorder, "expert_loss/stream_sum", 0, 0).tolist() == pytest.approx([0.0, 0.6, 0.3, 0.0])

    expected_entropy_hist = torch.zeros(32, dtype=torch.int32)
    expected_entropy_hist[25] = 1
    expected_entropy_hist[31] = 1
    assert _read(recorder, "router/entropy_hist", 0, 0).tolist() == expected_entropy_hist.tolist()


def test_detailed_mode_builds_position_stats_and_sample_hist_by_stream(shm_rings) -> None:
    recorder, stats = _detailed_stats(shm_rings)
    stats.observe_routing(0, torch.tensor([[1], [0]], dtype=torch.int64), torch.tensor([[1.0, 1.0], [3.0, 1.0]]))

    # sequence length 1, so both samples' single token falls in position bucket 0; sample 0 is on
    # stream 0 with loss 0.3 (bin floor(0.3/0.125)=2), sample 1 on stream 1 with loss 0.6 (bin 4).
    loss = torch.tensor([[[0.3], [0.6]]])
    labels_valid = torch.ones(1, 2, 1, dtype=torch.bool)
    stats.observe_loss(loss, labels_valid)

    expected_position_sum = [0.0] * LOSS_POSITION_BUCKETS
    expected_position_sum[0] = 0.9
    assert _read(recorder, "loss/position_sum", -1, 0).tolist() == pytest.approx(expected_position_sum)

    expected_position_count = [0] * LOSS_POSITION_BUCKETS
    expected_position_count[0] = 2
    assert _read(recorder, "loss/position_count", -1, 0).tolist() == expected_position_count

    expected_sample_hist = [0] * (2 * SAMPLE_LOSS_BINS.count)
    expected_sample_hist[2] = 1
    expected_sample_hist[SAMPLE_LOSS_BINS.count + 4] = 1
    assert _read(recorder, "loss/sample_hist_by_stream", -1, 0).tolist() == expected_sample_hist


def test_sample_hist_by_stream_skips_a_sample_with_no_valid_tokens(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, _detailed_requests())
    recorder.begin_step(True)
    stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=2)
    stats.begin_step(True)
    stats.begin_microbatch(0, torch.tensor([0, 1, 0]))

    # Sample 0 (stream 0): both tokens valid, mean 0.4 -> bin floor(0.4/0.125)=3.
    # Sample 1 (stream 1): no valid label at all, so it must not appear anywhere in the histogram.
    # Sample 2 (stream 0): one NaN token excluded, mean is the other token alone, 0.9 -> bin 7.
    loss = torch.tensor([[[0.2, 0.6], [0.7, 0.7], [float("nan"), 0.9]]])
    labels_valid = torch.tensor([[[True, True], [False, False], [True, True]]])
    stats.observe_loss(loss, labels_valid)

    expected_sample_hist = [0] * (2 * SAMPLE_LOSS_BINS.count)
    expected_sample_hist[3] = 1
    expected_sample_hist[7] = 1
    assert _read(recorder, "loss/sample_hist_by_stream", -1, 0).tolist() == expected_sample_hist


def test_sample_hist_by_stream_is_skipped_outside_a_detailed_step(shm_rings) -> None:
    requests = [*_single_stream_requests(), (_SAMPLE_HIST, -1, 0, SAMPLE_LOSS_BINS.count)]
    recorder = _make_recorder(shm_rings, requests)
    recorder.begin_step(False)
    stats = TokenLossStats(recorder, {}, [], experts=2, top_k=1, streams=1)
    stats.begin_step(False)
    stats.begin_microbatch(0, None)

    stats.observe_loss(torch.tensor([[[0.2, 0.4]]]), torch.ones(1, 1, 2, dtype=torch.bool))

    assert not _read(recorder, "loss/sample_hist_by_stream", -1, 0).any()


def test_statistics_save_no_tensors_for_backward_even_when_inputs_require_grad(shm_rings) -> None:
    # Under activation checkpointing the recomputed forward skips the observers, so anything they
    # saved for backward in the original forward breaks the checkpoint's saved-tensor count.
    _, stats = _detailed_stats(shm_rings)
    scores = torch.tensor([[0.0, 0.0], [1.0, -1.0]], requires_grad=True).sigmoid()
    loss = torch.tensor([[[0.3], [0.6]]], requires_grad=True) * 1.0
    saved = []

    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        stats.observe_routing(0, torch.tensor([[1], [0]], dtype=torch.int64), scores)
        stats.observe_loss(loss, torch.ones(1, 2, 1, dtype=torch.bool))

    assert saved == []


def test_begin_microbatch_requires_streams_one_when_stream_ids_is_none(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, [(_NONFINITE, -1, 0, 1)])
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=2)

    with pytest.raises(ValueError):
        stats.begin_microbatch(0, None)


def test_begin_microbatch_allows_stream_ids_none_when_streams_is_one(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, [(_NONFINITE, -1, 0, 1)])
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=1)

    stats.begin_microbatch(0, None)


def test_observe_routing_rejects_inconsistent_token_counts_within_one_microbatch(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, [(_EXPERT_SUM, 0, 0, 2), (_ENTROPY_HIST, 0, 0, 32),
                                          (_ROUTER_TOKENS, 0, 0, 2), (_ROUTER_STREAM_TOKENS, 0, 0, 2)])
    recorder.begin_step(False)
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=1)
    stats.begin_step(False)
    stats.begin_microbatch(0, None)
    stats.observe_routing(0, torch.zeros(4, 1, dtype=torch.int64), torch.ones(4, 2))

    with pytest.raises(ValueError):
        stats.observe_routing(0, torch.zeros(3, 1, dtype=torch.int64), torch.ones(3, 2))


def test_observe_loss_raises_value_error_when_a_routed_layer_was_not_observed(shm_rings) -> None:
    recorder = _make_recorder(shm_rings, _single_stream_requests())
    recorder.begin_step(False)
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=1)
    stats.begin_step(False)
    stats.begin_microbatch(0, None)

    with pytest.raises(ValueError):
        stats.observe_loss(torch.zeros(1, 1, 2), torch.ones(1, 1, 2, dtype=torch.bool))


def test_expert_statistics_survive_per_rank_per_microbatch_without_reduction(shm_rings, tmp_path) -> None:
    requests = [(metric, layer, microbatch, width) for microbatch in range(2)
                for metric, layer, _, width in _detailed_requests()]
    requests.sort(key=lambda request: request[0].detailed)
    layout = Layout(requests)
    rings = [shm_rings(layout, slot_count=1) for _ in range(2)]
    for rank, ring in enumerate(rings):
        recorder = RankRecorder(layout, ring, torch.device("cpu"))
        recorder.begin_step(True)
        stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=2)
        stats.begin_step(True)
        for microbatch in range(2):
            stats.begin_microbatch(microbatch, torch.tensor([0, 1]))
            indices = torch.tensor([[microbatch], [1 - microbatch]])
            stats.observe_routing(0, indices, torch.ones(2, 2))
            loss = torch.tensor([[[float(1 + 4 * microbatch + 10 * rank)], [float(3 + 4 * microbatch + 10 * rank)]]])
            stats.observe_loss(loss, torch.ones(1, 2, 1, dtype=torch.bool))
        assert recorder.step_end(1)
    aggregator = NodeAggregator(
        [RankRing(rank, ring) for rank, ring in enumerate(rings)], tmp_path, "node", "attempt",
        window_steps=1, rank_timeout_s=5.0, nodes_in_job=1, primary_node=True, clock=lambda: 0.0,
    )
    rows = [row for path in aggregator.poll_once() if path.suffix == ".parquet"
            for row in pq.read_table(path, partitioning=None).to_pylist()]
    values = {(row["rank"], row["microbatch"], row["metric_id"]): row["values"] for row in rows}
    for rank in range(2):
        for microbatch in range(2):
            loss = [float(1 + 4 * microbatch + 10 * rank), float(3 + 4 * microbatch + 10 * rank)]
            expert_loss = loss if microbatch == 0 else loss[::-1]
            assert values[rank, microbatch, _ROUTER_TOKENS.metric_id] == [1, 1]
            assert values[rank, microbatch, _ROUTER_STREAM_TOKENS.metric_id] == ([1, 0, 0, 1] if microbatch == 0
                                                                                 else [0, 1, 1, 0])
            assert values[rank, microbatch, _EXPERT_SUM.metric_id] == expert_loss
            assert values[rank, microbatch, _EXPERT_SUMSQ.metric_id] == [x * x for x in expert_loss]
            assert values[rank, microbatch, _EXPERT_COUNT.metric_id] == [1, 1]
            assert values[rank, microbatch, _STREAM_SUM.metric_id] == loss
            assert values[rank, microbatch, _STREAM_COUNT.metric_id] == [1, 1]
            histogram = [0] * 128
            for expert, value in enumerate(expert_loss):
                histogram[expert * 64 + min(int(value / 0.25), 63)] = 1
            assert values[rank, microbatch, _EXPERT_HIST.metric_id] == histogram


def test_routing_buffer_handles_partial_microbatches_and_later_growth(shm_rings) -> None:
    requests = [(metric, layer, microbatch, width) for microbatch in range(2)
                for metric, layer, _, width in _detailed_requests()]
    requests.sort(key=lambda request: request[0].detailed)
    recorder = _make_recorder(shm_rings, requests)
    stats = TokenLossStats(recorder, {0: 0}, [], experts=2, top_k=1, streams=2)
    for batch_sizes in ((3, 1), (5, 2)):
        recorder.begin_step(False)
        stats.begin_step(False)
        for microbatch, batch_size in enumerate(batch_sizes):
            stats.begin_microbatch(microbatch, torch.zeros(batch_size, dtype=torch.int32))
            stats.observe_routing(0, torch.full((batch_size, 1), microbatch), torch.ones(batch_size, 2))
            loss = torch.full((1, batch_size, 1), float(2 + microbatch))
            stats.observe_loss(loss, torch.ones_like(loss, dtype=torch.bool))
        for microbatch, batch_size in enumerate(batch_sizes):
            counts = [0, 0]
            counts[microbatch] = batch_size
            expected_loss = [0.0, 0.0]
            expected_loss[microbatch] = batch_size * float(2 + microbatch)
            assert _read(recorder, _ROUTER_TOKENS.name, 0, microbatch).tolist() == counts
            assert _read(recorder, _ROUTER_STREAM_TOKENS.name, 0, microbatch).tolist() == [counts[0], 0,
                                                                                         counts[1], 0]
            assert _read(recorder, _EXPERT_COUNT.name, 0, microbatch).tolist() == counts
            assert _read(recorder, _EXPERT_SUM.name, 0, microbatch).tolist() == expected_loss
            assert _read(recorder, _STREAM_COUNT.name, -1, microbatch).tolist() == [batch_size, 0]
