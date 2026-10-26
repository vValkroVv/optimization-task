from __future__ import annotations

import uuid

import numpy as np
import pytest
import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import RankRecorder
from metric_sampler.capture.ring import ShmRing
from metric_sampler.capture.stats.attention import AttentionStats, linear_slots, linear_step_slots, softmax_slots
from metric_sampler.contract.catalog import metric_by_name

pytestmark = pytest.mark.unit

HEADS, NOPE, ROPE, VALUE = 3, 5, 2, 4
LSE, QKV, SOFTMAX_GATE = "attn/softmax/lse_stats", "attn/softmax/qkv_stats", "attn/softmax/gate_stats"
LINEAR_GATE, LINEAR_NORM, DECAY = "attn/linear/gate_stats", "attn/linear/norm_stats", "attn/linear/decay_params"
ROW_STATS, ENTROPY_HIST, HEAD_ENTROPY_HIST = (
    "attn/softmax/row_stats", "attn/softmax/entropy_hist", "attn/softmax/head_entropy_hist"
)
QUANTILES, QUANTILE_COUNTS = "attn/softmax/entropy_quantiles", "attn/softmax/entropy_quantile_counts"
STATE = "attn/linear/state_stats"
ROW_STAT_NAMES = (
    "ent_sum", "ent_sumsq", "ent_min", "ent_max", "entn_sum", "entn_min", "entn_max", "pmax_sum",
    "pmax_min", "pmax_max",
    "pmin_sum", "lpmin_sum", "lpmin_min", "lpmin_max", "mlogit_sum", "mlogit_max", "coll_sum", "sink_sum", "sink_max",
    "dist_sum", "local_sum",
)
WIDTHS = {
    LSE: 1 + 3 * HEADS, QKV: 1 + 8 * HEADS, SOFTMAX_GATE: 1 + 3 * HEADS,
    LINEAR_GATE: 1 + 7 * HEADS, LINEAR_NORM: 1 + 3 * HEADS,
}
STAGE_TWO_WIDTHS = {
    ROW_STATS: 1 + len(ROW_STAT_NAMES) * HEADS, ENTROPY_HIST: 32, QUANTILES: 9, QUANTILE_COUNTS: 5,
    STATE: 1 + 2 * HEADS,
}
HEAD_HIST_WIDTH = 32 * HEADS
A_LOG = torch.tensor([0.0, 1.0, -2.5])
DT_BIAS = torch.tensor([-3.0, 0.5, 25.0])


@pytest.fixture
def started_step():
    requests = [(metric_by_name(name), layer, microbatch, width)
                for layer in (0, 1) for microbatch in (0, 1) for name, width in (WIDTHS | STAGE_TWO_WIDTHS).items()]
    requests += [(metric_by_name(DECAY), layer, -1, 2 * HEADS) for layer in (0, 1)]
    requests += [(metric_by_name(HEAD_ENTROPY_HIST), layer, microbatch, HEAD_HIST_WIDTH)
                 for layer in (0, 1) for microbatch in (0, 1)]
    layout = Layout(requests)
    ring = ShmRing.create('attention-cpu-' + uuid.uuid4().hex, layout, 1)
    recorder = RankRecorder(layout, ring, torch.device('cpu'))
    recorder.begin_step(False)
    yield recorder, layout, ring
    recorder.close()
    ring.unlink()


@pytest.fixture
def capture(started_step):
    recorder, layout, ring = started_step
    delivered = []

    def read(name, layer=0, microbatch=0):
        # What the step delivers: the device buffer with host-staged words overlaid.
        if not delivered:
            recorder.step_end(step=1)
            delivered.append(torch.from_numpy(ring.payload(ring.ready_slots()[0]).copy()))
        slot = layout.slot(name, layer, microbatch)
        return delivered[0][slot.offset:slot.offset + slot.width].clone().view(torch.float32)

    return AttentionStats(recorder), read


@pytest.fixture
def recorded(started_step, capture):
    """Whether the step delivered a slot as recorded."""
    _, layout, ring = started_step
    _, read = capture

    def is_recorded(name, layer=0, microbatch=0) -> bool:
        read(name, layer, microbatch)
        payload = ring.payload(ring.ready_slots()[0])
        mask = payload[layout.words:layout.words + ring.mask_words].view(np.uint8)
        return bool(np.unpackbits(mask, bitorder="little")[layout.slot_index(name, layer, microbatch)])

    return is_recorded


def test_slots_are_the_vectors_the_statistics_write_in_layout_order():
    widths = WIDTHS | STAGE_TWO_WIDTHS | {HEAD_ENTROPY_HIST: HEAD_HIST_WIDTH, DECAY: 2 * HEADS}
    softmax_names = [LSE, QKV, SOFTMAX_GATE, ROW_STATS, ENTROPY_HIST, HEAD_ENTROPY_HIST, QUANTILES, QUANTILE_COUNTS]

    assert softmax_slots(HEADS, gated=True) == tuple((name, widths[name]) for name in softmax_names)
    softmax_names.remove(SOFTMAX_GATE)
    assert softmax_slots(HEADS, gated=False) == tuple((name, widths[name]) for name in softmax_names)
    assert linear_slots(HEADS) == tuple((name, widths[name]) for name in (LINEAR_GATE, LINEAR_NORM, STATE))
    assert linear_step_slots(HEADS) == ((DECAY, widths[DECAY]),)


def _rows(count: int) -> torch.Tensor:
    return torch.tensor([float(count)], dtype=torch.float64)


def _softmax_data(rows: int, dtype: torch.dtype, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    query, key = (torch.randn(rows, HEADS, NOPE + ROPE, generator=generator).to(dtype) for _ in range(2))
    value = torch.randn(rows, HEADS, VALUE, generator=generator).to(dtype)
    lse = torch.randn(HEADS, rows, generator=generator) * 2 + 8
    gate = torch.randn(rows, HEADS, VALUE, generator=generator).to(dtype)
    return query, key, value, lse, gate


def _linear_data(rows: int, dtype: torch.dtype, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    beta = torch.rand(rows, HEADS, generator=generator).to(dtype)
    log_decay = (-torch.rand(rows, HEADS, generator=generator) * 3).to(dtype)
    value, output = (torch.randn(rows, HEADS, VALUE, generator=generator).to(dtype) for _ in range(2))
    return beta, log_decay, value, output


def _softmax_expected(query, key, value, lse, gate) -> dict[str, torch.Tensor]:
    query, key, value, lse, gate = (tensor.double() for tensor in (query, key, value, lse, gate))

    def norm_sum(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(dim=-1).sum(0)

    def peak(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.abs().amax(dim=(0, 2))

    gate = gate.mean(-1)
    return {
        LSE: torch.cat((_rows(query.shape[0]), lse.sum(1), lse.amin(1), lse.amax(1))),
        QKV: torch.cat((
            _rows(query.shape[0]), norm_sum(query[..., :NOPE]), norm_sum(query[..., NOPE:]),
            norm_sum(key[..., :NOPE]), norm_sum(key[..., NOPE:]), norm_sum(value),
            peak(query), peak(key), peak(value),
        )),
        SOFTMAX_GATE: torch.cat((_rows(query.shape[0]), gate.sum(0), gate.amin(0), gate.amax(0))),
    }


def _linear_expected(beta, log_decay, value, output) -> dict[str, torch.Tensor]:
    beta, log_decay, value, output = (tensor.double() for tensor in (beta, log_decay, value, output))
    alpha = log_decay.exp()
    return {
        LINEAR_GATE: torch.cat((
            _rows(beta.shape[0]), beta.sum(0), beta.amin(0), beta.amax(0),
            alpha.sum(0), alpha.amin(0), alpha.amax(0), log_decay.sum(0),
        )),
        LINEAR_NORM: torch.cat((
            _rows(beta.shape[0]), value.norm(dim=-1).sum(0), output.norm(dim=-1).sum(0), output.abs().amax(dim=(0, 2)),
        )),
    }


def _decay_expected() -> torch.Tensor:
    return torch.cat((A_LOG.double().exp(), torch.nn.functional.softplus(DT_BIAS.double())))


def _flash_layout(query, key, value, lse, gate, batch: int):
    """The tensors as the flash-attention call sees a [batch, sequence_length] input: rows split over two axes and
    the log-partition [batch, heads, sequence_length] viewed head-first."""
    sequence_length = query.shape[0] // batch

    def split(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, sequence_length, *tensor.shape[1:])

    flash_lse = lse.reshape(HEADS, batch, sequence_length).transpose(0, 1).contiguous()
    return split(query), split(key), split(value), flash_lse.transpose(0, 1), split(gate)


def _observe_softmax(stats, data, layer=0, microbatch=0, batch=None):
    query, key, value, lse, gate = data if batch is None else _flash_layout(*data, batch)
    stats.observe_softmax(layer, microbatch, query, key, value, lse, NOPE)
    stats.observe_softmax_gate(layer, microbatch, gate)


def _observe_linear(stats, data, layer=0, microbatch=0):
    stats.observe_linear(layer, microbatch, *data, A_LOG, DT_BIAS)


def _assert_recorded(read, expected: dict[str, torch.Tensor], layer=0, microbatch=0) -> None:
    for name, values in expected.items():
        assert values.numel() == (WIDTHS | STAGE_TWO_WIDTHS)[name]
        torch.testing.assert_close(read(name, layer, microbatch), values.float(), rtol=2e-6, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [None, 1, 3])
def test_softmax_statistics_equal_the_per_head_reference_for_flat_and_batched_rows(capture, dtype, batch):
    stats, read = capture
    data = _softmax_data(12, dtype)

    _observe_softmax(stats, data, batch=batch)

    _assert_recorded(read, _softmax_expected(*data))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_statistics_equal_the_per_head_reference(capture, dtype):
    stats, read = capture
    data = _linear_data(12, dtype)

    _observe_linear(stats, data)

    _assert_recorded(read, _linear_expected(*data))


def test_linear_statistics_of_batched_rows_equal_those_of_flat_rows(capture):
    stats, read = capture
    beta, log_decay, value, output = _linear_data(12, torch.float32)

    stats.observe_linear(0, 0, beta.reshape(3, 4, HEADS), log_decay.reshape(3, 4, HEADS),
                         value.reshape(3, 4, HEADS, VALUE), output.reshape(3, 4, HEADS, VALUE), A_LOG, DT_BIAS)

    _assert_recorded(read, _linear_expected(beta, log_decay, value, output))


def test_a_single_row_is_its_own_sum_minimum_and_maximum(capture):
    stats, read = capture
    softmax, linear = _softmax_data(1, torch.float32), _linear_data(1, torch.float32)

    _observe_softmax(stats, softmax)
    _observe_linear(stats, linear)

    lse = read(LSE)
    assert lse[0].item() == 1
    assert lse[1:1 + HEADS].tolist() == lse[1 + HEADS:1 + 2 * HEADS].tolist() == lse[1 + 2 * HEADS:].tolist() \
        == softmax[3][:, 0].tolist()
    _assert_recorded(read, {**_softmax_expected(*softmax), **_linear_expected(*linear)})


def test_repeated_observations_of_one_layer_and_microbatch_form_one_population(capture):
    stats, read = capture
    first, second = _softmax_data(6, torch.float32, seed=1), _softmax_data(10, torch.bfloat16, seed=2)
    linear_first, linear_second = _linear_data(6, torch.float32, seed=1), _linear_data(10, torch.bfloat16, seed=2)

    for data, linear in ((first, linear_first), (second, linear_second)):
        _observe_softmax(stats, data)
        _observe_linear(stats, linear)

    both = tuple(torch.cat((a, b), dim=1 if index == 3 else 0) for index, (a, b) in enumerate(zip(first, second)))
    both_linear = tuple(torch.cat((a.float(), b.float())) for a, b in zip(linear_first, linear_second))
    _assert_recorded(read, {**_softmax_expected(*both), **_linear_expected(*both_linear)})
    assert read(LSE)[0].item() == read(LINEAR_GATE)[0].item() == 16


def test_layers_and_microbatches_are_recorded_separately(capture):
    stats, read = capture
    data = {(layer, microbatch): _softmax_data(4 + layer + 2 * microbatch, torch.float32, seed=10 * layer + microbatch)
            for layer, microbatch in [(0, 0), (1, 0), (1, 1)]}

    for (layer, microbatch), observed in data.items():
        _observe_softmax(stats, observed, layer, microbatch)

    for (layer, microbatch), observed in data.items():
        _assert_recorded(read, _softmax_expected(*observed), layer, microbatch)


def test_a_new_step_starts_its_slots_over_instead_of_merging_with_the_last_one(started_step, capture):
    recorder, _, _ = started_step
    stats, read = capture
    _observe_softmax(stats, _softmax_data(5, torch.float32, seed=3))
    _observe_linear(stats, _linear_data(5, torch.float32, seed=3))
    recorder.begin_step(False)
    stats.begin_step(False)
    data, linear = _softmax_data(7, torch.float32, seed=4), _linear_data(7, torch.float32, seed=4)

    _observe_softmax(stats, data)
    _observe_linear(stats, linear)

    _assert_recorded(read, {**_softmax_expected(*data), **_linear_expected(*linear)})


def test_decay_parameters_are_recorded_once_per_layer_with_the_first_microbatch(capture):
    stats, read = capture
    data = _linear_data(4, torch.float32)

    _observe_linear(stats, data, layer=1, microbatch=0)
    _observe_linear(stats, data, layer=1, microbatch=1)
    _observe_linear(stats, data, layer=0, microbatch=1)

    torch.testing.assert_close(read(DECAY, 1, -1), _decay_expected().float(), rtol=1e-6, atol=0)
    assert not read(DECAY, 0, -1).any()


def test_a_log_partition_of_other_rows_than_the_query_is_rejected(capture):
    stats, _ = capture
    query, key, value, lse, _gate = _softmax_data(6, torch.float32)

    with pytest.raises(ValueError, match="one value per query row and head"):
        stats.observe_softmax(0, 0, query, key, value, lse[:, :5], NOPE)


def test_nonfinite_values_make_only_the_statistics_of_their_head_nonfinite(capture):
    stats, read = capture
    softmax = _softmax_data(6, torch.float32)
    query, _, value, lse, gate = softmax
    query[2, 1, NOPE] = float("nan")
    value[0, 2, 0] = float("inf")
    lse[0, 3] = float("-inf")
    gate[4, 1, 1] = float("nan")
    linear = _linear_data(6, torch.float32)
    beta, log_decay, _, output = linear
    log_decay[1, 0] = float("-inf")
    beta[5, 2] = float("nan")
    output[3, 1, 2] = float("inf")

    _observe_softmax(stats, softmax)
    _observe_linear(stats, linear)

    _assert_recorded(read, {**_softmax_expected(*softmax), **_linear_expected(*linear)})
    nonfinite = {name: (~torch.isfinite(read(name)[1:])).reshape(-1, HEADS).any(dim=0).tolist() for name in WIDTHS}
    assert nonfinite == {
        LSE: [True, False, False], QKV: [False, True, True], SOFTMAX_GATE: [False, True, False],
        LINEAR_GATE: [True, False, True], LINEAR_NORM: [False, True, False],
    }


def test_observing_tensors_that_require_grad_builds_no_graph_and_leaves_them_untouched(capture):
    stats, read = capture
    data = _softmax_data(6, torch.float32)
    linear = _linear_data(6, torch.float32)
    for tensor in (*data, *linear):
        tensor.requires_grad_()

    _observe_softmax(stats, data)
    _observe_linear(stats, linear)

    assert all(tensor.grad is None and tensor.is_leaf for tensor in (*data, *linear))
    _assert_recorded(read, {**_softmax_expected(*data), **_linear_expected(*linear)})


def _stacked_words(softmax, linear) -> dict[str, torch.Tensor]:
    """The words of the implementation that stacked each call's reductions and wrote them at a flush."""
    query, key, value, lse, gate = softmax
    beta, log_decay, linear_value, output = linear

    def count(rows: int) -> torch.Tensor:
        return torch.full((1,), rows, dtype=torch.float64)

    axes = tuple(range(1, lse.ndim))
    norms = torch.stack([torch.linalg.vector_norm(tensor, dim=-1, dtype=torch.float32) for tensor in (
        query[..., :NOPE], query[..., NOPE:], key[..., :NOPE], key[..., NOPE:], value)])
    peaks = torch.stack([torch.linalg.vector_norm(tensor, ord=float("inf"), dim=-1, dtype=torch.float32)
                         for tensor in (query, key, value)])
    means = gate.mean(dim=-1, dtype=torch.float32)
    log_decay = log_decay.float()
    gates = torch.stack((beta.float(), log_decay.exp(), log_decay))
    norm_pair = torch.stack(
        [torch.linalg.vector_norm(tensor, dim=-1, dtype=torch.float32) for tensor in (linear_value, output)]
    )
    peak = torch.linalg.vector_norm(output, ord=float("inf"), dim=-1, dtype=torch.float32).amax(dim=0)
    rows = query.shape[0]
    return {
        LSE: torch.cat((count(rows), lse.sum(dim=axes, dtype=torch.float64), lse.amin(dim=axes), lse.amax(dim=axes))),
        QKV: torch.cat((count(rows), norms.sum(dim=1, dtype=torch.float64).flatten(), peaks.amax(dim=1).flatten())),
        SOFTMAX_GATE: torch.cat((
            count(rows), means.sum(dim=0, dtype=torch.float64), means.amin(dim=0), means.amax(dim=0),
        )),
        LINEAR_GATE: torch.cat((
            count(rows), gates[0].sum(0, dtype=torch.float64), gates[0].amin(0), gates[0].amax(0),
            gates[1].sum(0, dtype=torch.float64), gates[1].amin(0), gates[1].amax(0),
            gates[2].sum(0, dtype=torch.float64),
        )),
        LINEAR_NORM: torch.cat((count(rows), norm_pair.sum(dim=1, dtype=torch.float64).flatten(), peak)),
    }


@pytest.mark.parametrize("case", ["bfloat16", "float32", "nonfinite"])
def test_the_words_equal_those_of_the_stacked_implementation_bit_for_bit(capture, case):
    stats, read = capture
    dtype = torch.bfloat16 if case == "bfloat16" else torch.float32
    softmax, linear = list(_softmax_data(37, dtype, seed=5)), list(_linear_data(37, dtype, seed=5))
    if case == "nonfinite":
        softmax[0][3, 1, 4] = float("nan")
        softmax[2][9, 0, 0] = float("inf")
        softmax[3][2, 5] = float("-inf")
        softmax[4][11, 2, 3] = float("nan")
        linear[0][7, 1] = float("nan")
        linear[1][4, 2] = float("-inf")
        linear[3][1, 0, 1] = float("inf")

    _observe_softmax(stats, softmax)
    _observe_linear(stats, linear)

    for name, words in _stacked_words(softmax, linear).items():
        assert torch.equal(read(name).view(torch.int32), words.float().view(torch.int32)), name


def _row_records(seed: int, rows: int = 40):
    """Records 174-178 of one call as the kernel returns them."""
    generator = torch.Generator().manual_seed(seed)
    row_stats = torch.randn(1 + len(ROW_STAT_NAMES) * HEADS, generator=generator)
    row_stats[0] = rows
    histogram = torch.randint(0, 100, (32,), generator=generator, dtype=torch.int32)
    head_histogram = torch.randint(0, 100, (HEAD_HIST_WIDTH,), generator=generator, dtype=torch.int32)
    quantiles = torch.randn(9, generator=generator)
    counts = torch.randint(0, 1000, (5,), generator=generator, dtype=torch.int32)
    return row_stats, histogram, head_histogram, quantiles, counts


def _row_records_without_rows():
    """What the kernel returns when no row has enough keys: the identities of the merge."""
    identity = {"min": torch.inf, "max": -torch.inf}
    fields = [identity.get(name[-3:], 0.0) for name in ROW_STAT_NAMES]
    row_stats = torch.cat((torch.zeros(1), *(torch.full((HEADS,), value) for value in fields)))
    return (row_stats, torch.zeros(32, dtype=torch.int32), torch.zeros(HEAD_HIST_WIDTH, dtype=torch.int32),
            torch.full((9,), torch.nan), torch.tensor([0, 0, 0, 0, 7 * HEADS], dtype=torch.int32))


class _Produced:
    """The `compute` of `observe_softmax_rows` for records: given the words to write them into it checks the
    histograms start as zeros and writes them, given none it returns them. Keeps what it was given."""

    def __init__(self, records) -> None:
        self.records = records
        self.calls = 0
        self.given = None

    def __call__(self, targets):
        self.calls += 1
        self.given = targets
        if targets is None:
            return self.records
        for index, (target, record) in enumerate(zip(targets, self.records)):
            if index in (1, 2) and target is not None:
                assert not target.any()
            if target is not None:
                target.copy_(record)
        return targets


def _merged_row_stats(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    merged = [first[:1] + second[:1]]
    for name, mine, theirs in zip(ROW_STAT_NAMES, first[1:].reshape(-1, HEADS), second[1:].reshape(-1, HEADS)):
        if name.endswith("_min"):
            merged.append(torch.minimum(mine, theirs))
        elif name.endswith("_max"):
            merged.append(torch.maximum(mine, theirs))
        else:
            merged.append(mine + theirs)
    return torch.cat(merged)


def _words(read, name, layer=0, microbatch=0, dtype=torch.float32) -> torch.Tensor:
    return read(name, layer, microbatch).view(dtype)


def _assert_row_words(read, expected, layer=0, microbatch=0) -> None:
    row_stats, histogram, quantiles, counts = expected
    assert torch.equal(_words(read, ROW_STATS, layer, microbatch), row_stats)
    assert torch.equal(_words(read, ENTROPY_HIST, layer, microbatch, torch.int32), histogram)
    assert torch.equal(_words(read, QUANTILES, layer, microbatch).view(torch.int32), quantiles.view(torch.int32))
    assert torch.equal(_words(read, QUANTILE_COUNTS, layer, microbatch, torch.int32), counts)


@pytest.mark.parametrize("detailed", [False, True])
def test_row_statistics_are_recorded_as_they_are_with_the_head_histogram_on_detailed_steps_only(
    started_step, capture, recorded, detailed,
):
    recorder, _, _ = started_step
    stats, read = capture
    recorder.begin_step(detailed)
    stats.begin_step(detailed)
    row_stats, histogram, head_histogram, quantiles, counts = records = _row_records(0)
    compute = _Produced(records)

    stats.observe_softmax_rows(1, 1, compute)

    assert compute.calls == 1 and (compute.given[2] is not None) is detailed
    _assert_row_words(read, (row_stats, histogram, quantiles, counts), layer=1, microbatch=1)
    assert recorded(HEAD_ENTROPY_HIST, 1, 1) is detailed
    assert torch.equal(
        _words(read, HEAD_ENTROPY_HIST, 1, 1, torch.int32),
        head_histogram if detailed else torch.zeros_like(head_histogram),
    )
    assert all(recorded(name, 1, 1) for name in (ROW_STATS, ENTROPY_HIST, QUANTILES, QUANTILE_COUNTS))
    assert not recorded(ROW_STATS, 0, 0) and not recorded(HEAD_ENTROPY_HIST, 0, 0)
    assert not _words(read, ROW_STATS, 0, 0).any()


@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize(("first", "second"), [
    (_row_records(1, rows=40), _row_records(2, rows=25)),
    (_row_records_without_rows(), _row_records(3)),
    (_row_records(4), _row_records_without_rows()),
], ids=["both", "first_without_rows", "second_without_rows"])
def test_repeated_row_statistics_merge_exactly_except_for_the_quantiles(started_step, capture, first, second, detailed):
    recorder, _, _ = started_step
    stats, read = capture
    recorder.begin_step(detailed)
    stats.begin_step(detailed)

    first_compute, second_compute = _Produced(first), _Produced(second)
    stats.observe_softmax_rows(0, 0, first_compute)
    stats.observe_softmax_rows(0, 0, second_compute)

    assert first_compute.given is not None and second_compute.given is None
    _assert_row_words(read, (
        _merged_row_stats(first[0], second[0]), first[1] + second[1], torch.full((9,), torch.nan), first[4] + second[4],
    ))
    head_histogram = _words(read, HEAD_ENTROPY_HIST, 0, 0, torch.int32)
    assert torch.equal(head_histogram, first[2] + second[2] if detailed else torch.zeros_like(head_histogram))
    assert _words(read, ROW_STATS)[0].item() == first[0][0].item() + second[0][0].item()


def test_row_statistics_of_a_new_step_replace_those_of_the_last_one(started_step, capture, recorded):
    recorder, _, _ = started_step
    stats, read = capture
    recorder.begin_step(True)
    stats.begin_step(True)
    stats.observe_softmax_rows(0, 0, _Produced(_row_records(5)))
    recorder.begin_step(False)
    stats.begin_step(False)
    row_stats, histogram, head_histogram, quantiles, counts = _row_records(6)
    compute = _Produced((row_stats, histogram, head_histogram, quantiles, counts))

    stats.observe_softmax_rows(0, 0, compute)

    assert compute.given is not None
    _assert_row_words(read, (row_stats, histogram, quantiles, counts))
    assert not recorded(HEAD_ENTROPY_HIST)


def _state(documents: int, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(documents, HEADS, 5, 2, generator=generator).to(dtype)


def _state_expected(state: torch.Tensor) -> torch.Tensor:
    norms = state.double().flatten(2).norm(dim=-1)
    return torch.cat((torch.tensor([float(state.shape[0])], dtype=torch.float64), norms.sum(0), norms.amax(0)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_state_statistics_are_the_frobenius_norms_of_the_final_state_of_each_document(capture, dtype):
    stats, read = capture
    state = _state(4, dtype)

    stats.observe_linear_state(0, 0, state)

    _assert_recorded(read, {STATE: _state_expected(state)})
    assert read(STATE)[0].item() == 4


def test_state_statistics_of_repeated_observations_form_one_population(capture):
    stats, read = capture
    first, second = _state(4, torch.float32, seed=1), _state(3, torch.float32, seed=2)

    stats.observe_linear_state(0, 0, first)
    stats.observe_linear_state(0, 0, second)

    _assert_recorded(read, {STATE: _state_expected(torch.cat((first, second)))})
    assert read(STATE)[0].item() == 7


def test_a_nonfinite_state_makes_only_the_statistics_of_its_head_nonfinite(capture):
    stats, read = capture
    state = _state(4, torch.float32)
    state[1, 2, 0, 1] = float("nan")
    state[3, 0, 4, 0] = float("inf")

    stats.observe_linear_state(0, 0, state)

    words = read(STATE)
    assert words[0].item() == 4
    assert (
        (~torch.isfinite(words[1:1 + HEADS])).tolist()
        == (~torch.isfinite(words[1 + HEADS:])).tolist()
        == [True, False, True]
    )
    assert torch.equal(words[2:3].view(torch.int32), _state_expected(state)[2:3].float().view(torch.int32))


def test_a_state_that_requires_grad_builds_no_graph(started_step, capture):
    recorder, _, _ = started_step
    stats, read = capture
    state = _state(4, torch.float32).requires_grad_()

    stats.observe_linear_state(0, 0, state)

    assert state.grad is None and not recorder.float_buffer().requires_grad
    _assert_recorded(read, {STATE: _state_expected(state.detach())})
