"""Attention statistics of the capture: which layers have slots, how the observers are wired to the
modules and to AttentionStats, which forwards they record, and that a failure then stops training."""

import uuid
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("composer", reason="Requires the CPU training runtime")

import torch  # noqa: E402
from torch import nn  # noqa: E402

import llmfoundry.callbacks.metric_sampler as sampler_module  # noqa: E402
from llmfoundry.callbacks.metric_sampler import MetricSampler, _GuardedObserver  # noqa: E402
from llmfoundry.models.layers.attention import LlamaLatentAttention, Qwen3NextGatedDeltaNet  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.recorder import RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.capture.stats.attention import AttentionStats  # noqa: E402
from sber_tests.unit.attn_row_stats_reference import reference_records, reference_rows  # noqa: E402
from sber_tests.unit.cpu.test_attention_observers import (  # noqa: E402
    HEADS as GATED_HEADS, _FlashStub, _GatedMLA, _forward, _reference_lse, _stand_in_forward_kernels,
)

HEADS, NOPE, ROPE, VALUE, VALUE_HEADS, ROWS = 2, 5, 3, 4, 3, 6
SCALE = 0.3
PACKED = (torch.tensor([0, 2, 6], dtype=torch.int32), 4)
LSE, QKV, GATE = "attn/softmax/lse_stats", "attn/softmax/qkv_stats", "attn/softmax/gate_stats"
ROW_STATS, ENTROPY_HIST, HEAD_ENTROPY_HIST = "attn/softmax/row_stats", "attn/softmax/entropy_hist", "attn/softmax/head_entropy_hist"
QUANTILES, QUANTILE_COUNTS = "attn/softmax/entropy_quantiles", "attn/softmax/entropy_quantile_counts"
LINEAR_GATE, LINEAR_NORM, DECAY = "attn/linear/gate_stats", "attn/linear/norm_stats", "attn/linear/decay_params"
STATE = "attn/linear/state_stats"
ROW_RECORDS = (ROW_STATS, ENTROPY_HIST, HEAD_ENTROPY_HIST, QUANTILES, QUANTILE_COUNTS)


class _RowStats:
    """`attention_row_stats` on the torch reference of its records, keeping what it was called with and
    holding it to its shapes: 4-D query and key, the log-partition [heads, tokens] for a packed batch and
    [batch, heads, seq] otherwise. Given `out` it writes the records into it, its histograms zero."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.outs: list = []

    @torch.no_grad()
    def __call__(self, query, key, lse, softmax_scale, cu_seqlens=None, max_seqlen=None, out=None):
        self.calls.append((query, key, lse, softmax_scale, cu_seqlens, max_seqlen))
        self.outs.append(out)
        batch, seq, heads = query.shape[:3]
        assert key.shape == query.shape
        assert lse.shape == ((heads, batch * seq) if cu_seqlens is not None else (batch, heads, seq))
        boundaries = cu_seqlens.tolist() if cu_seqlens is not None else list(range(0, batch * seq + 1, seq))
        reference = reference_rows(query.flatten(0, 1), key.flatten(0, 1), softmax_scale, boundaries)
        records = reference_records(reference.rows.float(), reference.num_keys)
        if out is not None:
            for index, (target, record) in enumerate(zip(out, records)):
                if target is not None:
                    assert index not in (1, 2) or not target.any()
                    target.copy_(record)
        return records


@pytest.fixture(autouse=True)
def row_stats(monkeypatch):
    stub = _RowStats()
    monkeypatch.setattr(sampler_module, "attention_row_stats", stub)
    return stub


class _MLA(LlamaLatentAttention):
    def __init__(self, gated: bool) -> None:
        nn.Module.__init__(self)
        self.num_heads_per_rank = HEADS
        self.qk_nope_head_dim = NOPE
        self.gated_attention = gated
        self.attention_observer = None
        self.gate_observer = None


class _GDN(Qwen3NextGatedDeltaNet):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.num_v_heads = VALUE_HEADS
        self.attention_observer = None


def _layout_requests(sampler: MetricSampler, microbatches: int = 2):
    sampler.microbatches = microbatches
    activation_layers = {point: [] for point in (
        "residual", "attn_out", "mlp_out", "shared_expert_out", "attn_residual_contribution",
        "mlp_residual_contribution", "swiglu_in", "down_in",
    )}
    return sampler._layout_requests(
        experts=4, routed_layers={}, mtp_layers=[], streams=1, pre_reduce_keys=set(),
        activation_layers=activation_layers, dumps_enabled=False,
    )


@pytest.fixture
def recording():
    """Starts a step of the sampler with a CPU recorder over its attention slots, its observers installed
    for a training forward; returns what the step delivers for a slot."""
    started = []

    def start(sampler, detailed=False):
        requests = [request for request in _layout_requests(sampler) if request[0].name.startswith("attn/")]
        layout = Layout(requests)
        ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
        sampler.recorder = RankRecorder(layout, ring, torch.device("cpu"))
        started.append((sampler.recorder, ring))
        sampler.recorder.begin_step(detailed)
        sampler.attention_stats = AttentionStats(sampler.recorder)
        sampler.attention_stats.begin_step(detailed)
        sampler._build_attention_observers()
        sampler._set_attention_observers(True)
        sampler._capture_forward_active = sampler._grad_forward = True
        delivered = []

        def read(name, layer, microbatch):
            if not delivered:
                sampler.recorder.step_end(step=1)
                delivered.append(torch.from_numpy(ring.payload(ring.ready_slots()[0]).copy()))
            slot = layout.slot(name, layer, microbatch)
            return delivered[0][slot.offset:slot.offset + slot.width].clone().view(torch.float32)

        return read

    yield start
    for recorder, ring in started:
        recorder.close()
        ring.unlink()


@pytest.fixture
def detailed():
    return False


@pytest.fixture
def capture(tmp_path, recording, detailed):
    sampler = MetricSampler(str(tmp_path), attention_stats=True)
    gated_layer, plain_layer, linear_layer = _MLA(gated=True), _MLA(gated=False), _GDN()
    sampler._softmax_attention = [(0, gated_layer), (2, plain_layer)]
    sampler._linear_attention = [(1, linear_layer)]
    return sampler, gated_layer, plain_layer, linear_layer, recording(sampler, detailed)


def _softmax_inputs():
    generator = torch.Generator().manual_seed(0)
    query, key = (torch.randn(ROWS, HEADS, NOPE + ROPE, generator=generator) for _ in range(2))
    value = torch.randn(ROWS, HEADS, VALUE, generator=generator)
    lse = torch.randn(HEADS, ROWS, generator=generator) + 5
    gate = torch.randn(ROWS, HEADS, VALUE, generator=generator)
    return query, key, value, lse, gate


def _state():
    return torch.randn(4, VALUE_HEADS, 5, 2, generator=torch.Generator().manual_seed(2))


def _softmax_call(varlen: bool):
    """The arguments of an observer call over two documents of at least 32 rows, packed or as a batch, and the
    boundaries of the documents in the [rows, heads, dim] layout."""
    generator = torch.Generator().manual_seed(3)
    lengths = (40, 33) if varlen else (40, 40)
    rows = sum(lengths)
    query, key = (torch.randn(rows, HEADS, NOPE + ROPE, generator=generator) for _ in range(2))
    value = torch.randn(rows, HEADS, VALUE, generator=generator)
    lse = torch.randn(HEADS, rows, generator=generator) + 5
    boundaries = [0, lengths[0], rows]
    if varlen:
        return (query, key, value, lse, SCALE, torch.tensor(boundaries, dtype=torch.int32), max(lengths)), boundaries
    def batched(x):
        return x.reshape(2, 40, HEADS, -1)

    return (batched(query), batched(key), batched(value), lse.reshape(HEADS, 2, 40), SCALE), boundaries


def _linear_inputs():
    generator = torch.Generator().manual_seed(1)
    beta, g = torch.rand(ROWS, VALUE_HEADS, generator=generator), -torch.rand(ROWS, VALUE_HEADS, generator=generator)
    value, output = (torch.randn(ROWS, VALUE_HEADS, VALUE, generator=generator) for _ in range(2))
    return beta, g, value, output, torch.randn(VALUE_HEADS, generator=generator), torch.randn(VALUE_HEADS, generator=generator)


def test_attention_layers_are_found_in_trunk_and_mtp_layers_through_their_wrappers(tmp_path):
    trunk_mla, mtp_mla, gdn = _MLA(gated=False), _MLA(gated=True), _GDN()
    layers = [
        SimpleNamespace(self_attn=trunk_mla),
        SimpleNamespace(self_attn=SimpleNamespace(_attn=gdn)),
        SimpleNamespace(self_attn=SimpleNamespace(_attn=nn.Linear(1, 1))),
    ]
    mtp_layers = [(3, SimpleNamespace(self_attn=SimpleNamespace(base_attn=SimpleNamespace(_attn=mtp_mla))))]
    sampler = MetricSampler(str(tmp_path), attention_stats=True)
    sampler.causal_lm = SimpleNamespace(model=SimpleNamespace(layers=layers))

    sampler._find_attention_layers(mtp_layers)

    assert sampler._softmax_attention == [(0, trunk_mla), (3, mtp_mla)]
    assert sampler._linear_attention == [(1, gdn)]


def test_a_model_without_mla_or_gdn_layers_cannot_record_attention_statistics(tmp_path):
    layers = [SimpleNamespace(self_attn=SimpleNamespace(_attn=nn.Linear(1, 1)))]
    sampler = MetricSampler(str(tmp_path), attention_stats=True)
    sampler.causal_lm = SimpleNamespace(model=SimpleNamespace(layers=layers))

    with pytest.raises(ValueError, match="attention_stats requires"):
        sampler._find_attention_layers([])


def test_layout_requests_every_attention_layer_each_microbatch_and_the_decay_parameters_once(tmp_path):
    sampler = MetricSampler(str(tmp_path), attention_stats=True)
    sampler._softmax_attention = [(0, _MLA(gated=True)), (2, _MLA(gated=False))]
    sampler._linear_attention = [(1, _GDN())]

    requests = _layout_requests(sampler)

    Layout(requests)
    attention = {(metric.name, layer, microbatch): width for metric, layer, microbatch, width in requests
                 if metric.name.startswith("attn/")}
    expected = {}
    for microbatch in (0, 1):
        expected |= {
            (LSE, 0, microbatch): 1 + 3 * HEADS, (QKV, 0, microbatch): 1 + 8 * HEADS, (GATE, 0, microbatch): 1 + 3 * HEADS,
            (LSE, 2, microbatch): 1 + 3 * HEADS, (QKV, 2, microbatch): 1 + 8 * HEADS,
            (LINEAR_GATE, 1, microbatch): 1 + 7 * VALUE_HEADS, (LINEAR_NORM, 1, microbatch): 1 + 3 * VALUE_HEADS,
            (STATE, 1, microbatch): 1 + 2 * VALUE_HEADS,
        }
        for layer in (0, 2):
            expected |= {
                (ROW_STATS, layer, microbatch): 1 + 21 * HEADS, (ENTROPY_HIST, layer, microbatch): 32,
                (HEAD_ENTROPY_HIST, layer, microbatch): 32 * HEADS, (QUANTILES, layer, microbatch): 9,
                (QUANTILE_COUNTS, layer, microbatch): 5,
            }
    expected[(DECAY, 1, -1)] = 2 * VALUE_HEADS
    assert attention == expected


def test_a_sampler_without_attention_stats_requests_no_attention_slots(tmp_path):
    sampler = MetricSampler(str(tmp_path))

    assert not [request for request in _layout_requests(sampler) if request[0].name.startswith("attn/")]
    assert sampler.attention_stats is None and not sampler.attention_stats_enabled


def test_observers_are_installed_for_a_forward_and_cleared_by_the_end_of_it_and_by_disable(capture):
    sampler, gated_layer, plain_layer, linear_layer, _ = capture
    modules = [gated_layer, plain_layer, linear_layer]

    assert all(module.attention_observer is not None for module in modules)
    assert gated_layer.gate_observer is not None and plain_layer.gate_observer is None
    sampler._set_attention_observers(False)
    assert all(module.attention_observer is None for module in modules) and gated_layer.gate_observer is None
    sampler._set_attention_observers(True)
    sampler.disable(RuntimeError("failure"), "test")
    assert all(module.attention_observer is None for module in modules) and gated_layer.gate_observer is None


@pytest.mark.parametrize("microbatch", [0, 1])
def test_observed_tensors_reach_the_slots_of_their_layer_and_microbatch(capture, microbatch):
    sampler, gated_layer, plain_layer, linear_layer, read = capture
    query, key, value, lse, gate = _softmax_inputs()
    beta, g, linear_value, output, a_log, dt_bias = _linear_inputs()
    sampler.microbatch = microbatch

    gated_layer.attention_observer(query, key, value, lse, SCALE, *PACKED)
    gated_layer.gate_observer(gate)
    linear_layer.attention_observer(beta, g, linear_value, output, a_log, dt_bias, None)

    qkv = read(QKV, 0, microbatch)
    assert qkv[0].item() == ROWS
    nope, rope = qkv[1:1 + HEADS], qkv[1 + HEADS:1 + 2 * HEADS]
    torch.testing.assert_close(nope, query[..., :NOPE].norm(dim=-1).sum(0), rtol=1e-6, atol=0)
    torch.testing.assert_close(rope, query[..., NOPE:].norm(dim=-1).sum(0), rtol=1e-6, atol=0)
    torch.testing.assert_close(read(LSE, 0, microbatch)[1:1 + HEADS], lse.sum(1), rtol=1e-6, atol=0)
    torch.testing.assert_close(read(GATE, 0, microbatch)[1:1 + HEADS], gate.mean(-1).sum(0), rtol=1e-6, atol=0)
    torch.testing.assert_close(read(LINEAR_GATE, 1, microbatch)[1:1 + VALUE_HEADS], beta.sum(0), rtol=1e-6, atol=0)
    torch.testing.assert_close(read(LINEAR_NORM, 1, microbatch)[1 + VALUE_HEADS:1 + 2 * VALUE_HEADS],
                               output.norm(dim=-1).sum(0), rtol=1e-6, atol=0)
    other = 1 - microbatch
    assert not read(LSE, 0, other).any() and not read(LINEAR_GATE, 1, other).any()
    decay = read(DECAY, 1, -1)
    if microbatch == 0:
        expected = torch.cat((a_log.exp(), torch.nn.functional.softplus(dt_bias)))
        torch.testing.assert_close(decay, expected, rtol=1e-6, atol=0)
    else:
        assert not decay.any()


@pytest.mark.parametrize(("forward_active", "grad_forward"), [(False, True), (True, False)])
def test_only_the_forward_of_a_training_microbatch_is_recorded(capture, row_stats, forward_active, grad_forward):
    sampler, gated_layer, plain_layer, linear_layer, read = capture
    sampler._capture_forward_active, sampler._grad_forward = forward_active, grad_forward
    query, key, value, lse, gate = _softmax_inputs()

    gated_layer.attention_observer(query, key, value, lse, SCALE, *PACKED)
    gated_layer.gate_observer(gate)
    linear_layer.attention_observer(*_linear_inputs(), _state())

    assert not read(LSE, 0, 0).any() and not read(GATE, 0, 0).any()
    assert not read(LINEAR_GATE, 1, 0).any() and not read(DECAY, 1, -1).any()
    assert not any(read(name, 0, 0).any() for name in ROW_RECORDS) and not read(STATE, 1, 0).any()
    assert row_stats.calls == []


def test_the_recomputed_forward_of_a_backward_pass_is_not_recorded(capture, row_stats):
    sampler, gated_layer, plain_layer, linear_layer, read = capture
    query, key, value, lse, gate = _softmax_inputs()
    trigger = torch.ones(1, requires_grad=True)

    def recompute(gradient):
        plain_layer.attention_observer(query, key, value, lse, SCALE, *PACKED)
        linear_layer.attention_observer(*_linear_inputs(), _state())
        gated_layer.gate_observer(gate)

    trigger.register_hook(recompute)
    (trigger * 2).sum().backward()

    assert not read(LSE, 2, 0).any() and not read(LINEAR_GATE, 1, 0).any() and not read(GATE, 0, 0).any()
    assert not any(read(name, 2, 0).any() for name in ROW_RECORDS) and not read(STATE, 1, 0).any()
    assert row_stats.calls == []


@pytest.mark.parametrize("attention_stats", [False, True])
def test_a_failing_observer_stops_training_only_when_attention_stats_are_requested(tmp_path, attention_stats):
    sampler = MetricSampler(str(tmp_path), attention_stats=attention_stats)
    failing = ZeroDivisionError("failure")

    def fail() -> None:
        raise failing

    guarded = _GuardedObserver(fail, sampler, "attention_observer(layer=0)")
    if attention_stats:
        with pytest.raises(ZeroDivisionError):
            guarded()
        with pytest.raises(ZeroDivisionError):
            sampler._guard("before_forward", fail)
        assert not sampler.disabled
    else:
        guarded()
        assert sampler.disabled


def test_a_gated_attention_forward_is_recorded_through_the_sampler_observers(tmp_path, monkeypatch, recording):
    _FlashStub(monkeypatch, "fa3")
    _stand_in_forward_kernels(monkeypatch)
    torch.manual_seed(0)
    sampler = MetricSampler(str(tmp_path), attention_stats=True)
    attention = _GatedMLA(hidden=6, gated=True)
    sampler._softmax_attention = [(0, attention)]
    read = recording(sampler)
    hidden = torch.randn(2, 5, 6)

    _forward(attention, hidden)

    with torch.no_grad():
        query, k_nope, k_rope, value = attention._compute_qkv(hidden)
        key = torch.cat((k_nope, k_rope), dim=-1)
        lse = _reference_lse(query.flatten(0, 1), key.flatten(0, 1), torch.arange(2).repeat_interleave(5))
        gate = attention.gate_proj(hidden).reshape(10, GATED_HEADS, -1).mean(-1)
    for name in (LSE, QKV, GATE):
        assert read(name, 0, 0)[0].item() == 10
    reference = reference_rows(query.flatten(0, 1), key.flatten(0, 1), attention.softmax_scale, [0, 5, 10])
    for name, words in zip(ROW_RECORDS, reference_records(reference.rows.float(), reference.num_keys)):
        if name != HEAD_ENTROPY_HIST:
            assert torch.equal(read(name, 0, 0).view(torch.int32), words.view(torch.int32)), name
    expected_lse = torch.cat((lse.sum(1), lse.amin(1), lse.amax(1)))
    torch.testing.assert_close(read(LSE, 0, 0)[1:], expected_lse, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read(QKV, 0, 0)[1 + 7 * GATED_HEADS:], value.abs().amax(dim=(0, 1, 3)), rtol=1e-6, atol=0)
    torch.testing.assert_close(read(GATE, 0, 0)[1:1 + GATED_HEADS], gate.sum(0), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("microbatch", [0, 1])
def test_softmax_row_statistics_reach_the_slots_of_their_layer_and_microbatch(capture, varlen, detailed, microbatch):
    sampler, _, plain_layer, _, read = capture
    sampler.microbatch = microbatch
    args, boundaries = _softmax_call(varlen)
    reference = reference_rows(args[0].flatten(0, 1) if not varlen else args[0],
                               args[1].flatten(0, 1) if not varlen else args[1], SCALE, boundaries)
    expected = reference_records(reference.rows.float(), reference.num_keys)

    plain_layer.attention_observer(*args)

    for name, words in zip(ROW_RECORDS, expected):
        if name == HEAD_ENTROPY_HIST and not detailed:
            assert not read(name, 2, microbatch).any()
        else:
            assert torch.equal(read(name, 2, microbatch).view(torch.int32), words.view(torch.int32)), name
    documents = [end - begin for begin, end in zip(boundaries, boundaries[1:])]
    assert read(ROW_STATS, 2, microbatch)[0].item() == sum(length - 31 for length in documents)
    for name in ROW_RECORDS:
        assert not read(name, 0, microbatch).any() and not read(name, 2, 1 - microbatch).any()


@pytest.mark.parametrize("detailed", [False, True])
def test_the_first_call_of_a_layer_and_microbatch_writes_its_row_statistics_into_the_slots_and_a_repeat_merges(
    capture, row_stats, detailed,
):
    _, _, plain_layer, _, read = capture
    args, boundaries = _softmax_call(varlen=True)
    reference = reference_rows(args[0], args[1], SCALE, boundaries)
    expected = reference_records(reference.rows.float(), reference.num_keys)

    plain_layer.attention_observer(*args)
    plain_layer.attention_observer(*args)

    first, second = row_stats.outs
    assert len(first) == 5 and (first[2] is not None) is detailed and second is None
    assert [tensor.dtype for tensor in first if tensor is not None] == [
        tensor.dtype for index, tensor in enumerate(expected) if first[index] is not None
    ]
    assert read(ROW_STATS, 2, 0)[0].item() == 2 * expected[0][0].item()
    assert torch.equal(read(ENTROPY_HIST, 2, 0).view(torch.int32), 2 * expected[1])
    assert torch.equal(read(QUANTILE_COUNTS, 2, 0).view(torch.int32), 2 * expected[4])
    assert read(QUANTILES, 2, 0).isnan().all()
    head_hist = read(HEAD_ENTROPY_HIST, 2, 0).view(torch.int32)
    assert torch.equal(head_hist, 2 * expected[2] if detailed else torch.zeros_like(head_hist))


@pytest.mark.parametrize("varlen", [False, True])
def test_the_row_statistics_kernel_gets_a_packed_batch_as_one_row_without_copies(capture, row_stats, varlen):
    _, _, plain_layer, _, _ = capture
    args, _ = _softmax_call(varlen)
    query, key, _, lse = args[:4]

    plain_layer.attention_observer(*args)

    ((kernel_query, kernel_key, kernel_lse, softmax_scale, cu_seqlens, max_seqlen),) = row_stats.calls
    assert softmax_scale == SCALE and kernel_lse.data_ptr() == lse.data_ptr()
    if varlen:
        assert kernel_query.shape == (1, 73, HEADS, NOPE + ROPE) and kernel_key.shape == kernel_query.shape
        assert kernel_query.data_ptr() == query.data_ptr() and kernel_key.data_ptr() == key.data_ptr()
        assert kernel_lse is lse and cu_seqlens is args[5] and max_seqlen == 40 and type(max_seqlen) is int
    else:
        assert kernel_query is query and kernel_key is key
        assert torch.equal(kernel_lse, lse.transpose(0, 1)) and cu_seqlens is None and max_seqlen is None


@pytest.mark.parametrize("microbatch", [0, 1])
def test_the_final_state_reaches_the_slot_of_its_layer_and_microbatch(capture, microbatch):
    sampler, _, _, linear_layer, read = capture
    sampler.microbatch = microbatch
    state = _state()

    linear_layer.attention_observer(*_linear_inputs(), state)

    norms = state.double().flatten(2).norm(dim=-1)
    expected = torch.cat((torch.tensor([4.0], dtype=torch.float64), norms.sum(0), norms.amax(0)))
    torch.testing.assert_close(read(STATE, 1, microbatch), expected.float(), rtol=1e-6, atol=0)
    assert not read(STATE, 1, 1 - microbatch).any()


def test_a_delta_rule_call_without_a_final_state_records_no_state_statistics(capture):
    _, _, _, linear_layer, read = capture

    linear_layer.attention_observer(*_linear_inputs(), None)

    assert read(LINEAR_GATE, 1, 0)[0].item() == ROWS
    assert not read(STATE, 1, 0).any()
