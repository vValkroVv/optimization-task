"""Attention statistics of the metric capture on CUDA, through the sampler's observers and `AttentionStats`.

The records 174-178 of an MLA call reach the recorder's slots word for word as `attention_row_stats`
returns them, repeated calls merge, and the Frobenius norms of the final delta-rule state of every document
reach record 181. The delta-rule test runs fla's `chunk_gated_delta_rule` that returns its final state
(what `Qwen3NextGatedDeltaNet.forward` requests for an observer) and holds the state to a float64 reference
and the gradients to those of a call that does not return it.
"""

import uuid

import pytest
import torch
from torch import nn

from llmfoundry.callbacks.metric_sampler import MetricSampler
from llmfoundry.models.layers.attention import LlamaLatentAttention, Qwen3NextGatedDeltaNet
from llmfoundry.models.layers.attn_row_stats import RECORD_FIELDS, attention_row_stats
from llmfoundry.models.utils import flash_attn_import
from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import RankRecorder
from metric_sampler.capture.ring import ShmRing
from metric_sampler.capture.stats.attention import AttentionStats
from sber_tests.unit.attn_row_stats_reference import reference_rows

pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

DEVICE = "cuda"
HEADS, NOPE, ROPE, VALUE = 4, 128, 64, 128  # MLA: q/k 192 = 128 nope + 64 rope
SCALE = 3 * (NOPE + ROPE) ** -0.5
MIN_KEYS = 32
KEY_HEADS, VALUE_HEADS, KEY_DIM, VALUE_DIM = 2, 4, 64, 64
ROW_RECORDS = (
    "attn/softmax/row_stats",
    "attn/softmax/entropy_hist",
    "attn/softmax/head_entropy_hist",
    "attn/softmax/entropy_quantiles",
    "attn/softmax/entropy_quantile_counts",
)
STATE = "attn/linear/state_stats"


@pytest.mark.skipif(not flash_attn_import.HAS_FA3, reason="requires FA3")
def test_real_fa3_varlen_lse_observation_preserves_output_and_gradients():
    torch.manual_seed(71)
    inputs = [torch.randn(96, 2, 192, device=DEVICE, dtype=torch.bfloat16) for _ in range(3)]
    cu = torch.tensor([0, 64, 96], device=DEVICE, dtype=torch.int32)
    upstream = torch.randn_like(inputs[0])
    results, gradients = [], []
    for observe in (False, True):
        q, k, v = [tensor.clone().requires_grad_() for tensor in inputs]
        result = flash_attn_import.flash_attn_varlen_func_v3_wrapper(
            q, k, v, cu, cu, 64, 64, softmax_scale=0.05, causal=True,
            deterministic=True, return_attn_probs=observe,
        )
        if observe:
            result, lse = result
            assert not lse.requires_grad
            reference = []
            for start, end in ((0, 64), (64, 96)):
                scores = torch.einsum("qhd,khd->hqk", q[start:end].double(), k[start:end].double()) * 0.05
                future = torch.ones(end-start, end-start, device=DEVICE, dtype=torch.bool).triu(1)
                reference.append(scores.masked_fill(future, -torch.inf).logsumexp(-1))
            torch.testing.assert_close(lse.double(), torch.cat(reference, dim=1), rtol=2e-4, atol=1e-4)
        result.backward(upstream)
        results.append(result.detach())
        gradients.append([tensor.grad for tensor in (q, k, v)])
    assert torch.equal(*results)
    for plain, observed in zip(*gradients):
        torch.testing.assert_close(plain, observed, rtol=0, atol=0)
ACTIVATION_POINTS = (
    "residual", "attn_out", "mlp_out", "shared_expert_out", "attn_residual_contribution",
    "mlp_residual_contribution", "swiglu_in", "down_in",
)


class _MLA(LlamaLatentAttention):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.num_heads_per_rank = HEADS
        self.qk_nope_head_dim = NOPE
        self.gated_attention = False
        self.softmax_scale = SCALE
        self.attention_observer = None
        self.gate_observer = None


class _GDN(Qwen3NextGatedDeltaNet):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.num_v_heads = VALUE_HEADS
        self.attention_observer = None


class _Capture:
    """A sampler over a CUDA recorder, its observers installed as for a training forward, without the loop."""

    def __init__(self, directory: str, detailed: bool) -> None:
        self.detailed = detailed
        self.mla, self.gdn = _MLA(), _GDN()
        self.sampler = sampler = MetricSampler(directory, attention_stats=True)
        sampler._softmax_attention = [(0, self.mla)]
        sampler._linear_attention = [(1, self.gdn)]
        sampler.microbatches = 1
        requests = sampler._layout_requests(
            experts=4, routed_layers={}, mtp_layers=[], streams=1, pre_reduce_keys=set(),
            activation_layers={point: [] for point in ACTIVATION_POINTS}, dumps_enabled=False,
        )
        self.layout = Layout([request for request in requests if request[0].name.startswith("attn/")])
        self.ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", self.layout, slot_count=1)
        sampler.recorder = self.recorder = RankRecorder(self.layout, self.ring, torch.device(DEVICE))
        self.recorder.begin_step(detailed)
        sampler.attention_stats = AttentionStats(self.recorder)
        # In the order of _fit_start and batch_start: the extension's observers take the step from begin_step.
        sampler._build_attention_observers()
        sampler.attention_stats.begin_step(detailed)
        sampler._set_attention_observers(True)
        sampler._capture_forward_active = sampler._grad_forward = True

    def words(self, name: str, layer: int) -> torch.Tensor:
        """The int32 words of a slot of microbatch 0, read back from the device buffer."""
        torch.cuda.synchronize()
        slot = self.layout.slot(name, layer, 0)
        return self.recorder.buffer[slot.offset:slot.offset + slot.width].cpu()

    def close(self) -> None:
        self.recorder.close()
        self.ring.unlink()


@pytest.fixture(params=[False, True], ids=["base_step", "detailed_step"])
def capture(request, tmp_path):
    started = _Capture(str(tmp_path), detailed=request.param)
    yield started
    started.close()


def _documents(lengths) -> list[int]:
    return [0, *torch.tensor(lengths).cumsum(0).tolist()]


def _mla_call(lengths, dense: bool, seed: int = 0):
    """The arguments of an MLA observer call over documents of `lengths` (equal ones for a dense batch), the
    bf16 q, k and v of [rows, heads, dim] and flash-attn's log-partition, from the float64 reference."""
    generator = torch.Generator().manual_seed(seed)
    rows = sum(lengths)
    query, key = (torch.randn(rows, HEADS, NOPE + ROPE, generator=generator).to(torch.bfloat16) for _ in range(2))
    value = torch.randn(rows, HEADS, VALUE, generator=generator).to(torch.bfloat16)
    lse = reference_rows(query, key, SCALE, _documents(lengths)).lse.float()
    query, key, value, lse = (x.to(DEVICE) for x in (query, key, value, lse))
    if dense:
        batch, seq = len(lengths), lengths[0]
        query, key, value = (x.reshape(batch, seq, HEADS, -1) for x in (query, key, value))
        return (query, key, value, lse.reshape(HEADS, batch, seq), SCALE), (query, key, lse.reshape(HEADS, batch, seq).transpose(0, 1), SCALE)
    cu_seqlens = torch.tensor(_documents(lengths), dtype=torch.int32, device=DEVICE)
    return ((query, key, value, lse, SCALE, cu_seqlens, max(lengths)),
            (query[None], key[None], lse, SCALE, cu_seqlens, max(lengths)))


def _counted_rows(lengths) -> int:
    return sum(max(length - MIN_KEYS + 1, 0) for length in lengths)


def _assert_row_words(capture: _Capture, records, row_stats_rtol: float = 0.0) -> None:
    for name, record in zip(ROW_RECORDS, records):
        words = capture.words(name, 0)
        if name.endswith("head_entropy_hist") and not capture.detailed:
            assert not words.any()
        elif record.is_floating_point():
            rtol = row_stats_rtol if name == ROW_RECORDS[0] else 0.0
            torch.testing.assert_close(words.view(torch.float32), record.cpu(), rtol=rtol, atol=0, equal_nan=True, msg=name)
        else:
            assert torch.equal(words, record.cpu()), name


@pytest.mark.parametrize(("lengths", "dense"), [([100, 28, 200], False), ([150, 150], True)], ids=["packed", "dense"])
def test_the_row_statistics_of_an_mla_call_reach_the_slots_as_the_kernel_returns_them(capture, lengths, dense):
    observer_args, kernel_args = _mla_call(lengths, dense)

    capture.mla.attention_observer(*observer_args)

    records = attention_row_stats(*kernel_args)
    _assert_row_words(capture, records)
    row_stats, hist = capture.words(ROW_RECORDS[0], 0).view(torch.float32), capture.words(ROW_RECORDS[1], 0)
    counted = _counted_rows(lengths)
    assert row_stats[0].item() == counted and hist.sum().item() == counted * HEADS
    assert capture.words("attn/softmax/lse_stats", 0).view(torch.float32)[0].item() == sum(lengths)


def test_repeated_mla_calls_merge_exactly_and_leave_the_quantiles_unknown(capture):
    first_args, first_kernel = _mla_call([100, 60], dense=False, seed=1)
    second_args, second_kernel = _mla_call([90, 40, 70], dense=False, seed=2)

    capture.mla.attention_observer(*first_args)
    capture.mla.attention_observer(*second_args)

    first, second = attention_row_stats(*first_kernel), attention_row_stats(*second_kernel)
    merged_stats = [first.row_stats[:1] + second.row_stats[:1]]
    for name, mine, theirs in zip(
        RECORD_FIELDS[1:], first.row_stats[1:].reshape(-1, HEADS), second.row_stats[1:].reshape(-1, HEADS)
    ):
        if name.endswith("_min"):
            merged_stats.append(torch.minimum(mine, theirs))
        elif name.endswith("_max"):
            merged_stats.append(torch.maximum(mine, theirs))
        else:
            merged_stats.append(mine + theirs)
    merged_stats = torch.cat(merged_stats)
    assert merged_stats[0].item() == _counted_rows([100, 60]) + _counted_rows([90, 40, 70])
    _assert_row_words(capture, (
        merged_stats,
        first.entropy_hist + second.entropy_hist,
        first.head_entropy_hist + second.head_entropy_hist,
        torch.full_like(first.entropy_quantiles, torch.nan),
        first.entropy_quantile_counts + second.entropy_quantile_counts,
    ), row_stats_rtol=1e-6)


def _state_norm_words(state: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.matrix_norm(state.double().cpu())
    return torch.cat((torch.tensor([float(state.shape[0])], dtype=torch.float64), norms.sum(0), norms.amax(0))).float()


def _delta_rule_observer_args(rows: int, output: torch.Tensor):
    """The stage-1 operands of a GDN observer call, around the kernel `output` [1, rows, heads, dim]."""
    generator = torch.Generator().manual_seed(5)
    beta = torch.rand(1, rows, VALUE_HEADS, generator=generator).to(torch.bfloat16)
    g = -torch.rand(1, rows, VALUE_HEADS, generator=generator)
    value = torch.randn(1, rows, VALUE_HEADS, VALUE_DIM, generator=generator).to(torch.bfloat16)
    a_log, dt_bias = torch.randn(VALUE_HEADS, generator=generator), torch.randn(VALUE_HEADS, generator=generator)
    return [x.to(DEVICE) for x in (beta, g, value)] + [output, a_log.to(DEVICE), dt_bias.to(DEVICE)]


def test_the_final_state_norms_of_a_gdn_call_reach_the_slot(capture):
    state = torch.randn(3, VALUE_HEADS, KEY_DIM, VALUE_DIM, device=DEVICE)
    output = torch.randn(1, 50, VALUE_HEADS, VALUE_DIM, device=DEVICE, dtype=torch.bfloat16)

    capture.gdn.attention_observer(*_delta_rule_observer_args(50, output), state)

    torch.testing.assert_close(
        capture.words(STATE, 1).view(torch.float32), _state_norm_words(state), rtol=1e-5, atol=0,
    )
    assert capture.words("attn/linear/gate_stats", 1).view(torch.float32)[0].item() == 50


def _normalized(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1).to(torch.bfloat16)


def _reference_final_states(k, v, g, beta, boundaries) -> torch.Tensor:
    """float64 [documents, value heads, key dim, value dim]: h = a h, then h += k (b (v - h^T k))^T per head,
    a = exp(g), each value head reading its group's key head."""
    group = VALUE_HEADS // KEY_HEADS
    states = []
    for begin, end in zip(boundaries, boundaries[1:]):
        state = torch.zeros(VALUE_HEADS, KEY_DIM, VALUE_DIM, dtype=torch.float64)
        for step in range(begin, end):
            key = k[0, step].double().repeat_interleave(group, dim=0)
            state = state * g[0, step].double().exp()[:, None, None]
            update = beta[0, step].double()[:, None] * (v[0, step].double() - torch.einsum("hkv,hk->hv", state, key))
            state = state + torch.einsum("hk,hv->hkv", key, update)
        states.append(state)
    return torch.stack(states)


def test_a_delta_rule_call_that_returns_its_final_state_trains_as_one_that_does_not(capture):
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    lengths = [100, 28, 200]
    boundaries, rows = _documents(lengths), sum(lengths)
    generator = torch.Generator().manual_seed(7)
    q = _normalized(torch.randn(1, rows, KEY_HEADS, KEY_DIM, generator=generator))
    k = _normalized(torch.randn(1, rows, KEY_HEADS, KEY_DIM, generator=generator))
    v = torch.randn(1, rows, VALUE_HEADS, VALUE_DIM, generator=generator).to(torch.bfloat16)
    g = -torch.rand(1, rows, VALUE_HEADS, generator=generator) * 0.2
    beta = torch.rand(1, rows, VALUE_HEADS, generator=generator).to(torch.bfloat16)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=DEVICE)

    def run(output_final_state: bool):
        inputs = [x.to(DEVICE).requires_grad_() for x in (q, k, v, g, beta)]
        output, state = fla.chunk_gated_delta_rule(
            q=inputs[0], k=inputs[1], v=inputs[2], g=inputs[3], beta=inputs[4],
            output_final_state=output_final_state, use_qk_l2norm_in_kernel=False, cu_seqlens=cu_seqlens,
        )
        output.float().pow(2).sum().backward()
        return output.detach(), state, [x.grad for x in inputs]

    plain_output, plain_state, plain_grads = run(False)
    output, state, grads = run(True)

    assert plain_state is None and state.shape == (len(lengths), VALUE_HEADS, KEY_DIM, VALUE_DIM)
    torch.testing.assert_close(output, plain_output, rtol=0, atol=0)
    for name, got, want in zip("qkvgb", grads, plain_grads):
        torch.testing.assert_close(got.float(), want.float(), rtol=1e-2, atol=1e-2 * want.float().abs().max().item(), msg=name)
    reference = _reference_final_states(k, v, g, beta, boundaries)
    torch.testing.assert_close(
        torch.linalg.matrix_norm(state.detach().double().cpu()), torch.linalg.matrix_norm(reference), rtol=5e-2, atol=1e-3,
    )

    capture.gdn.attention_observer(*_delta_rule_observer_args(rows, output), state)

    torch.testing.assert_close(
        capture.words(STATE, 1).view(torch.float32), _state_norm_words(state.detach()), rtol=1e-5, atol=0,
    )


def _two_microbatch_capture(directory: str, native: bool):
    """A sampler over a CUDA recorder for two microbatches with an MLA layer (0), a GDN layer (1) and a gated MLA
    layer (2), whose observers are the extension's or, without `native`, the Python ones."""
    mla, gdn, gated = _MLA(), _GDN(), _MLA()
    gated.gated_attention = True
    sampler = MetricSampler(directory, attention_stats=True)
    sampler._softmax_attention = [(0, mla), (2, gated)]
    sampler._linear_attention = [(1, gdn)]
    sampler.microbatches = 2
    requests = sampler._layout_requests(
        experts=4, routed_layers={}, mtp_layers=[], streams=1, pre_reduce_keys=set(),
        activation_layers={point: [] for point in ACTIVATION_POINTS}, dumps_enabled=False,
    )
    layout = Layout([request for request in requests if request[0].name.startswith("attn/")])
    ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    sampler.recorder = RankRecorder(layout, ring, torch.device(DEVICE))
    sampler.attention_stats = AttentionStats(sampler.recorder)
    if not native:
        sampler.attention_stats.native_calls = lambda sampler: None
    sampler._build_attention_observers()
    sampler._set_attention_observers(True)
    sampler._capture_forward_active = sampler._grad_forward = True
    return sampler, (mla, gdn, gated), ring


@pytest.mark.skipif(activation_ext.prebuilt() is None, reason="requires the prebuilt extension")
@pytest.mark.parametrize("detailed", [False, True], ids=["base_step", "detailed_step"])
def test_the_extensions_attention_observers_write_every_word_the_python_ones_write(tmp_path, detailed):
    runs = []
    for native in (True, False):
        sampler, (mla, gdn, gated), ring = _two_microbatch_capture(str(tmp_path / f"native-{native}"), native)
        assert (type(mla.attention_observer).__name__ == "SoftmaxAttentionCall") == native
        steps = []
        try:
            for step in range(2):
                sampler.recorder.begin_step(detailed)
                sampler.attention_stats.begin_step(detailed)
                for microbatch in range(2):
                    sampler.microbatch = microbatch
                    seed = 10 * step + microbatch
                    packed, _ = _mla_call([100, 28, 200], dense=False, seed=seed)
                    dense, _ = _mla_call([150, 150], dense=True, seed=seed + 100)
                    mla.attention_observer(*packed)
                    mla.attention_observer(*dense)
                    gated.attention_observer(*packed)
                    generator = torch.Generator().manual_seed(seed + 200)
                    gated.gate_observer(torch.randn(2, 150, HEADS, VALUE, generator=generator).to(DEVICE, torch.bfloat16))
                    output = torch.randn(1, 64, VALUE_HEADS, VALUE_DIM, generator=generator).to(DEVICE, torch.bfloat16)
                    state = torch.randn(3, VALUE_HEADS, KEY_DIM, VALUE_DIM, generator=generator).to(DEVICE)
                    # Microbatch 0 records the decay parameters and a final state; microbatch 1 calls twice.
                    gdn.attention_observer(*_delta_rule_observer_args(64, output), state if microbatch == 0 else None)
                    if microbatch == 1:
                        gdn.attention_observer(*_delta_rule_observer_args(64, output), None)
                torch.cuda.synchronize()
                steps.append((sampler.recorder.buffer.cpu().clone(), sampler.recorder._recorded.copy()))
        finally:
            sampler.recorder.close()
            ring.unlink()
        runs.append(steps)

    for (native_words, native_flags), (python_words, python_flags) in zip(*runs):
        assert native_flags.any() and (native_flags == python_flags).all()
        assert torch.equal(native_words, python_words)
