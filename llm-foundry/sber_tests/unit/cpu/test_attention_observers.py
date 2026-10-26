"""Attention observers of the metric capture: the flash-attn log-partition is requested only for an
installed observer, the observers see the tensors the kernels ran on, and the module's output is the
same with and without them. The flash-attn calls and the delta-rule kernels are stood in by dense torch
references, so everything but the kernels runs as in training."""

import inspect
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import llmfoundry.models.layers.attention as attention_module
from llmfoundry.models.layers.attention import LlamaLatentAttention, Qwen3NextGatedDeltaNet
from llmfoundry.models.utils import flash_attn_import

pytestmark = pytest.mark.unit

HEADS, NOPE, ROPE, VALUE = 2, 128, 64, 128
SCALE = 0.05


def _reference_lse(query, key, doc_ids):
    """Natural-log LSE [heads, rows] of the scaled scores of [rows, heads, dim] tensors, causal within a document."""
    scores = torch.einsum("qhd,khd->hqk", query, key) * SCALE
    rows = torch.arange(query.shape[0])
    allowed = (rows[None, :] <= rows[:, None]) & (doc_ids[None, :] == doc_ids[:, None])
    return torch.logsumexp(scores.masked_fill(~allowed, -math.inf), dim=-1)


def _causal_attention(query, key, value, doc_ids):
    scores = torch.einsum("qhd,khd->hqk", query, key) * SCALE
    rows = torch.arange(query.shape[0])
    allowed = (rows[None, :] <= rows[:, None]) & (doc_ids[None, :] == doc_ids[:, None])
    probabilities = scores.masked_fill(~allowed, -math.inf).softmax(dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, value)


class _FlashStub:
    """flash-attn's dense and varlen calls on torch. `style` is how a build returns the log-partition:
    flash-attn 2 as (out, lse, None), flash-attn 3 as (out, lse)."""

    def __init__(self, monkeypatch, style: str) -> None:
        self.style = style
        self.requested: list[bool] = []
        monkeypatch.setattr(attention_module, "flash_attn_func", self.dense)
        monkeypatch.setattr(attention_module, "flash_attn_varlen_func", self.varlen)

    def _result(self, output, lse, requested):
        self.requested.append(requested)
        if not requested:
            return output
        return (output, lse, None) if self.style == "fa2" else (output, lse)

    def dense(self, query, key, value, causal, deterministic, softmax_scale, return_attn_probs=False):
        assert causal and softmax_scale == SCALE
        batch, seq = query.shape[:2]
        flat = [x.flatten(0, 1) for x in (query, key, value)]
        doc_ids = torch.arange(batch).repeat_interleave(seq)
        output = _causal_attention(*flat, doc_ids).reshape(batch, seq, HEADS, -1)
        lse = _reference_lse(flat[0], flat[1], doc_ids).reshape(HEADS, batch, seq).transpose(0, 1)
        return self._result(output, lse, return_attn_probs)

    def varlen(self, query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
               softmax_scale, causal, deterministic, return_attn_probs=False):
        assert causal and softmax_scale == SCALE and torch.equal(cu_seqlens_q, cu_seqlens_k)
        doc_ids = torch.arange(len(cu_seqlens_q) - 1).repeat_interleave(cu_seqlens_q.diff())
        output = _causal_attention(query, key, value, doc_ids)
        return self._result(output, _reference_lse(query, key, doc_ids), return_attn_probs)


class _Observer:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args) -> None:
        self.calls.append(tuple(arg.detach().clone() if isinstance(arg, torch.Tensor) else arg for arg in args))


class _MLA(LlamaLatentAttention):
    """`LlamaLatentAttention` without its projections: what `_flash_attention_mla_forward` reads."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.softmax_scale = SCALE
        self.deterministic = False
        self.v_head_dim = VALUE
        self._qk_logit_active = False
        self.attention_observer = None
        self.gate_observer = None


def _qkv(batch: int, seq: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    query, key = (torch.randn(batch, seq, HEADS, NOPE + ROPE, generator=generator) for _ in range(2))
    return query, key, torch.randn(batch, seq, HEADS, VALUE, generator=generator)


@pytest.mark.parametrize("style", ["fa2", "fa3"])
@pytest.mark.parametrize("padded_value", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
def test_the_observer_gets_the_kernel_inputs_and_the_head_first_log_partition(monkeypatch, style, padded_value, varlen):
    flash = _FlashStub(monkeypatch, style)
    query, key, value = _qkv(2, 5)
    sent_value = F.pad(value, [0, NOPE + ROPE - VALUE]) if padded_value else value
    cu_seqlens = torch.tensor([0, 3, 5, 8, 10], dtype=torch.int32) if varlen else None
    attention = _MLA()
    expected_output = attention._flash_attention_mla_forward(query, key, sent_value, None, cu_seqlens, 5 if varlen else None)
    observer = attention.attention_observer = _Observer()

    output = attention._flash_attention_mla_forward(query, key, sent_value, None, cu_seqlens, 5 if varlen else None)

    assert flash.requested == [False, True]
    assert torch.equal(output, expected_output)
    ((seen_query, seen_key, seen_value, lse, softmax_scale, *packed),) = observer.calls
    assert softmax_scale == SCALE
    rows = query.flatten(0, 1)
    doc_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3, 3]) if varlen else torch.arange(2).repeat_interleave(5)
    expected_lse = _reference_lse(rows, key.flatten(0, 1), doc_ids)
    if varlen:
        assert torch.equal(seen_query, rows) and torch.equal(seen_key, key.flatten(0, 1))
        assert torch.equal(seen_value, value.flatten(0, 1))
        assert lse.shape == (HEADS, 10)
        torch.testing.assert_close(lse, expected_lse, rtol=1e-6, atol=1e-6)
        seen_cu_seqlens, max_seqlen = packed
        assert torch.equal(seen_cu_seqlens, cu_seqlens) and max_seqlen == 5 and type(max_seqlen) is int
    else:
        assert packed == []
        assert torch.equal(seen_query, query) and torch.equal(seen_key, key) and torch.equal(seen_value, value)
        assert lse.shape == (HEADS, 2, 5)
        torch.testing.assert_close(lse, expected_lse.reshape(HEADS, 2, 5), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("varlen", [False, True])
def test_without_an_observer_or_without_grad_the_log_partition_is_not_requested(monkeypatch, varlen):
    flash = _FlashStub(monkeypatch, "fa3")
    query, key, value = _qkv(2, 5)
    cu_seqlens = torch.tensor([0, 5, 10], dtype=torch.int32) if varlen else None
    attention = _MLA()
    attention._flash_attention_mla_forward(query, key, value, None, cu_seqlens, 5 if varlen else None)
    attention.attention_observer = observer = _Observer()

    with torch.no_grad():
        attention._flash_attention_mla_forward(query, key, value, None, cu_seqlens, 5 if varlen else None)

    assert flash.requested == [False, False]
    assert observer.calls == []


def test_a_padded_batch_is_not_observed(monkeypatch):
    flash = _FlashStub(monkeypatch, "fa2")
    monkeypatch.setattr(attention_module, "unpad_input", _unpad_input)
    monkeypatch.setattr(attention_module, "pad_input", _pad_input)
    query, key, value = _qkv(2, 5)
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    attention = _MLA()
    expected_output = attention._flash_attention_mla_forward(query, key, value, mask)
    attention.attention_observer = observer = _Observer()

    output = attention._flash_attention_mla_forward(query, key, value, mask)

    assert flash.requested == [False, False]
    assert observer.calls == []
    assert torch.equal(output, expected_output)


def _unpad_input(x, mask):
    indices = mask.flatten().nonzero().flatten()
    lengths = mask.sum(-1, dtype=torch.int32)
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    return x.flatten(0, 1)[indices], indices, cu_seqlens, int(lengths.max()), lengths


def _pad_input(x, indices, batch, seq):
    padded = torch.zeros(batch * seq, *x.shape[1:])
    padded[indices] = x
    return padded.reshape(batch, seq, *x.shape[1:])


def _forward(attention, hidden, **kwargs):
    # The undecorated forward, without the profiler annotation composer wraps around it.
    return inspect.unwrap(type(attention).forward)(attention, hidden, **kwargs)[0]


def _stand_in_forward_kernels(monkeypatch):
    monkeypatch.setattr(attention_module, "apply_rotary_emb_mla", lambda query, *args, **kwargs: query)
    monkeypatch.setattr(attention_module, "apply_rotary_emb", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(attention_module, "_apply_sigmoid_gate_fp32",
                        lambda output, gate: output * torch.sigmoid(gate.float()).to(output.dtype))


class _GatedMLA(_MLA):
    """Everything `forward` runs around the flash-attn call, on small projections of the hidden states."""

    def __init__(self, hidden: int, gated: bool) -> None:
        super().__init__()
        self.config = SimpleNamespace(rope_origin="custom")
        self.num_heads_per_rank = HEADS
        self.gated_attention = gated
        self.use_custom_rotary_kernel = True
        self.varlen_input = False
        self.is_ring_attn = False
        self.sp_size = self.tp_size = 1
        self.qk_nope_head_dim = NOPE
        self.projection = nn.Linear(hidden, HEADS * (NOPE + ROPE + NOPE + ROPE + VALUE), bias=False)
        self.gate_proj = nn.Linear(hidden, HEADS * VALUE, bias=False)
        self.o_proj = nn.Linear(HEADS * VALUE, hidden, bias=False)
        self.rotary_emb = lambda x, **kwargs: (torch.ones(1, 1), torch.ones(1, 1))

    def _compute_qkv(self, hidden_states, logical_batch_size=None):
        batch, seq, _ = hidden_states.shape
        shapes = [NOPE + ROPE, NOPE, ROPE, VALUE]
        query, k_nope, k_rope, value = (
            part.reshape(batch, seq, HEADS, -1)
            for part in self.projection(hidden_states).split([HEADS * size for size in shapes], dim=-1)
        )
        return query, k_nope, k_rope, value


@pytest.mark.parametrize("gated", [False, True])
def test_forward_observers_see_the_attention_and_gate_tensors_and_leave_the_output_unchanged(monkeypatch, gated):
    flash = _FlashStub(monkeypatch, "fa3")
    _stand_in_forward_kernels(monkeypatch)
    torch.manual_seed(0)
    attention = _GatedMLA(hidden=6, gated=gated)
    hidden = torch.randn(2, 5, 6, requires_grad=True)
    expected = _forward(attention, hidden)
    expected.sum().backward()
    expected_gradient, hidden.grad = hidden.grad, None
    attention.attention_observer, attention.gate_observer = observed, gate_observed = _Observer(), _Observer()

    output = _forward(attention, hidden)
    output.sum().backward()

    assert torch.equal(output, expected) and torch.equal(hidden.grad, expected_gradient)
    assert flash.requested == [False, True]
    assert len(observed.calls) == 1 and observed.calls[0][3].shape == (HEADS, 2, 5)
    if gated:
        ((gate,),) = gate_observed.calls
        assert torch.equal(gate, attention.gate_proj(hidden).reshape(2, 5, HEADS, VALUE))
    else:
        assert gate_observed.calls == []

    with torch.no_grad():
        _forward(attention, hidden)
    assert flash.requested[-1] is False and len(observed.calls) == 1 and len(gate_observed.calls) == int(gated)


class _OutputNorm(nn.Module):
    def forward(self, output, gate):
        return output * torch.sigmoid(gate)


class _GDN(Qwen3NextGatedDeltaNet):
    """`Qwen3NextGatedDeltaNet` with the projections of `forward` kept and its kernels stood in."""

    key_heads, value_heads, key_dim, value_dim, hidden = 2, 4, 3, 5, 6

    def __init__(self, allow_neg_eigval: bool) -> None:
        nn.Module.__init__(self)
        self.attention_observer = None
        self._actlog_prefix, self._actlog_layer_depends = "", False
        self._qkv_z_loss_coef = 0.0
        self.num_heads, self.num_v_heads = self.key_heads, self.value_heads
        self.head_dim, self.head_v_dim = self.key_dim, self.value_dim
        self.allow_neg_eigval = allow_neg_eigval
        self.use_legacy_qkvz_layout = False
        self.linear_sp_implementation = "fla_cp"
        self.linear_gating_type = "gated_rmsnorm"
        self.linear_sigmoid_gate_scale = 1.0
        self.output_mult = 1.0
        self.ba_proj = nn.Linear(self.hidden, 2 * self.value_heads, bias=False)
        self.qkv_proj = nn.Linear(self.hidden, 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim, bias=False)
        self.z_proj = nn.Linear(self.hidden, self.value_heads * self.value_dim, bias=False)
        self.qkv_conv1d = SimpleNamespace(weight=torch.zeros(1, 1, 1))
        self.A_log = nn.Parameter(torch.randn(self.value_heads))
        self.dt_bias = nn.Parameter(torch.randn(self.value_heads))
        self.o_norm = _OutputNorm()
        self.o_proj = nn.Linear(self.value_heads * self.value_dim, self.hidden, bias=False)


class _Kernels:
    """What the delta-rule path runs: the convolution unpack, FusedG and the chunked and recurrent kernels on
    torch. `requested` holds `output_final_state` of every kernel call, `states` the final state returned."""

    def __init__(self, monkeypatch, sp_size=None) -> None:
        self.seen: list[tuple[torch.Tensor, ...]] = []
        self.requested: list[bool] = []
        self.states: list[torch.Tensor] = []
        self.cp_contexts: list[object] = []
        monkeypatch.setattr(attention_module, "dist", SimpleNamespace(get_sp_group=lambda: None, get_sp_group_size=lambda: sp_size))
        monkeypatch.setattr(attention_module, "global_actlog_state", SimpleNamespace(use_monitor_variable=lambda *a, **k: None))
        monkeypatch.setattr(attention_module, "fused_qkv_conv1d_unpack", self.unpack)
        monkeypatch.setattr(attention_module, "FusedG", SimpleNamespace(apply=self.fused_g))
        monkeypatch.setattr(attention_module, "chunk_gated_delta_rule", self.chunk)
        monkeypatch.setattr(attention_module, "fused_recurrent_gated_delta_rule", self.recurrent)
        monkeypatch.setattr(attention_module, "build_cp_context", lambda **kwargs: "cp_context")

    @staticmethod
    def unpack(projected, weight, num_heads, num_v_heads, head_dim, head_v_dim, cp_context, cu_seqlens, normalize_qk, eps):
        q, k, v = projected.split([num_heads * head_dim, num_heads * head_dim, num_v_heads * head_v_dim], dim=-1)
        return q.reshape(*q.shape[:-1], num_heads, head_dim), k.reshape(*k.shape[:-1], num_heads, head_dim),             v.reshape(*v.shape[:-1], num_v_heads, head_v_dim)

    @staticmethod
    def fused_g(a, a_log, dt_bias):
        return -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())

    def chunk(self, q, k, v, g, beta, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens, cp_context):
        self.cp_contexts.append(cp_context)
        return self._run(q, k, v, g, beta, output_final_state, cu_seqlens)

    def recurrent(self, q, k, v, g, beta, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens):
        return self._run(q, k, v, g, beta, output_final_state, cu_seqlens)

    def _run(self, q, k, v, g, beta, output_final_state, cu_seqlens):
        output = v * beta.unsqueeze(-1) * g.exp().unsqueeze(-1) + q.sum() + k.sum()
        self.seen.append((beta, g, v, output))
        self.requested.append(output_final_state)
        documents = v.shape[0] if cu_seqlens is None else len(cu_seqlens) - 1
        state = torch.randn(documents, v.shape[2], q.shape[-1], v.shape[-1]) if output_final_state else None
        self.states.append(state)
        return output, state


@pytest.mark.parametrize("allow_neg_eigval", [False, True])
def test_the_delta_rule_observer_gets_the_kernel_operands_and_decay_parameters(monkeypatch, allow_neg_eigval):
    kernels = _Kernels(monkeypatch)
    torch.manual_seed(0)
    attention = _GDN(allow_neg_eigval)
    hidden = torch.randn(2, 70, _GDN.hidden, requires_grad=True)
    expected = _forward(attention, hidden)
    expected.sum().backward()
    expected_gradient, hidden.grad = hidden.grad, None
    observed = []
    attention.attention_observer = lambda *args: observed.append(args)

    output = _forward(attention, hidden)
    output.sum().backward()

    assert torch.equal(output, expected) and torch.equal(hidden.grad, expected_gradient)
    assert len(observed) == 1
    ((beta, g, value, kernel_output, a_log, dt_bias, final_state),) = observed
    kernel_beta, kernel_g, kernel_value, returned_output = kernels.seen[-1]
    assert beta is kernel_beta and g is kernel_g and kernel_output is returned_output
    assert torch.equal(value, kernel_value)
    assert a_log is attention.A_log and dt_bias is attention.dt_bias
    assert beta.shape == g.shape == (2, 70, _GDN.value_heads)
    assert value.shape == kernel_output.shape == (2, 70, _GDN.value_heads, _GDN.value_dim)
    _, expected_beta, _ = attention.fix_query_key_value_ordering(
        attention.z_proj(hidden), attention.ba_proj(hidden)
    )
    torch.testing.assert_close(beta, expected_beta.sigmoid() * (2.0 if allow_neg_eigval else 1.0))
    assert final_state is kernels.states[-1]

    with torch.no_grad():
        _forward(attention, hidden)
    assert len(observed) == 1


@pytest.mark.parametrize(("batch", "cu_seqlens"), [(2, None), (1, torch.tensor([0, 30, 70, 100, 140]))], ids=["batched", "packed"])
def test_the_final_state_of_every_document_is_requested_only_for_an_observer_with_grad(monkeypatch, batch, cu_seqlens):
    kernels = _Kernels(monkeypatch)
    torch.manual_seed(0)
    attention = _GDN(allow_neg_eigval=False)
    hidden = torch.randn(batch, 140 // batch, _GDN.hidden)
    observed = []
    expected = _forward(attention, hidden, cu_seqlens=cu_seqlens)
    attention.attention_observer = lambda *args: observed.append(args)

    output = _forward(attention, hidden, cu_seqlens=cu_seqlens)
    with torch.no_grad():
        _forward(attention, hidden, cu_seqlens=cu_seqlens)

    assert torch.equal(output, expected)
    assert kernels.requested == [False, True, False]
    assert kernels.states[0] is None and kernels.states[2] is None
    ((*_, final_state),) = observed
    assert final_state is kernels.states[1]
    assert final_state.shape == (2 if cu_seqlens is None else 4, _GDN.value_heads, _GDN.key_dim, _GDN.value_dim)


def test_the_recurrent_kernel_of_a_short_sequence_returns_the_final_state_too(monkeypatch):
    kernels = _Kernels(monkeypatch)
    torch.manual_seed(0)
    attention = _GDN(allow_neg_eigval=False).eval()
    hidden = torch.randn(2, 8, _GDN.hidden)
    observed = []
    attention.attention_observer = lambda *args: observed.append(args)

    _forward(attention, hidden)

    assert kernels.requested == [True]
    ((*_, final_state),) = observed
    assert final_state is kernels.states[0] and final_state.shape[0] == 2


def test_under_context_parallel_the_state_of_a_slice_is_neither_requested_nor_observed(monkeypatch):
    kernels = _Kernels(monkeypatch, sp_size=2)
    torch.manual_seed(0)
    attention = _GDN(allow_neg_eigval=False)
    attention.conv_size = 4
    hidden = torch.randn(2, 70, _GDN.hidden)
    expected = _forward(attention, hidden)
    observed = []
    attention.attention_observer = lambda *args: observed.append(args)

    output = _forward(attention, hidden)

    assert torch.equal(output, expected)
    assert kernels.cp_contexts == ["cp_context"] * 2
    assert kernels.requested == [False, False] and kernels.states == [None, None]
    ((*_, final_state),) = observed
    assert final_state is None


@pytest.mark.skipif(not flash_attn_import.HAS_FA3, reason="the FA3 varlen wrapper exists only on Hopper builds")
@pytest.mark.parametrize("return_attn_probs", [False, True])
def test_the_fa3_varlen_wrapper_exposes_lse_without_changing_the_public_call(monkeypatch, return_attn_probs):
    calls = []
    monkeypatch.setattr(
        flash_attn_import, "flash_attn_varlen_v3_func", lambda *args, **kwargs: calls.append(kwargs) or "result"
    )
    stats_calls = []
    monkeypatch.setattr(
        flash_attn_import._FlashAttnVarlenWithLSE, "apply",
        lambda *args: stats_calls.append(args) or ("result", "lse"),
    )

    result = flash_attn_import.flash_attn_varlen_func_v3_wrapper(
        "q", "k", "v", "cu", "cu", 8, 8, softmax_scale=0.5, causal=True, return_attn_probs=return_attn_probs,
    )

    if return_attn_probs:
        assert result == ("result", "lse")
        assert not calls
        assert stats_calls == [("q", "k", "v", "cu", "cu", 8, 8, 0.5, True, (-1, -1), False)]
    else:
        assert result == "result"
        assert not stats_calls
        assert calls == [dict(softmax_scale=0.5, causal=True, window_size=(-1, -1), deterministic=False)]
