# Copyright 2024 LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `RouterMarginMonitor`.

The monitor's value is the *number* it reports: a per-layer top-k selection
margin whose ratio across layers is used directly as a `bias_update_rate_scales`
suggestion. So these tests check the arithmetic against hand-computed margins on
crafted router logits, plus the gate-discovery / disable paths and the reported
metric names.
"""

import pytest

torch = pytest.importorskip("torch")

from llmfoundry.callbacks.router_margin_monitor import (  # noqa: E402
    _SCORE_MEDIAN_GAP_IDX,
    _STAT_NAMES,
    RouterMarginMonitor,
)


class _Gate(torch.nn.Module):
    """Minimal stand-in for `MoEGate`: `.gate` is the linear producing logits."""

    def __init__(self, n_experts: int, top_k: int, scoring_func: str = "sigmoid"):
        super().__init__()
        self.gate = torch.nn.Linear(4, n_experts, bias=False)
        self.n_routed_experts = n_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.tp_size = 1
        self.group_topk = None
        self._is_recompute = False


class _Attn(torch.nn.Module):
    def __init__(self, gate, is_gate=True):
        super().__init__()
        self.is_gate = is_gate
        self._gate = gate


class _Layer(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


class _Inner(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class _Wrapper(torch.nn.Module):
    """Mirrors `state.model.model.model` -> the decoder stack."""

    def __init__(self, inner):
        super().__init__()

        class _Mid(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m

        self.model = _Mid(inner)


class _State:
    def __init__(self, model, batch=0):
        self.model = model

        class _T:
            def __init__(self, v):
                class _B:
                    def __init__(self, v):
                        self.value = v

                self.batch = _B(v)

        self.timestamp = _T(batch)


class _RecordingLogger:
    def __init__(self):
        self.metrics = {}

    def log_metrics(self, metrics):
        self.metrics.update(metrics)


def _build(n_experts=8, top_k=2, n_layers=2, scoring_func="sigmoid"):
    gates = [_Gate(n_experts, top_k, scoring_func) for _ in range(n_layers)]
    inner = _Inner([_Layer(_Attn(g)) for g in gates])
    return _Wrapper(inner), gates


@pytest.mark.unit
@pytest.mark.fast
def test_stats_match_hand_computed_margins():
    monitor = RouterMarginMonitor(batch_log_interval=1, near_m=2)
    gate = _Gate(n_experts=5, top_k=2, scoring_func="sigmoid")
    # One token, descending logits -> boundary is the 2nd largest (=3.0).
    logits = torch.tensor([[5.0, 3.0, 1.0, 0.0, -1.0]])

    vec = monitor._stats(logits, k=2, n_experts=5, scoring_func="sigmoid")

    assert vec.shape == (len(_STAT_NAMES),)
    lv = torch.tensor([5.0, 3.0, 1.0, 0.0, -1.0])
    sv = torch.sigmoid(lv)
    # logit space: gap = v[k-1] - v[k] = 3.0 - 1.0
    logit_gap = _STAT_NAMES.index("logit/median_gap")
    assert vec[logit_gap] == pytest.approx(2.0)
    # near_m=2 -> median over boundary - {v[2], v[3]} = {2.0, 3.0}. torch.median
    # returns the LOWER of the two middle values (not their average) -> 2.0.
    logit_near = _STAT_NAMES.index("logit/median_near")
    assert vec[logit_near] == pytest.approx(2.0)
    # score space uses sigmoid of the same values
    score_gap = _STAT_NAMES.index("score/median_gap")
    assert vec[score_gap] == pytest.approx(float(sv[1] - sv[2]), abs=1e-6)
    assert _SCORE_MEDIAN_GAP_IDX == score_gap


@pytest.mark.unit
@pytest.mark.fast
def test_stats_returns_none_when_nothing_is_unselected():
    monitor = RouterMarginMonitor()
    assert monitor._stats(torch.randn(3, 4), k=4, n_experts=4, scoring_func="sigmoid") is None


@pytest.mark.unit
@pytest.mark.fast
def test_stats_returns_none_for_unknown_scoring_func():
    monitor = RouterMarginMonitor()
    assert monitor._stats(torch.randn(3, 8), k=2, n_experts=8, scoring_func="relu") is None


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize(
    "kwargs", [{"batch_log_interval": 0}, {"near_m": 0}, {"ema_decay": 1.0}, {"ema_decay": -0.1}]
)
def test_invalid_args_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RouterMarginMonitor(**kwargs)


@pytest.mark.unit
@pytest.mark.fast
def test_disabled_without_gates():
    monitor = RouterMarginMonitor()
    inner = _Inner([_Layer(_Attn(gate=None, is_gate=False))])
    monitor.fit_start(_State(_Wrapper(inner)), _RecordingLogger())
    assert monitor._enabled is False
    assert monitor._handles == []


@pytest.mark.unit
@pytest.mark.fast
def test_disabled_under_tensor_parallel():
    model, gates = _build()
    gates[0].tp_size = 2
    monitor = RouterMarginMonitor()
    monitor.fit_start(_State(model), _RecordingLogger())
    assert monitor._enabled is False


@pytest.mark.unit
@pytest.mark.fast
def test_reports_per_layer_metrics_and_bias_rate_scale_ratio():
    model, gates = _build(n_experts=8, top_k=2, n_layers=2)
    monitor = RouterMarginMonitor(batch_log_interval=1, ema_decay=0.0)
    logger = _RecordingLogger()
    monitor.fit_start(_State(model), logger)
    assert monitor._enabled is True

    # The suggested scale is driven by the SCORE-space margin (that is the space
    # `expert_bias` lives in), so give the two layers deliberately different
    # score gaps and assert the reported scales follow them. `logits1` is scaled
    # far into sigmoid saturation, which *shrinks* its score gap even though its
    # logit gap is 10x wider -- exactly the distinction the monitor exists for.
    logits0 = torch.tensor([5.0, 3.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    logits1 = 10.0 * logits0
    with torch.no_grad():
        for gate, w in zip(gates, (logits0, logits1)):
            gate.gate.bias = None
            gate.gate.weight.data.zero_()
            gate.gate.weight[:, 0] = w

    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    expected = [
        float(monitor._stats(w.unsqueeze(0), k=2, n_experts=8, scoring_func="sigmoid")[
            _SCORE_MEDIAN_GAP_IDX
        ])
        for w in (logits0, logits1)
    ]
    state = _State(model, batch=0)
    monitor.batch_start(state, logger)          # batch 0 -> (0+1) % 1 == 0 -> active
    for g in gates:
        g.train()
        g.gate(x)
    monitor.batch_end(state, logger)

    for tag in ("layer0", "layer1"):
        for name in _STAT_NAMES:
            assert f"router_margin/{name}/{tag}" in logger.metrics
        assert f"router_margin/score/median_gap_ema/{tag}" in logger.metrics
        assert f"router_bias_rate_scale/{tag}" in logger.metrics
    assert "router_margin/score/median_gap_ema/mean" in logger.metrics

    for tag, want in zip(("layer0", "layer1"), expected):
        assert logger.metrics[f"router_margin/score/median_gap/{tag}"] == pytest.approx(
            want, abs=1e-6
        )

    scales = [logger.metrics["router_bias_rate_scale/layer0"],
              logger.metrics["router_bias_rate_scale/layer1"]]
    assert sum(scales) == pytest.approx(2.0)          # ratios around a mean of 1
    # scales follow the score-space margins, not the logit-space ones
    assert (scales[0] > scales[1]) == (expected[0] > expected[1])
    mean = sum(expected) / 2
    assert scales[0] == pytest.approx(expected[0] / mean, rel=1e-5)


@pytest.mark.unit
@pytest.mark.fast
def test_inactive_batches_log_nothing():
    model, gates = _build()
    monitor = RouterMarginMonitor(batch_log_interval=25)
    logger = _RecordingLogger()
    monitor.fit_start(_State(model), logger)

    state = _State(model, batch=0)  # (0+1) % 25 != 0
    monitor.batch_start(state, logger)
    assert monitor._active is False
    for g in gates:
        g.train()
        g.gate(torch.randn(2, 4))
    monitor.batch_end(state, logger)
    assert logger.metrics == {}


@pytest.mark.unit
@pytest.mark.fast
def test_recompute_forward_is_not_double_counted():
    model, gates = _build(n_layers=1)
    monitor = RouterMarginMonitor(batch_log_interval=1)
    logger = _RecordingLogger()
    monitor.fit_start(_State(model), logger)

    state = _State(model, batch=0)
    monitor.batch_start(state, logger)
    gates[0].train()
    gates[0].gate(torch.randn(2, 4))
    assert len(monitor._acc["layer0"]) == 1
    gates[0]._is_recompute = True
    gates[0].gate(torch.randn(2, 4))
    assert len(monitor._acc["layer0"]) == 1


@pytest.mark.unit
@pytest.mark.fast
def test_close_removes_hooks():
    model, gates = _build(n_layers=2)
    monitor = RouterMarginMonitor()
    logger = _RecordingLogger()
    monitor.fit_start(_State(model), logger)
    assert len(monitor._handles) == 2
    monitor.close(_State(model), logger)
    assert monitor._handles == []
    # no hook left to append into the accumulator
    monitor._active = True
    gates[0].train()
    gates[0].gate(torch.randn(2, 4))
    assert monitor._acc["layer0"] == []
