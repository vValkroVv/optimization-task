# Copyright 2024 LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `UpdateWeightRatioMonitor`.

The monitor reconstructs the step `DecoupledAdamW` actually applied from the
optimizer state, so the thing worth testing is that the reconstruction matches
a real optimizer step (bias correction, eps, per-param LR scale) rather than an
idealised `lr * m / sqrt(v)`. A drifting reconstruction would still log a
plausible-looking ~1e-3 ratio while being wrong, so it is checked numerically
against a hand-rolled AdamW update.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from llmfoundry.callbacks.update_weight_ratio_monitor import (  # noqa: E402
    UpdateWeightRatioMonitor,
)


class _State:
    def __init__(self, model, optimizer, batch=0):
        self.model = model
        self.optimizers = [optimizer]
        self.gigafsdp_enabled = False

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


def _model():
    torch.manual_seed(0)
    m = torch.nn.Sequential()
    m.add_module("lin", torch.nn.Linear(8, 4, bias=False))
    return m


def _stepped(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, n_steps=3, amsgrad=False):
    """A model+AdamW that has taken real steps, so optimizer state is populated."""
    model = _model()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=0.0,
        amsgrad=amsgrad,
    )
    for _ in range(n_steps):
        opt.zero_grad()
        model(torch.randn(16, 8)).square().mean().backward()
        opt.step()
    # leave a live .grad behind: the monitor skips params whose grad is None
    opt.zero_grad(set_to_none=False)
    model(torch.randn(16, 8)).square().mean().backward()
    return model, opt


@pytest.mark.unit
@pytest.mark.fast
def test_invalid_log_level_is_rejected():
    with pytest.raises(ValueError):
        UpdateWeightRatioMonitor(log_level="layer")


@pytest.mark.unit
@pytest.mark.fast
def test_invalid_batch_log_interval_is_rejected():
    with pytest.raises(ValueError):
        UpdateWeightRatioMonitor(batch_log_interval=0)


@pytest.mark.unit
@pytest.mark.fast
def test_reconstructed_update_matches_a_real_adamw_step():
    lr, betas, eps = 1e-3, (0.9, 0.999), 1e-8
    model, opt = _stepped(lr=lr, betas=betas, eps=eps)

    monitor = UpdateWeightRatioMonitor(batch_log_interval=1, log_level="fqn")
    logger = _RecordingLogger()
    state = _State(model, opt, batch=0)
    monitor.fit_start(state, logger)
    monitor.batch_end(state, logger)

    p = model.lin.weight
    st = opt.state[p]
    step = st["step"].item() if torch.is_tensor(st["step"]) else st["step"]
    beta1, beta2 = betas
    bc1 = 1 - beta1**step
    bc2 = 1 - beta2**step
    den = st["exp_avg_sq"].float().sqrt() / math.sqrt(bc2) + eps
    want_update = ((lr / bc1) * st["exp_avg"].float() / den).abs()

    w = p.data.float()
    want = {
        "update_abs_mean": float(want_update.mean()),
        "weight_std": float(
            math.sqrt(max(float(w.square().mean()) - float(w.mean()) ** 2, 0.0))
        ),
    }
    want["ratio"] = want["update_abs_mean"] / want["weight_std"]

    for stat, expected in want.items():
        got = logger.metrics[f"update_weight_ratio/{stat}/lin.weight"]
        assert got == pytest.approx(expected, rel=1e-6), stat


@pytest.mark.unit
@pytest.mark.fast
def test_effective_lr_scale_is_applied():
    model, opt = _stepped()
    baseline = _RecordingLogger()
    monitor = UpdateWeightRatioMonitor(batch_log_interval=1)
    state = _State(model, opt)
    monitor.fit_start(state, baseline)
    monitor.batch_end(state, baseline)

    # An optimizer exposing a per-param scale must have it folded into the step.
    opt.get_effective_lr_scale = lambda _p: 4.0
    scaled = _RecordingLogger()
    monitor.batch_end(state, scaled)

    key = "update_weight_ratio/update_abs_mean/lin.weight"
    assert scaled.metrics[key] == pytest.approx(4.0 * baseline.metrics[key], rel=1e-6)
    # weight std is untouched by the LR scale
    std_key = "update_weight_ratio/weight_std/lin.weight"
    assert scaled.metrics[std_key] == pytest.approx(baseline.metrics[std_key], rel=1e-9)


@pytest.mark.unit
@pytest.mark.fast
def test_per_fqn_lr_vector_is_applied_elementwise():
    model, opt = _stepped()
    p = model.lin.weight
    lr_vector = torch.linspace(0.25, 2.0, p.numel()).view_as(p)
    opt.get_per_fqn_lr_multiplier = lambda _p: lr_vector

    monitor = UpdateWeightRatioMonitor(batch_log_interval=1)
    logger = _RecordingLogger()
    state = _State(model, opt)
    monitor.fit_start(state, logger)
    monitor.batch_end(state, logger)

    st = opt.state[p]
    step = st["step"].item()
    beta1, beta2 = opt.param_groups[0]["betas"]
    bc1 = 1 - beta1**step
    bc2 = 1 - beta2**step
    den = st["exp_avg_sq"].float().sqrt() / math.sqrt(bc2) + opt.param_groups[0]["eps"]
    expected = (
        ((opt.param_groups[0]["lr"] / bc1) * lr_vector * st["exp_avg"].float() / den)
        .abs()
        .mean()
    )

    key = "update_weight_ratio/update_abs_mean/lin.weight"
    assert logger.metrics[key] == pytest.approx(float(expected), rel=1e-6)


@pytest.mark.unit
@pytest.mark.fast
def test_amsgrad_uses_max_second_moment():
    model, opt = _stepped(amsgrad=True)
    p = model.lin.weight
    monitor = UpdateWeightRatioMonitor(batch_log_interval=1)
    logger = _RecordingLogger()
    state = _State(model, opt)
    monitor.fit_start(state, logger)
    monitor.batch_end(state, logger)

    st = opt.state[p]
    step = st["step"].item()
    beta1, beta2 = opt.param_groups[0]["betas"]
    bc1 = 1 - beta1**step
    bc2 = 1 - beta2**step
    den = (
        st["max_exp_avg_sq"].float().sqrt() / math.sqrt(bc2)
        + opt.param_groups[0]["eps"]
    )
    expected = (
        ((opt.param_groups[0]["lr"] / bc1) * st["exp_avg"].float() / den).abs().mean()
    )

    key = "update_weight_ratio/update_abs_mean/lin.weight"
    assert logger.metrics[key] == pytest.approx(float(expected), rel=1e-6)


@pytest.mark.unit
@pytest.mark.fast
def test_group_log_level_aggregates_instead_of_per_fqn():
    model, opt = _stepped()
    monitor = UpdateWeightRatioMonitor(batch_log_interval=1, log_level="group")
    logger = _RecordingLogger()
    state = _State(model, opt)
    monitor.fit_start(state, logger)
    monitor.batch_end(state, logger)

    assert logger.metrics, "group level must still emit metrics"
    assert not any(k.endswith("/lin.weight") for k in logger.metrics)
    assert all(k.startswith("update_weight_ratio/") for k in logger.metrics)


@pytest.mark.unit
@pytest.mark.fast
def test_only_logs_on_the_interval():
    model, opt = _stepped()
    monitor = UpdateWeightRatioMonitor(batch_log_interval=100)
    logger = _RecordingLogger()
    monitor.fit_start(_State(model, opt), logger)

    monitor.batch_end(_State(model, opt, batch=37), logger)
    assert logger.metrics == {}
    monitor.batch_end(_State(model, opt, batch=100), logger)
    assert logger.metrics != {}


@pytest.mark.unit
@pytest.mark.fast
def test_params_without_optimizer_state_report_no_update():
    """A param that has never been stepped has no exp_avg -> update stays 0."""
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(4, 8)).square().mean().backward()  # grad but no opt.step()

    monitor = UpdateWeightRatioMonitor(batch_log_interval=1)
    logger = _RecordingLogger()
    state = _State(model, opt)
    monitor.fit_start(state, logger)
    monitor.batch_end(state, logger)

    assert logger.metrics["update_weight_ratio/update_abs_mean/lin.weight"] == 0.0
    assert logger.metrics["update_weight_ratio/weight_std/lin.weight"] > 0.0
