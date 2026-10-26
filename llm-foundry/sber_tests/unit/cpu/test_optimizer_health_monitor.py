import math
from types import SimpleNamespace

import torch
from torch import nn

import llmfoundry.callbacks.optimizer_health_monitor as monitor_module
from llmfoundry.callbacks.optimizer_health_monitor import OptimizerHealthMonitor


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 1, bias=False)
        self.q_proj.weight.data.copy_(torch.tensor([[1.0, 2.0]]))
        self.q_proj.weight.grad = torch.ones_like(self.q_proj.weight)


def _state():
    model = _Model()
    optimizer = torch.optim.Adam(model.parameters(), eps=0.1, weight_decay=0.25)
    p = model.q_proj.weight
    optimizer.state[p] = {
        "exp_avg": torch.tensor([[0.2, 0.4]]),
        "exp_avg_sq": torch.tensor([[0.01, 0.25]]),
    }
    return SimpleNamespace(
        model=model,
        optimizers=[optimizer],
        timestamp=SimpleNamespace(batch=SimpleNamespace(value=1)),
        gigafsdp_enabled=False,
    )


class _Logger:
    def __init__(self):
        self.metrics = None

    def log_metrics(self, metrics):
        self.metrics = metrics


def _legacy_expected_metrics():
    v = torch.tensor([[0.01, 0.25]])
    m = torch.tensor([[0.2, 0.4]])
    p = torch.tensor([[1.0, 2.0]])
    eps = 0.1
    sqrt_v = v.float().sqrt()
    update = m.float().abs() / (sqrt_v + eps)
    frac_eps = (sqrt_v < eps).sum().item() / p.numel()
    frac_10eps = (sqrt_v < 10 * eps).sum().item() / p.numel()
    grad_l2 = math.sqrt(update.square().sum().item())
    wd_l2 = math.sqrt((0.25**2) * p.float().square().sum().item())
    return {
        "optim_health/frac_eps/attn_q": frac_eps,
        "optim_health/frac_10eps/attn_q": frac_10eps,
        "optim_health/update_mean/attn_q": update.sum().item() / p.numel(),
        "optim_health/Q_wd_over_grad/attn_q": wd_l2 / grad_l2,
    }


def test_ordinary_monitor_keeps_original_per_parameter_reductions(monkeypatch):
    state = _state()
    callback = OptimizerHealthMonitor(batch_log_interval=1)
    logger = _Logger()
    callback.fit_start(state, logger)
    monkeypatch.setattr(monitor_module.dist, "get_fsdp_group", lambda: None)
    monkeypatch.setattr(
        monitor_module.dist, "all_reduce", lambda tensor, **kwargs: tensor
    )

    def unexpected_fused_path(*args, **kwargs):
        raise AssertionError(
            "ordinary monitoring must not use the sampler fused helper"
        )

    monkeypatch.setattr(monitor_module, "_health_sums", unexpected_fused_path)
    callback.batch_end(state, logger)

    assert logger.metrics == _legacy_expected_metrics()


def test_observer_capture_uses_fused_sums_and_reports_rank_local_values(monkeypatch):
    state = _state()
    callback = OptimizerHealthMonitor(batch_log_interval=10)
    logger = _Logger()
    callback.fit_start(state, logger)
    observed = []
    callback.rank_local_observer = observed.append
    monkeypatch.setattr(monitor_module.dist, "get_fsdp_group", lambda: None)
    monkeypatch.setattr(
        monitor_module.dist, "all_reduce", lambda tensor, **kwargs: tensor
    )

    calls = []
    original = monitor_module._health_sums

    def track_fused_path(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(monitor_module, "_health_sums", track_fused_path)
    callback.batch_end(state, logger)

    assert calls
    assert len(observed) == 1
    assert observed[0].dtype == torch.float64
    assert observed[0].reshape(len(monitor_module.PARAM_GROUPS), 6)[1, 0].item() == 2
    assert logger.metrics is None
