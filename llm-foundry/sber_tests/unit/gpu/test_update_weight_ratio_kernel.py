"""The CUDA pass of `UpdateWeightRatioMonitor` against the torch expressions it replaces: the
reconstructed update of every element bit for bit, and NaN weights propagating to the absolute max."""

import math
from types import SimpleNamespace

import pytest
import torch

from llmfoundry.callbacks.update_weight_ratio_monitor import UpdateWeightRatioMonitor

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def _observed(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-parameter [numel, update_abs_sum, w_sum, w_sq_sum] rows and [amax, l2] the monitor
    hands its rank-local observer on a step it does not log."""
    state = SimpleNamespace(model=model, optimizers=[optimizer], gigafsdp_enabled=False,
                            timestamp=SimpleNamespace(batch=SimpleNamespace(value=1)))
    monitor = UpdateWeightRatioMonitor(batch_log_interval=100)
    observed = []
    monitor.rank_local_observer = lambda names, rows, weights: observed.append((rows.clone(), weights.clone()))
    monitor.fit_start(state, None)
    monitor.batch_end(state, None)
    return observed[0]


def _optimizer(model: torch.nn.Module, step: float) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    generator = torch.Generator(device="cuda").manual_seed(1)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
        # Second moments reach the subnormal range.
        scale = 10 ** torch.empty_like(parameter).uniform_(-44, -2, generator=generator)
        optimizer.state[parameter] = {
            "exp_avg": torch.randn(parameter.shape, device="cuda", generator=generator) * 1e-4,
            "exp_avg_sq": torch.rand(parameter.shape, device="cuda", generator=generator) * scale,
            "step": torch.tensor(step),
        }
    return optimizer


def test_update_of_every_element_is_the_torch_update():
    model = torch.nn.Module()
    for index in range(256):
        model.register_parameter(f"w{index:03d}", torch.nn.Parameter(torch.randn(1, device="cuda")))
    optimizer = _optimizer(model, step=7.0)

    rows, weights = _observed(model, optimizer)

    lr, eps, (beta1, beta2) = 1e-3, 1e-8, (0.9, 0.95)
    m = torch.cat([optimizer.state[p]["exp_avg"] for p in model.parameters()])
    v = torch.cat([optimizer.state[p]["exp_avg_sq"] for p in model.parameters()])
    w = torch.cat([p.data for p in model.parameters()])
    den = v.float().sqrt() / math.sqrt(1 - beta2**7) + eps
    update = (lr / (1 - beta1**7) * m.float() / den).abs()
    assert torch.equal(rows, torch.stack((torch.ones_like(w), update, w, w.square()), dim=1).double())
    assert torch.equal(weights[:, 0], w.abs())


def test_nan_weights_propagate_to_the_absolute_max():
    model = torch.nn.Module()
    model.register_parameter("large", torch.nn.Parameter(torch.randn(200_000, device="cuda")))
    model.register_parameter("small", torch.nn.Parameter(torch.randn(3, device="cuda")))
    model.register_parameter("finite", torch.nn.Parameter(torch.randn(200_000, device="cuda")))
    with torch.no_grad():
        model.large[150_000] = float("nan")
        model.small[1] = float("nan")

    _, weights = _observed(model, _optimizer(model, step=3.0))

    assert weights[:, 0].isnan().tolist() == [False, True, True]
    assert weights[0, 0] == model.finite.abs().amax()
