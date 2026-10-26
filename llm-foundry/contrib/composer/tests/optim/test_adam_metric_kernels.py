import math

import pytest
import torch

from composer.functional.triton_kernels import (
    adam_metric_sums, adam_metric_sums_batch, slice_chunks, slice_square_sums, slice_square_sums_batch,
)
from composer.optim import DecoupledAdamW

_STEP_SIZE = 1e-3 / (1 - 0.9**7)
_SQRT_BIAS_CORRECTION2 = math.sqrt(1 - 0.95**7)
_EPS = 1e-8


def _adam_state(numel: int, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device='cuda').manual_seed(seed)
    param = torch.randn(numel, device='cuda', generator=generator)
    grad = torch.randn(numel, device='cuda', generator=generator) * 1e-3
    exp_avg = torch.randn(numel, device='cuda', generator=generator) * 1e-4
    # Second moments reach the subnormal range.
    scale = 10**torch.empty(numel, device='cuda').uniform_(-44, -2, generator=generator)
    exp_avg_sq = torch.rand(numel, device='cuda', generator=generator) * scale
    return param, grad, exp_avg, exp_avg_sq


def _torch_step(param, exp_avg, exp_avg_sq, decay: float, cautious: bool) -> torch.Tensor:
    """The step as DecoupledAdamW.report_per_parameter_metrics computes it without the kernels."""
    denom = (exp_avg_sq.sqrt() / _SQRT_BIAS_CORRECTION2).add_(_EPS)
    step = _STEP_SIZE * exp_avg.div(denom)
    step.add_(torch.where(step * param >= 0, param, 0) if cautious else param, alpha=decay)
    return step


@pytest.mark.gpu
@pytest.mark.parametrize('decay', [0.0, -1e-4])
def test_slice_update_squares_equal_the_torch_update_element_by_element(decay: float):
    param, grad, exp_avg, exp_avg_sq = _adam_state(4099, seed=3)
    elements = [(index, index + 1) for index in range(param.numel())]

    sums = slice_square_sums(
        param, grad, exp_avg, exp_avg_sq, slice_chunks(elements, param.device), _STEP_SIZE, _SQRT_BIAS_CORRECTION2,
        _EPS, decay,
    )

    update = _torch_step(param, exp_avg, exp_avg_sq, decay, cautious=False)
    assert torch.equal(sums, torch.stack((param.square(), update.square(), grad.square()), dim=1).double())


@pytest.mark.gpu
@pytest.mark.parametrize('cautious', [False, True])
def test_adam_metric_step_equals_the_torch_step_element_by_element(cautious: bool):
    param, grad, exp_avg, exp_avg_sq = _adam_state(512, seed=5)
    step = _torch_step(param, exp_avg, exp_avg_sq, -1e-4, cautious)

    for index in range(param.numel()):
        element = slice(index, index + 1)
        sums, maxima = adam_metric_sums(
            param[element], grad[element], exp_avg[element], exp_avg_sq[element], _STEP_SIZE, _SQRT_BIAS_CORRECTION2,
            _EPS, -1e-4, cautious,
        )
        assert sums[3] == step[index].square().double()
        assert maxima.tolist() == [exp_avg[index].abs().item(), exp_avg_sq[index].item()]


@pytest.mark.gpu
@pytest.mark.parametrize('cautious', [False, True])
def test_adam_metric_sums_are_the_torch_sums_and_maxima(cautious: bool):
    param, grad, exp_avg, exp_avg_sq = _adam_state(3 * 65536 + 17, seed=7)

    sums, maxima = adam_metric_sums(
        param, grad, exp_avg, exp_avg_sq, _STEP_SIZE, _SQRT_BIAS_CORRECTION2, _EPS, -1e-4, cautious,
    )

    step = _torch_step(param, exp_avg, exp_avg_sq, -1e-4, cautious)
    expected = torch.stack([value.double().square().sum() for value in (exp_avg, exp_avg_sq, param, step, grad)])
    torch.testing.assert_close(sums, expected, rtol=1e-5, atol=0)
    assert torch.equal(maxima, torch.stack((exp_avg.abs().amax(), exp_avg_sq.amax())))


@pytest.mark.gpu
@pytest.mark.parametrize('nan_in', ['exp_avg', 'exp_avg_sq'])
def test_adam_metric_maxima_propagate_nan_like_torch_amax(nan_in: str):
    state = dict(zip(('param', 'grad', 'exp_avg', 'exp_avg_sq'), _adam_state(2 * 65536, seed=11)))
    state[nan_in][70000] = float('nan')

    _, maxima = adam_metric_sums(
        state['param'], state['grad'], state['exp_avg'], state['exp_avg_sq'], _STEP_SIZE, _SQRT_BIAS_CORRECTION2,
        _EPS, -1e-4, False,
    )

    expected = torch.stack((state['exp_avg'].abs().amax(), state['exp_avg_sq'].amax()))
    assert torch.equal(maxima.isnan(), expected.isnan())
    assert maxima.isnan().any()


@pytest.mark.gpu
@pytest.mark.parametrize('cautious', [False, True])
def test_a_batch_gives_each_parameter_what_it_gets_alone(cautious: bool):
    entries = [(*_adam_state(numel, seed=numel), _STEP_SIZE * (1 + index), _SQRT_BIAS_CORRECTION2, _EPS, -1e-4)
               for index, numel in enumerate((5, 65536, 3 * 65536 + 17, 1000))]

    sums, maxima = adam_metric_sums_batch(entries, cautious)

    for index, entry in enumerate(entries):
        alone_sums, alone_maxima = adam_metric_sums(*entry, cautious)
        assert torch.equal(sums[index], alone_sums)
        assert torch.equal(maxima[index], alone_maxima)


@pytest.mark.gpu
@pytest.mark.parametrize('with_state', [False, True])
def test_a_slice_batch_gives_each_shard_what_it_gets_alone(with_state: bool):
    shards = []
    for numel, bounds, keep_grad in ((70000, [(0, 3), (3, 69990)], True), (200, [(10, 200)], False),
                                     (300000, [(5, 131072), (131072, 300000)], True)):
        param, grad, exp_avg, exp_avg_sq = _adam_state(numel, seed=numel)
        shards.append((param, grad if keep_grad else None, exp_avg if with_state else None,
                       exp_avg_sq if with_state else None, slice_chunks(bounds, param.device), _STEP_SIZE,
                       _SQRT_BIAS_CORRECTION2, _EPS, -1e-4))

    sums = slice_square_sums_batch(shards)

    assert torch.equal(sums, torch.cat([slice_square_sums(*shard) for shard in shards]))


@pytest.mark.gpu
def test_reporting_all_parameters_at_once_gives_what_each_reports_alone():
    params = [torch.nn.Parameter(torch.randn(numel, device='cuda')) for numel in (7, 65537, 1000)]
    optimizer = DecoupledAdamW(params, lr=1e-3, weight_decay=1e-4)
    for index, param in enumerate(params):
        _, grad, exp_avg, exp_avg_sq = _adam_state(param.numel(), seed=index)
        param.grad = grad
        optimizer.state[param] = {'exp_avg': exp_avg, 'exp_avg_sq': exp_avg_sq, 'step': torch.tensor(3.0)}
    named = [(f'p{index}', param) for index, param in enumerate(params)]

    together = optimizer.report_metrics_for_parameters(named, {})
    alone = {}
    for name, param in named:
        optimizer.report_per_parameter_metrics(param, name, alone)

    assert together.keys() == alone.keys()
    for key, value in together.items():
        assert torch.equal(value, alone[key]), key
    # The bias-corrected moments are the moments divided by the correction, as torch divides a tensor by a scalar.
    beta1, beta2 = optimizer.param_groups[0]['betas']
    for name, _ in named:
        assert torch.equal(alone[f'l2_norm/moment_bc/{name}'], alone[f'l2_norm/moment/{name}'] / (1 - beta1**3))
        assert torch.equal(alone[f'l2_norm/moment_sq_bc/{name}'], alone[f'l2_norm/moment_sq/{name}'] / (1 - beta2**3))
