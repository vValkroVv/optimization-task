from __future__ import annotations

import pytest
import torch

from metric_sampler.capture.stats.helpers.reductions import float64_scalar

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [
    torch.tensor(0.1), torch.tensor(0.1, dtype=torch.float64), torch.tensor([0.1]), torch.tensor([[0.1]]),
    torch.tensor(0.1, dtype=torch.bfloat16), torch.tensor(3), torch.tensor(0.1, requires_grad=True),
])
def test_scalar_on_the_population_device_is_a_float64_scalar_of_its_value(value):
    with torch.no_grad():
        scalar = float64_scalar(value, value.device)

    assert (scalar.dtype, scalar.shape, scalar.requires_grad) == (torch.float64, (), False)
    assert scalar.item() == value.detach().double().reshape(()).item()


def test_scalar_from_another_device_is_filled_on_the_population_device():
    value = float64_scalar(torch.tensor(3), torch.device('meta'))
    assert (value.device.type, value.dtype, value.shape) == ('meta', torch.float64, ())
