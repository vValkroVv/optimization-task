import pytest
import torch

from stand.activation_kernel_checks.populations import selection_populations

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def populations() -> dict[str, torch.Tensor]:
    return dict(selection_populations(CPU))
