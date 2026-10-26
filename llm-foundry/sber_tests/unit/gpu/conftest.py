import pytest

from .fixtures.imports import (  # noqa: F401
    dpo_modules,
    mla_modules,
    moe_modules,
    unified_module,
)


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    """Set the seed for the random number generators. Applied for all tests in folder."""
    import torch

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
