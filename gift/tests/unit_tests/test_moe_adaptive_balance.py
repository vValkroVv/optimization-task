"""Adaptive bias updates retain the baseline rule and optimizer-hook contract."""

import pytest
import torch

from torchtitan.config import load_document
from torchtitan.config.build import prepare_document
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.models.common.moe.load_balance import (
    expert_bias_delta,
    register_moe_load_balancing_hook,
)


@pytest.mark.parametrize(
    "loads,scale",
    [
        ([0.0, 60.0, 30.0, 30.0], 2.0),
        ([1600.0] + [0.0] * 9, 5.0),
        ([25.0] * 8, 1.0),
        ([0.0] * 8, 1.0),
    ],
)
def test_adaptive_matches_pinned_max_vio_rule(loads, scale):
    tokens = torch.tensor(loads)
    baseline = expert_bias_delta(tokens, coeff=1e-3, strategy="baseline")
    raw = 1e-3 * torch.sign(tokens.mean() - tokens)
    torch.testing.assert_close(baseline, raw - raw.mean(), rtol=0, atol=0)
    adaptive = expert_bias_delta(tokens, coeff=1e-3, strategy="adaptive")
    torch.testing.assert_close(adaptive, baseline * scale, rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(adaptive.mean(), torch.tensor(0.0), atol=1e-8, rtol=0)
    assert torch.isfinite(adaptive).all()


@pytest.mark.parametrize("strategy", ["baseline", "adaptive"])
def test_strategy_reaches_optimizer_update_and_preserves_checkpoint(strategy):
    document, _ = load_document("configs/experiments/qwen3_moe_debug.yaml")
    config = prepare_document(document).model_config.layers[0].moe
    assert config.load_balance_strategy == "baseline"
    config = config._copy(load_balance_strategy=strategy)
    moe = config.build()
    moe.init_states(buffer_device=torch.device("cpu"))
    checkpoint_keys = set(moe.state_dict())
    loads = torch.arange(moe.tokens_per_expert_E.numel(), dtype=torch.float32)
    moe.tokens_per_expert_E.copy_(loads)

    class Hooks:
        def register_step_pre_hook(self, function):
            self.function = function

    class Dims:
        full_dtensor = False

        def get_optional_mesh(self, name):
            return None

    hooks = Hooks()
    register_moe_load_balancing_hook(hooks, [moe], Dims())
    hooks.function()
    expected = expert_bias_delta(
        loads, coeff=config.load_balance_coeff, strategy=strategy
    )
    torch.testing.assert_close(moe.expert_bias_E, expected, rtol=0, atol=0)
    assert torch.count_nonzero(moe.tokens_per_expert_E) == 0
    assert set(moe.state_dict()) == checkpoint_keys
    hooks.function()
    torch.testing.assert_close(moe.expert_bias_E, expected, rtol=0, atol=0)


def test_invalid_strategy_rejected_at_config_boundary():
    document, _ = load_document("configs/experiments/qwen3_moe_debug.yaml")
    config = prepare_document(document).model_config.layers[0].moe
    with pytest.raises(ConfigSchemaError, match="load_balance_strategy"):
        config._copy(load_balance_strategy="adaptiv")
