"""The first-group debug routing sends every token to experts ``0..top_k-1``."""

import pytest
import torch

from tests.unit_tests.config_helpers import load_experiment_config
from torchtitan.models.common.decoder_blocks.decoder import Decoder
from torchtitan.models.common.moe.router import TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import Linear


def _router(**flags) -> TokenChoiceTopKRouter:
    config = TokenChoiceTopKRouter.Config(
        gate=Linear.Config(in_features=8, out_features=6, bias=False),
        top_k=2,
        **flags,
    )
    return config.build()


@pytest.mark.parametrize("route_norm", [False, True])
@pytest.mark.parametrize("route_scale", [0.5, 2.0])
def test_first_group_routing_ids_and_scores(route_norm, route_scale):
    router = _router(
        _debug_force_first_group=True, route_norm=route_norm, route_scale=route_scale
    )
    x = torch.randn(3, 5, 8)
    topk_scores, topk_ids, scores, _ = router(x)
    assert topk_ids.shape == (3, 5, 2)
    assert torch.equal(topk_ids, torch.tensor([0, 1]).expand(3, 5, 2))
    torch.testing.assert_close(
        topk_scores, torch.full_like(topk_scores, 0.5), rtol=0, atol=0
    )
    assert not topk_scores.requires_grad
    # Auxiliary balancing losses can still use the original gate scores.
    assert scores.requires_grad


def test_debug_routings_are_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _router(_debug_force_first_group=True, _debug_force_load_balance=True)


@pytest.mark.parametrize(
    "experiment",
    [
        "gift_moe_debug.yaml",
        "deepseek_v3_debug.yaml",
        "llama4_debug.yaml",
        "qwen3_moe_debug.yaml",
        "gpt_oss_debug.yaml",
    ],
)
@pytest.mark.parametrize("ep_size,valid", [(2, True), (4, False)])
def test_first_group_rejects_multi_receiver_training_layout(experiment, ep_size, valid):
    from torchtitan.models.common.moe import MoE
    from torchtitan.models.common.moe.dispatch import AllToAllTokenDispatcher

    config = load_experiment_config(experiment)
    config.debug.moe_force_first_group = True
    config.parallelism.expert_parallel_degree = ep_size
    moes = [cfg for _, cfg, *_ in config.model_spec.model.traverse(MoE.Config)]
    assert moes
    # Reproduce the review's E=8, K=3, EP=4 example on every shared decoder.
    for moe in moes:
        moe.router.gate.out_features = 8
        moe.router.top_k = 3
        moe.experts.num_experts = 8
        moe.experts.token_dispatcher = AllToAllTokenDispatcher.Config()
    if valid:
        Decoder.prepare_config(config.model_spec.model, config=config)
        assert all(moe.router._debug_force_first_group for moe in moes)
    else:
        with pytest.raises(
            ValueError, match="debug.moe_force_first_group.*top_k <= num_experts"
        ):
            Decoder.prepare_config(config.model_spec.model, config=config)
