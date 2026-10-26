"""Capacity thresholds follow the actual local MoE boundary, including PP/TP/CP."""

from dataclasses import asdict
from types import SimpleNamespace

from tests.unit_tests.config_helpers import load_experiment_config
from torchtitan.models.common.decoder_blocks.decoder import Decoder

import pytest
import torch

from torchtitan.config.moe import dispatcher_peak_tokens, TokenDropConfig
from torchtitan.models.common.moe.dispatch import (
    BaseTokenDispatcher,
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
    LocalTokenDispatcher,
)


@pytest.mark.parametrize(
    "pp,tp,cp,ep,expected",
    [
        (1, 1, 1, 2, 32768),
        (2, 1, 1, 2, 4096),
        (2, 2, 1, 2, 2048),
        (2, 2, 2, 2, 1024),
        (1, 2, 1, 1, 32768),
        (2, 2, 2, 1, 2048),
    ],
)
def test_peak_tokens_match_boundary(pp, tp, cp, ep, expected):
    assert (
        dispatcher_peak_tokens(
            local_batch_size=8,
            seq_len=4096,
            pipeline_parallel_degree=pp,
            pipeline_parallel_microbatch_size=1,
            tensor_parallel_degree=tp,
            context_parallel_degree=cp,
            expert_parallel_degree=ep,
        )
        == expected
    )


@pytest.mark.parametrize(
    "backend",
    [
        LocalTokenDispatcher,
        AllToAllTokenDispatcher,
        DeepEPTokenDispatcher,
        HybridEPTokenDispatcher,
    ],
)
def test_peak_quota_does_not_mutate_shared_config(backend):
    policy = TokenDropConfig(
        strategy="peak_capacity_factor", expert_capacity_factor=1.25
    )
    config = backend.Config(token_drop=policy)
    resolved = config.with_capacity_context(
        num_tokens=7, top_k=2, num_experts=4, ep_size=2
    )
    assert config._capacity_num_tokens is None
    assert resolved._capacity_num_tokens == 7
    assert resolved.token_drop is policy
    assert resolved.to_dict() == config.to_dict()
    assert TokenDropConfig(**asdict(policy)) == policy
    assert "peak_local_tokens" not in asdict(policy)
    assert (
        policy.source_capacity(
            num_tokens=3,
            top_k=2,
            num_experts=4,
            ep_size=2,
            max_num_tokens=resolved._capacity_num_tokens,
        )
        == 4
    )
    # Run each backend's shared selector without loading optional CUDA transport.
    dispatcher = object.__new__(backend)
    BaseTokenDispatcher.__init__(dispatcher, resolved)
    scores = torch.ones(3, 2)
    ids = torch.tensor([0, 1]).expand(3, 2)
    _, _, keep = dispatcher._drop_routes(
        scores, ids, num_experts=4, ep_group=SimpleNamespace(size=lambda: 2)
    )
    assert keep.all()  # quota 4 from the bound; current-input sizing would give 1
    with pytest.raises(ValueError, match="exceeding the configured peak"):
        dispatcher._drop_routes(
            torch.ones(8, 2),
            torch.tensor([0, 1]).expand(8, 2),
            num_experts=4,
            ep_group=SimpleNamespace(size=lambda: 2),
        )


@pytest.mark.parametrize(
    "backend",
    [
        LocalTokenDispatcher,
        AllToAllTokenDispatcher,
        DeepEPTokenDispatcher,
        HybridEPTokenDispatcher,
    ],
)
@pytest.mark.parametrize("branchless", [False, True])
def test_training_layout_resolves_capacity_for_every_backend(backend, branchless):
    from torchtitan.models.common.moe import MoE

    config = load_experiment_config("gift_moe_debug.yaml")
    config.model_spec.model.mtp = None
    config.loss.loss_weights = []
    config.training.local_batch_size = 2
    config.training.seq_len = 512
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = (
        1 if backend is LocalTokenDispatcher else 2
    )
    policy = TokenDropConfig(
        expert_capacity_factor=0.5, branchless_capacity_drop=branchless
    )
    kwargs = (
        dict(non_blocking_capacity_factor=0.25, pad_multiple=1, fuse_permute=False)
        if backend is HybridEPTokenDispatcher
        else {}
    )
    moes = [cfg for _, cfg, *_ in config.model_spec.model.traverse(MoE.Config)]
    assert moes
    for moe in moes:
        moe.experts.token_dispatcher = backend.Config(token_drop=policy, **kwargs)
    Decoder.prepare_config(config.model_spec.model, config=config)
    for moe in moes:
        dispatcher = moe.experts.token_dispatcher
        assert isinstance(dispatcher, backend.Config)
        assert dispatcher.token_drop is policy
        assert dispatcher._capacity_num_tokens == 1024


def test_explicit_fixed_quota_is_preserved_across_layouts():
    config = LocalTokenDispatcher.Config(
        token_drop=TokenDropConfig(
            strategy="fixed_capacity",
            expert_capacity_tokens=7,
        )
    )
    for n in (4, 8, 1024):
        resolved = config.with_capacity_context(
            num_tokens=n, top_k=2, num_experts=4, ep_size=2
        )
        assert (
            resolved.token_drop.source_capacity(
                num_tokens=n,
                top_k=2,
                num_experts=4,
                ep_size=2,
                max_num_tokens=resolved._capacity_num_tokens,
            )
            == 3
        )
        assert resolved.token_drop is config.token_drop


def test_peak_is_not_a_user_config_parameter():
    with pytest.raises(TypeError, match="peak_local_tokens"):
        TokenDropConfig(
            **dict(
                strategy="peak_capacity_factor",
                expert_capacity_factor=1,
                peak_local_tokens=32768,
            )
        )


@pytest.mark.parametrize("bound", [0, -1, 1.5, True])
def test_peak_capacity_rejects_invalid_runtime_bound(bound):
    policy = TokenDropConfig(strategy="peak_capacity_factor", expert_capacity_factor=1)
    with pytest.raises(ValueError, match="positive integer"):
        policy.source_capacity(
            num_tokens=1, top_k=2, num_experts=4, ep_size=2, max_num_tokens=bound
        )


def test_peak_refuses_unresolved_layout():
    policy = TokenDropConfig(strategy="peak_capacity_factor", expert_capacity_factor=1)
    with pytest.raises(ValueError, match="training layout"):
        policy.source_capacity(num_tokens=4096, top_k=4, num_experts=32, ep_size=2)


@pytest.mark.parametrize(
    "bad", [{"seq_len": 7}, {"pipeline_parallel_microbatch_size": 3}]
)
def test_layout_rejects_inexact_shards_or_pipeline_batches(bad):
    kwargs = dict(
        local_batch_size=8,
        seq_len=4096,
        pipeline_parallel_degree=2,
        pipeline_parallel_microbatch_size=1,
        tensor_parallel_degree=2,
        context_parallel_degree=1,
        expert_parallel_degree=2,
    )
    kwargs.update(bad)
    with pytest.raises(ValueError):
        dispatcher_peak_tokens(**kwargs)


def test_default_matches_foundry_conditional_path():
    assert not TokenDropConfig(expert_capacity_factor=1).branchless_capacity_drop


def test_derived_peak_rebinds_without_mutating_policy():
    config = LocalTokenDispatcher.Config(
        token_drop=TokenDropConfig(
            strategy="peak_capacity_factor",
            expert_capacity_factor=1,
        )
    )
    for n in (8, 16, 4):
        config = config.with_capacity_context(
            num_tokens=n, top_k=2, num_experts=4, ep_size=2
        )
        assert config._capacity_num_tokens == n
        assert (
            config.token_drop.source_capacity(
                num_tokens=1,
                top_k=2,
                num_experts=4,
                ep_size=2,
                max_num_tokens=config._capacity_num_tokens,
            )
            == n // 2
        )
        assert "peak_local_tokens" not in config.to_dict()["token_drop"]


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
def test_shared_decoder_resolves_every_moe_model(experiment):
    from torchtitan.models.common.moe import MoE

    config = load_experiment_config(experiment)
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_microbatch_size = 1
    config.training.local_batch_size = 4
    config.training.seq_len = 512
    moes = [cfg for _, cfg, *_ in config.model_spec.model.traverse(MoE.Config)]
    assert moes
    for moe in moes:
        moe.experts.token_dispatcher.token_drop = TokenDropConfig(
            strategy="peak_capacity_factor", expert_capacity_factor=1
        )
    Decoder.prepare_config(config.model_spec.model, config=config)
    for moe in moes:
        assert moe.experts.token_dispatcher._capacity_num_tokens == 512
        assert (
            "peak_local_tokens"
            not in moe.experts.token_dispatcher.to_dict()["token_drop"]
        )


@pytest.mark.parametrize("branchless", [False, True])
def test_hybridep_smoke_presets_resolve_safe_capacity(branchless):
    recipe = (
        "gift_moe_debug_hybridep_token_drop_branchless.yaml"
        if branchless
        else "gift_moe_debug_hybridep_token_drop.yaml"
    )
    config = load_experiment_config(recipe)
    Decoder.prepare_config(config.model_spec.model, config=config)
    dispatchers = [
        cfg
        for _, cfg, *_ in config.model_spec.model.traverse(
            HybridEPTokenDispatcher.Config
        )
    ]
    assert dispatchers
    for cfg in dispatchers:
        assert cfg.token_drop.branchless_capacity_drop == branchless
        assert cfg.non_blocking_capacity_factor == 0.25
        assert not cfg.fuse_permute
        assert (
            cfg._capacity_num_tokens
            == config.training.local_batch_size * config.training.seq_len
        )


@pytest.mark.parametrize(
    "policy,quota,peak",
    [
        (TokenDropConfig(expert_capacity_factor=1.25), 60, "60"),
        (
            TokenDropConfig(
                strategy="peak_capacity_factor", expert_capacity_factor=1.25
            ),
            60,
            "60",
        ),
        (
            TokenDropConfig(strategy="fixed_capacity", expert_capacity_tokens=113),
            56,
            "n/a (explicit quota)",
        ),
    ],
)
def test_startup_logs_absolute_capacity_and_unsharded_comparison(
    caplog, policy, quota, peak
):
    import logging
    from torchtitan.models.common.moe import MoE

    config = load_experiment_config("gift_moe_debug.yaml")
    config.model_spec.model.mtp = None
    config.loss.loss_weights = []
    config.training.local_batch_size = 8
    config.training.seq_len = 512
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_microbatch_size = 1
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    moes = [cfg for _, cfg, *_ in config.model_spec.model.traverse(MoE.Config)]
    for moe in moes:
        moe.experts.token_dispatcher = AllToAllTokenDispatcher.Config(token_drop=policy)
    with caplog.at_level(logging.INFO):
        Decoder.prepare_config(config.model_spec.model, config=config)
    messages = [
        record.getMessage()
        for record in caplog.records
        if "Token drop [" in record.getMessage()
    ]
    assert len(messages) == len(moes)
    for message in messages:
        assert "max_tokens_per_forward=128" in message
        assert "mean_routes_per_expert=48.000" in message
        assert "tokens_without_PP_TP_CP=4096" in message
        assert "mean_routes_per_expert_without_PP_TP_CP=1536.000" in message
        assert f"peak_factor_source_quota={peak}; source_quota={quota}," in message
        assert f"EP_group_quota_per_expert={2 * quota} (EP=2)" in message
    for moe in moes:
        assert moe.experts.token_dispatcher.token_drop is policy
        assert moe.experts.token_dispatcher._capacity_num_tokens == 128
