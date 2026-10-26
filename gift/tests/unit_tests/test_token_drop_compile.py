"""Fullgraph token-drop selection, conditional execution and gradient parity."""

from tests.unit_tests.config_helpers import load_experiment_config
from torchtitan.models.common.decoder_blocks.decoder import Decoder

import shlex
from pathlib import Path

import pytest
import torch

from torchtitan.config import CompileConfig
from torchtitan.config.moe import TokenDropConfig
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe.dispatch import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
    LocalTokenDispatcher,
)


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
@pytest.mark.parametrize("branchless", [None, False, True])
@pytest.mark.parametrize(
    "enabled,components",
    [(False, ["model"]), (True, ["loss"]), (True, ["model", "loss"])],
)
def test_model_compile_accepts_both_token_drop_paths(
    experiment, branchless, enabled, components
):
    config = load_experiment_config(experiment)
    config.compile = CompileConfig(enable=enabled, components=components)
    moes = [cfg for _, cfg, *_ in config.model_spec.model.traverse(MoE.Config)]
    assert moes
    policy = (
        None
        if branchless is None
        else TokenDropConfig(
            expert_capacity_factor=0.5, branchless_capacity_drop=branchless
        )
    )
    for moe in moes:
        moe.experts.token_dispatcher = AllToAllTokenDispatcher.Config(token_drop=policy)
    Decoder.prepare_config(config.model_spec.model, config=config)
    if policy is not None:
        assert policy.branchless_capacity_drop is branchless


@pytest.mark.parametrize(
    "backend",
    [
        LocalTokenDispatcher,
        AllToAllTokenDispatcher,
        DeepEPTokenDispatcher,
        HybridEPTokenDispatcher,
    ],
)
def test_compile_config_preserves_conditional_policy_for_every_backend(backend):
    config = load_experiment_config("gift_moe_debug.yaml")
    config.compile = CompileConfig(enable=True)
    config.parallelism.expert_parallel_degree = 2
    for _, moe, *_ in config.model_spec.model.traverse(MoE.Config):
        moe.experts.token_dispatcher = backend.Config(
            token_drop=TokenDropConfig(expert_capacity_factor=0.5)
        )
    Decoder.prepare_config(config.model_spec.model, config=config)
    for _, moe, *_ in config.model_spec.model.traverse(MoE.Config):
        assert not moe.experts.token_dispatcher.token_drop.branchless_capacity_drop


@pytest.mark.parametrize(
    "strategy", ["capacity_factor", "fixed_capacity", "peak_capacity_factor"]
)
@pytest.mark.parametrize("group_gate", [False, True])
@pytest.mark.parametrize("branchless", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_fullgraph_matches_eager_masks_and_gradients(
    strategy, group_gate, branchless, device
):
    # AOT on CPU verifies fullgraph forward/backward capture without a CUDA
    # dependency. CUDA exercises both selectors through Inductor.
    policy = TokenDropConfig(
        strategy=strategy,
        expert_capacity_factor=None if strategy == "fixed_capacity" else 0.5,
        expert_capacity_tokens=1 if strategy == "fixed_capacity" else None,
        branchless_capacity_drop=branchless,
        drop_by_group_capacity=group_gate,
    )
    dispatcher = (
        LocalTokenDispatcher.Config(token_drop=policy)
        .with_capacity_context(num_tokens=4, top_k=2, num_experts=4, ep_size=1)
        .build()
    )

    def route(scores, ids):
        return dispatcher._drop_routes(scores, ids, num_experts=4)

    torch._dynamo.reset()
    with torch._dynamo.config.patch(suppress_errors=False):
        compiled = torch.compile(
            route,
            backend="aot_eager" if device == "cpu" else "inductor",
            fullgraph=True,
        )
        # The same compiled function sees both overload outcomes. The group
        # gate must also skip local overload at exactly the destination budget.
        for ids, retained in (
            ([[0, 1], [2, 3], [-1, -1], [-1, -1]], 4),
            ([[0, 1], [0, 1], [0, 1], [0, 1]], 2),
            ([[0, 1], [0, 1], [-1, -1], [-1, -1]], 4 if group_gate else 2),
            ([[0, -1], [0, 1], [0, 1], [0, 1]], 2),
        ):
            ids = torch.tensor(ids, device=device)
            scores = (torch.arange(1, 9, device=device).float() / 16).reshape(4, 2)
            scores = scores.masked_fill(ids < 0, 0).requires_grad_()
            eager_scores = scores.detach().clone().requires_grad_()
            actual = compiled(scores, ids)
            expected = route(eager_scores, ids)
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
            assert actual[2].sum().item() == retained
            actual[0].square().sum().backward()
            expected[0].square().sum().backward()
            torch.testing.assert_close(scores.grad, eager_scores.grad, rtol=0, atol=0)
    torch._dynamo.reset()


@pytest.mark.parametrize("branchless", [False, True])
def test_h100_compile_row_uses_expected_hybridep_config(branchless):
    from tests.integration_tests.h100 import build_h100_tests_list
    from tests.integration_tests.run_tests import _split_config_file

    mode = "branchless" if branchless else "conditional"
    row = next(
        test
        for test in build_h100_tests_list()
        if test.test_name == f"gift_moe_fsdp+hybridep+token_drop+{mode}+compile"
    )
    config_file, overrides = _split_config_file(row.override_args[0])
    config = load_experiment_config(
        Path(config_file).name, *shlex.split(" ".join(overrides))
    )
    assert config.compile.enable and "model" in config.compile.components
    assert config.training.steps >= 10
    Decoder.prepare_config(config.model_spec.model, config=config)
    dispatchers = [
        cfg
        for _, cfg, *_ in config.model_spec.model.traverse(
            HybridEPTokenDispatcher.Config
        )
    ]
    assert dispatchers
    assert all(
        cfg.token_drop.branchless_capacity_drop is branchless and not cfg.fuse_permute
        for cfg in dispatchers
    )


@pytest.mark.parametrize("group_gate", [False, True])
def test_compiled_conditional_skips_sort_at_runtime(group_gate):
    from torch._dynamo.testing import CompileCounterWithBackend
    from torchtitan.models.common.moe.dispatch.capacity import select_capacity_routes

    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=1,
        drop_by_group_capacity=group_gate,
    )

    def route(scores, ids):
        return select_capacity_routes(scores, ids, num_experts=4, policy=policy)

    torch._dynamo.reset()
    backend = CompileCounterWithBackend("inductor")
    compiled = torch.compile(route, backend=backend, fullgraph=True)
    scores = torch.arange(1, 9).float().reshape(4, 2)
    cases = [
        (torch.tensor([[0, 1], [2, 3], [-1, -1], [-1, -1]]), 0),
        (torch.tensor([[0, 1]] * 4), 1),
        (torch.tensor([[0, 1], [0, 1], [-1, -1], [-1, -1]]), 0 if group_gate else 1),
    ]
    for ids, _ in cases:
        compiled(scores, ids)
    # Compile/trace work is finished before profiling. This checks execution of
    # the selected branch, including a locally overloaded but group-safe case.
    for ids, expected_calls in cases + cases[:1]:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as trace:
            actual = compiled(scores, ids)
        count = sum(
            event.count for event in trace.key_averages() if event.key == "aten::sort"
        )
        assert count == expected_calls
        torch.testing.assert_close(actual, route(scores, ids), rtol=0, atol=0)
    assert backend.frame_count == 1
    torch._dynamo.reset()


@pytest.mark.parametrize("num_tokens", [0, 1, 4])
@pytest.mark.parametrize("zero_quota", [False, True])
@pytest.mark.parametrize("branchless", [False, True])
def test_compiled_empty_inputs_and_quota_edges(num_tokens, zero_quota, branchless):
    from torchtitan.models.common.moe.dispatch.capacity import select_capacity_routes

    policy = (
        TokenDropConfig(expert_capacity_factor=0.125)
        if zero_quota
        else TokenDropConfig(strategy="fixed_capacity", expert_capacity_tokens=128)
    )

    policy.branchless_capacity_drop = branchless

    def route(scores, ids):
        keep = select_capacity_routes(scores, ids, num_experts=4, policy=policy)
        return scores * keep, keep

    torch._dynamo.reset()
    compiled = torch.compile(route, backend="inductor", fullgraph=True)
    ids = torch.tensor([0, 1]).expand(num_tokens, 2).clone()
    scores = torch.ones(num_tokens, 2, requires_grad=True)
    result, mask = compiled(scores, ids)
    expected = torch.full_like(ids, not zero_quota, dtype=torch.bool)
    torch.testing.assert_close(mask, expected, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(scores.grad, expected.float(), rtol=0, atol=0)
    torch._dynamo.reset()


@pytest.mark.parametrize("needs_drop", [False, True])
def test_conditional_operator_schema_fake_and_autograd(needs_drop):
    from torchtitan.models.common.moe.dispatch.capacity import (
        _conditional_capacity_mask,
    )

    scores = torch.arange(1, 9).float().reshape(4, 2).requires_grad_()
    buckets = torch.tensor([0, 1] * 4)
    valid = torch.ones(8, dtype=torch.bool)
    torch.library.opcheck(
        _conditional_capacity_mask,
        (scores, buckets, valid, torch.tensor(needs_drop), 4, 1),
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_compiled_conditional_checkpoint_matches_eager_gradients(device):
    from torch.utils.checkpoint import checkpoint
    from torchtitan.models.common.moe.dispatch.capacity import select_capacity_routes

    policy = TokenDropConfig(strategy="fixed_capacity", expert_capacity_tokens=1)

    def route(scores, ids):
        keep = select_capacity_routes(scores, ids, num_experts=4, policy=policy)
        return scores * keep

    def forward(scores, ids):
        return checkpoint(route, scores, ids, use_reentrant=False).square().sum()

    torch._dynamo.reset()
    compiled = torch.compile(forward, backend="inductor", fullgraph=True)
    for ids in (
        torch.tensor([[0, 1], [2, 3], [-1, -1], [-1, -1]], device=device),
        torch.tensor([[0, 1]] * 4, device=device),
    ):
        scores = (
            torch.arange(1, 9, device=device).float().reshape(4, 2).requires_grad_()
        )
        reference_scores = scores.detach().clone().requires_grad_()
        loss = compiled(scores, ids)
        expected = forward(reference_scores, ids)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        expected.backward()
        torch.testing.assert_close(scores.grad, reference_scores.grad, rtol=0, atol=0)
    torch._dynamo.reset()


@pytest.mark.parametrize("bound", [None, 4])
def test_hybridep_capacity_metadata_is_fullgraph_traceable(bound):
    config = HybridEPTokenDispatcher.Config(
        token_drop=TokenDropConfig(expert_capacity_factor=0.5),
        non_blocking_capacity_factor=1.0,
        fuse_permute=False,
    )
    if bound is not None:
        config = config.with_capacity_context(
            num_tokens=bound, top_k=2, num_experts=4, ep_size=2
        )

    def validate(value):
        config.validate_capacity(
            num_tokens=value.shape[0], top_k=2, num_experts=4, ep_size=2
        )
        return value + 1

    value = torch.ones(4)
    torch._dynamo.reset()
    compiled = torch.compile(validate, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(value), validate(value), rtol=0, atol=0)
