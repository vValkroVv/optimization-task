"""HybridEP capacity compatibility, without importing its CUDA extension."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchtitan.config.moe import TokenDropConfig
from torchtitan.distributed.deepep.capacity import (
    non_blocking_buffer_capacity,
    retained_receive_capacity,
)
from torchtitan.models.common.moe.dispatch import (
    BaseEPTokenDispatcher,
    HybridEPTokenDispatcher,
)


@pytest.mark.parametrize("group_gate", [False, True])
@pytest.mark.parametrize("branchless", [False, True])
@pytest.mark.parametrize(
    "strategy", ["capacity_factor", "fixed_capacity", "peak_capacity_factor"]
)
def test_buffer_boundary_accepts_equal_and_rejects_one_row_short(
    group_gate, branchless, strategy
):
    # T=8, E=4, K=2, S=2: Q=2, receiver cap=8. Buffer=32*cf.
    policy = TokenDropConfig(
        strategy=strategy,
        expert_capacity_tokens=4 if strategy == "fixed_capacity" else None,
        expert_capacity_factor=None if strategy == "fixed_capacity" else 0.5,
        drop_by_group_capacity=group_gate,
        branchless_capacity_drop=branchless,
    )
    config = HybridEPTokenDispatcher.Config(
        token_drop=policy, non_blocking_capacity_factor=0.25, fuse_permute=False
    )
    args = dict(num_tokens=8, top_k=2, num_experts=4, ep_size=2)
    config.with_capacity_context(**args)
    with pytest.raises(ValueError, match="allocates 7 rows.*require 8 padded rows"):
        replace(config, non_blocking_capacity_factor=7 / 32).with_capacity_context(
            **args
        )
    # Blocking mode allocates from actual counts, so this restriction is absent.
    replace(config, non_blocking_capacity_factor=None).with_capacity_context(**args)


def test_retained_routes_can_overflow_reduced_buffer():
    policy = TokenDropConfig(expert_capacity_factor=1.0)
    tokens, ep_size, top_k, experts = 1024, 2, 3, 8
    layout = dict(num_tokens=tokens, top_k=top_k, num_experts=experts, ep_size=ep_size)
    quota = policy.source_capacity(**layout)
    # Each source selects experts 0, 1, 2, all owned by rank zero.
    retained_at_rank_zero = ep_size * top_k * quota
    assert retained_at_rank_zero == 2304
    assert (
        non_blocking_buffer_capacity(tokens, ep_size, experts // ep_size, top_k, 0.125)
        == 768
    )
    assert (
        non_blocking_buffer_capacity(tokens, ep_size, experts // ep_size, top_k, 1.0)
        >= retained_at_rank_zero
    )
    config = HybridEPTokenDispatcher.Config(
        token_drop=policy,
        non_blocking_capacity_factor=1.0,
        pad_multiple=1,
        fuse_permute=False,
    )
    config.with_capacity_context(**layout)
    with pytest.raises(ValueError, match="additional, implicit drops"):
        replace(config, non_blocking_capacity_factor=0.125).with_capacity_context(
            **layout
        )
    assert replace(policy, expert_capacity_factor=0.5).source_capacity(**layout) == 192


def test_padding_and_group_gate_are_part_of_the_bound():
    # T=8, E=8, K=4, S=2, Q=2: strict receives <=4/expert.
    # With gate on, <=16 total may fill each expert asymmetrically, e.g.
    # (5, 5, 5, 1), which pads to 28 instead of 16.
    policy = TokenDropConfig(expert_capacity_factor=0.5)
    config = HybridEPTokenDispatcher.Config(
        token_drop=policy,
        non_blocking_capacity_factor=0.25,
        pad_multiple=4,
        fuse_permute=False,
    )
    args = dict(num_tokens=8, top_k=4, num_experts=8, ep_size=2)
    config.with_capacity_context(**args)
    gated = replace(config, token_drop=replace(policy, drop_by_group_capacity=True))
    with pytest.raises(ValueError, match="allocates 16 rows.*require 28 padded rows"):
        gated.with_capacity_context(**args)
    replace(gated, non_blocking_capacity_factor=28 / 64).with_capacity_context(**args)


def test_factor_one_does_not_hide_per_expert_padding_overflow():
    config = HybridEPTokenDispatcher.Config(
        token_drop=TokenDropConfig(strategy="fixed_capacity", expert_capacity_tokens=8),
        non_blocking_capacity_factor=1,
        pad_multiple=4,
        fuse_permute=False,
    )
    with pytest.raises(ValueError, match="allocates 4 rows.*require 8 padded rows"):
        config.with_capacity_context(num_tokens=2, top_k=1, num_experts=4, ep_size=2)


@pytest.mark.parametrize("alignment", [1, 2, 4])
@pytest.mark.parametrize("group_gate", [False, True])
def test_receive_bound_covers_exhaustive_small_assignments(alignment, group_gate):
    # Independent enumeration of source/expert counts, including absent experts.
    for n, k, quota in product(range(1, 4), (1, 2), range(4)):
        source_counts = [v for v in product(range(n + 1), repeat=2) if sum(v) <= n * k]
        bound = retained_receive_capacity(
            num_tokens=n,
            source_capacity=quota,
            ep_size=2,
            num_local_experts=2,
            top_k=k,
            drop_by_group_capacity=group_gate,
            pad_multiple=alignment,
        )
        for left, right in product(source_counts, repeat=2):
            overloaded = sum(left) + sum(right) > quota * 2 * 2
            rows = []
            for expert in range(2):
                a, b = left[expert], right[expert]
                if not group_gate or overloaded:
                    a, b = min(a, quota), min(b, quota)
                rows.append(((a + b + alignment - 1) // alignment) * alignment)
            assert sum(rows) <= bound, (n, k, quota, left, right, rows, bound)


@pytest.mark.parametrize("top_k", [1, 4, 8])
def test_buffer_uses_min_topk_and_local_experts_and_final_alignment(top_k):
    expected = int(7 * 2 * min(4, top_k) * 0.4)
    assert (
        non_blocking_buffer_capacity(7, 2, 4, top_k, 0.4, 4)
        == ((expected + 3) // 4) * 4
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"non_blocking_capacity_factor": True},
        {"non_blocking_capacity_factor": float("nan")},
        {"non_blocking_capacity_factor": 0},
        {"non_blocking_capacity_factor": 1.1},
        {"pad_multiple": 0},
        {"pad_multiple": True},
        {"pad_multiple": 1.5},
    ],
)
def test_invalid_buffer_config_fails_without_loading_extension(kwargs):
    with pytest.raises(ValueError):
        HybridEPTokenDispatcher.Config(**kwargs)


@pytest.mark.parametrize("token_drop", [False, True])
@pytest.mark.parametrize("factor", [None, 0.25, 1.0])
@pytest.mark.parametrize("fused", [False, True])
def test_fused_nonblocking_token_drop_config_guard(token_drop, factor, fused):
    kwargs = dict(
        token_drop=TokenDropConfig(expert_capacity_factor=0.5) if token_drop else None,
        non_blocking_capacity_factor=factor,
        fuse_permute=fused,
    )
    if token_drop and factor is not None and fused:
        with pytest.raises(
            ValueError, match="known bug.*incorrect expert counts"
        ) as exc:
            HybridEPTokenDispatcher.Config(**kwargs)
        assert "fuse_permute=False" in str(exc.value)
        assert "non_blocking_capacity_factor=None" in str(exc.value)
    else:
        HybridEPTokenDispatcher.Config(**kwargs).with_capacity_context(
            num_tokens=8, top_k=2, num_experts=4, ep_size=2
        )


@pytest.mark.parametrize("branchless", [False, True])
def test_default_fusion_rejected_even_with_worst_case_buffer(branchless):
    with pytest.raises(ValueError, match="fuse_permute=True"):
        HybridEPTokenDispatcher.Config(
            token_drop=TokenDropConfig(
                expert_capacity_factor=0.5, branchless_capacity_drop=branchless
            ),
            non_blocking_capacity_factor=1.0,
        )


@pytest.mark.parametrize(
    "overridden_field,safe_value",
    [
        ("token_drop", None),
        ("non_blocking_capacity_factor", None),
        ("fuse_permute", False),
    ],
)
def test_build_rejects_unsafe_config_override_before_loading_extension(
    overridden_field, safe_value
):
    kwargs = dict(
        token_drop=TokenDropConfig(expert_capacity_factor=0.5),
        non_blocking_capacity_factor=1.0,
        fuse_permute=True,
    )
    unsafe_value = kwargs[overridden_field]
    kwargs[overridden_field] = safe_value
    config = HybridEPTokenDispatcher.Config(**kwargs)
    setattr(config, overridden_field, unsafe_value)
    with pytest.raises(ValueError, match="known bug.*incorrect expert counts"):
        config.build()


def make_dispatcher(config):
    # Exercise the real dispatcher and its shared base; mock only CUDA transport.
    dispatcher = object.__new__(HybridEPTokenDispatcher)
    BaseEPTokenDispatcher.__init__(dispatcher, config)
    dispatcher._capacity_config = config
    dispatcher.non_blocking_capacity_factor = config.non_blocking_capacity_factor
    dispatcher.pad_multiple = config.pad_multiple
    dispatcher.ep_mesh = SimpleNamespace(
        get_group=lambda: SimpleNamespace(size=lambda: 2)
    )
    dispatcher._dispatch_tokens = Mock(
        return_value=(torch.ones(2, 3), torch.tensor([2, 0]), object())
    )
    return dispatcher


@pytest.mark.parametrize("branchless", [False, True])
@pytest.mark.parametrize(
    "strategy", ["capacity_factor", "fixed_capacity", "peak_capacity_factor"]
)
def test_hybridep_drops_before_transport_and_reports_requested_routes(
    branchless, strategy
):
    config = HybridEPTokenDispatcher.Config(
        token_drop=TokenDropConfig(
            strategy=strategy,
            expert_capacity_tokens=4 if strategy == "fixed_capacity" else None,
            expert_capacity_factor=None if strategy == "fixed_capacity" else 0.5,
            branchless_capacity_drop=branchless,
        ),
        non_blocking_capacity_factor=0.25,
        fuse_permute=False,
    ).with_capacity_context(num_tokens=8, top_k=2, num_experts=4, ep_size=2)
    dispatcher = make_dispatcher(config)
    observed = []
    dispatcher.dispatch_monitor = observed.append
    scores = (
        torch.stack((torch.arange(8, 0, -1), torch.arange(1, 9)), 1)
        .float()
        .requires_grad_()
    )
    ids = torch.tensor([0, 1]).expand(8, 2).clone()
    demand = torch.tensor([8, 8, 0, 0])
    dispatcher.dispatch(
        torch.ones(8, 3), scores, ids, demand, score_before_experts=False
    )
    call = dispatcher._dispatch_tokens.call_args
    expected = torch.zeros(8, 2, dtype=torch.bool)
    expected[:2, 0] = True
    expected[-2:, 1] = True
    torch.testing.assert_close(call.args[1], ids.masked_fill(~expected, -1))
    torch.testing.assert_close(call.args[2], scores * expected)
    assert call.kwargs["non_blocking_expert_capacity_factor"] == 0.25
    assert observed[0].requested_topk_scores is scores
    assert observed[0].local_tokens_per_expert is demand
    torch.testing.assert_close(observed[0].capacity_keep_mask, expected)
    call.args[2].sum().backward()
    torch.testing.assert_close(scores.grad, expected.float())


@pytest.mark.parametrize("actual_tokens", [0, 7, 9])
def test_nonblocking_refuses_dynamic_source_shape_before_transport(actual_tokens):
    config = HybridEPTokenDispatcher.Config(
        token_drop=TokenDropConfig(expert_capacity_factor=0.5),
        non_blocking_capacity_factor=0.25,
        fuse_permute=False,
    ).with_capacity_context(num_tokens=8, top_k=2, num_experts=4, ep_size=2)
    dispatcher = make_dispatcher(config)
    with pytest.raises(ValueError, match="equal source token counts"):
        dispatcher.dispatch(
            torch.ones(actual_tokens, 3),
            torch.ones(actual_tokens, 2),
            torch.zeros(actual_tokens, 2, dtype=torch.long),
            torch.zeros(4),
            score_before_experts=False,
        )
    dispatcher._dispatch_tokens.assert_not_called()


def test_runtime_guard_rechecks_changed_policy_before_transport():
    config = HybridEPTokenDispatcher.Config(
        token_drop=TokenDropConfig(expert_capacity_factor=0.5),
        non_blocking_capacity_factor=0.25,
        fuse_permute=False,
    ).with_capacity_context(num_tokens=8, top_k=2, num_experts=4, ep_size=2)
    config.token_drop.expert_capacity_factor = 1
    dispatcher = make_dispatcher(config)
    with pytest.raises(ValueError, match="additional, implicit drops"):
        dispatcher.dispatch(
            torch.ones(8, 3),
            torch.ones(8, 2),
            torch.zeros(8, 2, dtype=torch.long),
            torch.zeros(4),
            score_before_experts=False,
        )
    dispatcher._dispatch_tokens.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [0, 1, 7])
def test_dense_routing_preserves_dropless_metadata(dtype, num_tokens):
    from torchtitan.distributed.deepep.routing import indices_to_map

    ids = torch.tensor([0, 2]).expand(num_tokens, 2)
    weights = torch.arange(num_tokens * 2).view(num_tokens, 2).to(dtype) / 32
    routing, probs = indices_to_map(ids, weights, 4)
    # Match the pinned HybridEP helper, including a selected zero-weight route.
    expected_map = torch.zeros(num_tokens, 4, dtype=torch.bool).scatter(1, ids, True)
    expected_probs = torch.zeros(num_tokens, 4).scatter(1, ids, weights.float())
    torch.testing.assert_close(routing, expected_map, rtol=0, atol=0)
    torch.testing.assert_close(probs, expected_probs, rtol=0, atol=0)
    assert routing.is_contiguous() and probs.is_contiguous()
    # Both outputs view their final allocation; there is only one sink element.
    assert routing.untyped_storage().nbytes() == num_tokens * 4 + 1
    assert probs.untyped_storage().nbytes() == (num_tokens * 4 + 1) * 4


def test_dropped_slots_do_not_overwrite_expert_zero_or_leak_gradients():
    from torchtitan.distributed.deepep.routing import (
        indices_to_map,
        gather_route_gradients,
    )

    ids = torch.tensor([[0, -1], [-1, 0], [-1, -1], [2, 1]])
    weights = torch.tensor(
        [[0.5, 0], [0, 0.25], [0, 0], [0.75, 0.125]], requires_grad=True
    )
    routing, probs = indices_to_map(ids, weights, 3)
    expected = torch.tensor([[0.5, 0, 0], [0.25, 0, 0], [0, 0, 0], [0, 0.125, 0.75]])
    torch.testing.assert_close(probs, expected)
    torch.testing.assert_close(routing, expected != 0)
    dense_grad = torch.arange(1, 13).view(4, 3).float()
    (probs * dense_grad).sum().backward()
    actual = gather_route_gradients(dense_grad, ids)
    torch.testing.assert_close(
        actual, torch.tensor([[1.0, 0.0], [0.0, 4.0], [0.0, 0.0], [12.0, 11.0]])
    )
    torch.testing.assert_close(actual, weights.grad)


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
def test_dense_routing_does_not_clone_full_matrices(device):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torchtitan.distributed.deepep.routing import indices_to_map

    class RejectDenseClone(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.clone.default:
                raise AssertionError(
                    "Dense routing conversion must not materialize sliced matrices"
                )
            return func(*args, **(kwargs or {}))

    ids = torch.tensor([[0, -1], [-1, 2], [-1, -1], [1, 3]], device=device)
    scores = torch.ones_like(ids, dtype=torch.float32, requires_grad=True)
    with RejectDenseClone():
        routing, probs = indices_to_map(ids, scores, 4)
    assert routing.sum().item() == 4
    probs.sum().backward()
    torch.testing.assert_close(scores.grad, (ids >= 0).float(), rtol=0, atol=0)
