"""Capacity policy against an explicit per-expert reference, on CPU."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint

from torchtitan.config.moe import TokenDropConfig
from torchtitan.models.common.moe.dispatch.base import BaseEPTokenDispatcher
from torchtitan.models.common.moe.dispatch.capacity import select_capacity_routes
from torchtitan.models.common.moe.dispatch.deepep import DeepEPTokenDispatcher


def reference(scores, ids, num_experts, capacity):
    keep = torch.zeros_like(ids, dtype=torch.bool)
    for expert in range(num_experts):
        slots = [
            (t, k)
            for t in range(ids.shape[0])
            for k in range(ids.shape[1])
            if ids[t, k] == expert and scores[t, k] != 0
        ]
        slots.sort(key=lambda slot: -float(scores[slot]))
        for slot in slots[:capacity]:
            keep[slot] = True
    return keep


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expert_capacity_factor": 0},
        {"expert_capacity_factor": -1},
        {"expert_capacity_factor": float("nan")},
        {"expert_capacity_factor": float("inf")},
        {"expert_capacity_factor": True},
        {"strategy": "unknown"},
        {"expert_capacity_factor": 1.0, "branchless_capacity_drop": "false"},
        {"expert_capacity_factor": 1.0, "branchless_capacity_drop": 1},
        {"expert_capacity_factor": 1.0, "branchless_capacity_drop": None},
        {"strategy": "fixed_capacity", "expert_capacity_tokens": 1.5},
        {"strategy": "fixed_capacity", "expert_capacity_tokens": 0},
        {"strategy": "fixed_capacity", "expert_capacity_tokens": True},
        {"strategy": "fixed_capacity"},
        {"strategy": "fixed_capacity", "expert_capacity_factor": 1.0},
        {"strategy": "peak_capacity_factor", "expert_capacity_tokens": 10},
        {
            "strategy": "fixed_capacity",
            "expert_capacity_tokens": 10,
            "expert_capacity_factor": 1.0,
        },
        {"expert_capacity_factor": 1.0, "expert_capacity_tokens": 10},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TokenDropConfig(**kwargs)


@pytest.mark.parametrize("tokens", [None, 7])
def test_fixed_capacity_rejects_factor_with_migration_message(tokens):
    with pytest.raises(
        ValueError, match="does not accept expert_capacity_factor"
    ) as exc:
        TokenDropConfig(
            strategy="fixed_capacity",
            expert_capacity_tokens=tokens,
            expert_capacity_factor=1.25,
        )
    assert "peak_capacity_factor" in str(exc.value)


def test_dispatcher_revalidates_overridden_fixed_capacity_policy():
    from torchtitan.models.common.moe.dispatch import LocalTokenDispatcher

    policy = TokenDropConfig(strategy="fixed_capacity", expert_capacity_tokens=7)
    config = LocalTokenDispatcher.Config(token_drop=policy)
    policy.expert_capacity_factor = 1.25
    with pytest.raises(ValueError, match="does not accept expert_capacity_factor"):
        config.build()


@pytest.mark.parametrize("branchless_capacity_drop", [False, True])
def test_capacity_rounding_and_serialization(branchless_capacity_drop):
    relative = TokenDropConfig(expert_capacity_factor=1.25)
    fixed = TokenDropConfig(strategy="fixed_capacity", expert_capacity_tokens=7)
    peak = TokenDropConfig(
        strategy="peak_capacity_factor",
        expert_capacity_factor=1.25,
        branchless_capacity_drop=branchless_capacity_drop,
    )
    args = dict(num_tokens=7, top_k=2, num_experts=4, ep_size=2)
    assert not relative.branchless_capacity_drop
    assert relative.source_capacity(**args) == 3
    assert fixed.source_capacity(**args) == 3
    assert peak.source_capacity(**args, max_num_tokens=7) == 4
    assert TokenDropConfig(**asdict(peak)) == peak
    with pytest.raises(ValueError, match="divisible"):
        relative.source_capacity(**{**args, "ep_size": 3})


@pytest.mark.parametrize("branchless_capacity_drop", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("capacity", [0, 1, 3, 100])
def test_matches_reference_with_ties_and_empty_experts(
    dtype, capacity, branchless_capacity_drop
):
    ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, -1], [0, 1]])
    scores = torch.tensor(
        [[0.5, 0.5], [0.9, 0.1], [0.5, 0.5], [0.2, 0], [0, 0.8]], dtype=dtype
    )
    # A positive configured group quota may round down to zero per source.
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=max(capacity * 2, 1),
        branchless_capacity_drop=branchless_capacity_drop,
    )
    group = SimpleNamespace(size=lambda: 2)
    keep = select_capacity_routes(
        scores, ids, num_experts=4, policy=policy, ep_group=group
    )
    expected = reference(scores, ids, 4, capacity)
    torch.testing.assert_close(keep, expected)


@pytest.mark.parametrize("branchless_capacity_drop", [False, True])
def test_empty_source_and_small_relative_quota(branchless_capacity_drop):
    policy = TokenDropConfig(
        expert_capacity_factor=1.0, branchless_capacity_drop=branchless_capacity_drop
    )
    keep = select_capacity_routes(
        torch.empty(0, 2),
        torch.empty(0, 2, dtype=torch.long),
        num_experts=4,
        policy=policy,
    )
    assert keep.shape == (0, 2)
    keep = select_capacity_routes(
        torch.ones(1, 2), torch.tensor([[0, 1]]), num_experts=4, policy=policy
    )
    assert not keep.any()


@pytest.mark.parametrize("branchless_capacity_drop", [False, True])
@pytest.mark.parametrize("overloaded", [False, True])
def test_group_gate_uses_destination_total_and_enters_collective(
    monkeypatch, overloaded, branchless_capacity_drop
):
    ids = torch.tensor([[0], [0], [0], [0]])
    scores = torch.tensor([[0.4], [0.3], [0.2], [0.1]])
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=4,
        drop_by_group_capacity=True,
        branchless_capacity_drop=branchless_capacity_drop,
    )
    group = SimpleNamespace(size=lambda: 2)
    # Both sources have quota 2. Each destination owns 2 experts: budget 8.
    # Expert 0 can overflow its quota while the destination remains under budget.
    sort = Mock(wraps=torch.argsort)
    topk = Mock(wraps=torch.topk)
    monkeypatch.setattr(torch, "argsort", sort)
    monkeypatch.setattr(torch, "topk", topk)
    other = torch.tensor([0, 5 if overloaded else 0, 0, 0])
    collective = Mock(side_effect=lambda tensor, reduce_op, group: tensor + other)
    monkeypatch.setattr(
        "torchtitan.models.common.moe.dispatch.capacity.funcol.all_reduce", collective
    )
    keep = select_capacity_routes(
        scores, ids, num_experts=4, policy=policy, ep_group=group
    )
    assert keep.sum() == (2 if overloaded else 4)
    collective.assert_called_once()
    assert bool(sort.call_count) == (branchless_capacity_drop or overloaded)
    topk.assert_not_called()


@pytest.mark.parametrize("branchless_capacity_drop", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_expert_outputs_and_all_gradients_match_reference(
    recompute, branchless_capacity_drop
):
    ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    scores = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.7, 0.3]], requires_grad=True
    )
    x = torch.randn(4, 3, requires_grad=True)
    weights = torch.randn(2, 3, 3, requires_grad=True)
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=1,
        branchless_capacity_drop=branchless_capacity_drop,
    )

    def forward(x, scores, weights):
        keep = select_capacity_routes(scores, ids, num_experts=2, policy=policy)
        expert_out = torch.einsum("td,tkdf->tkf", x, weights[ids])
        return (expert_out * (scores * keep).unsqueeze(-1)).sum(1)

    actual = (
        checkpoint(forward, x, scores, weights, use_reentrant=False)
        if recompute
        else forward(x, scores, weights)
    )
    grads = torch.autograd.grad(actual.square().sum(), (x, scores, weights))
    xr, sr, wr = [v.detach().clone().requires_grad_() for v in (x, scores, weights)]
    keep = reference(sr.detach(), ids, 2, 1)
    expected = torch.stack(
        [
            sum(
                (sr[t, k] * (xr[t] @ wr[ids[t, k]]) for k in range(2) if keep[t, k]),
                xr[t] * 0,
            )
            for t in range(4)
        ]
    )
    expected_grads = torch.autograd.grad(expected.square().sum(), (xr, sr, wr))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert (
        actual[[1, 3]] == 0
    ).all()  # Tokens losing both routes stay in the sequence.
    assert (grads[1][~keep] == 0).all()


@pytest.mark.parametrize("branchless_capacity_drop", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_dispatch_masks_before_transport_and_preserves_router_demand(
    enabled, branchless_capacity_drop
):
    # CPU contract test; GPU integration rows exercise the real transport.
    dispatcher = object.__new__(DeepEPTokenDispatcher)
    policy = (
        TokenDropConfig(
            strategy="fixed_capacity",
            expert_capacity_tokens=2,
            branchless_capacity_drop=branchless_capacity_drop,
        )
        if enabled
        else None
    )
    BaseEPTokenDispatcher.__init__(
        dispatcher, DeepEPTokenDispatcher.Config(token_drop=policy)
    )
    group = SimpleNamespace(size=lambda: 2)
    dispatcher.ep_mesh = SimpleNamespace(get_group=lambda: group)
    observed = []
    dispatcher.dispatch_monitor = observed.append
    x = torch.randn(4, 3)
    scores = torch.tensor([[0.9], [0.8], [0.7], [0.6]], requires_grad=True)
    ids = torch.zeros(4, 1, dtype=torch.long)
    demand = torch.tensor([4, 0])
    received = torch.tensor([1 if enabled else 4])
    dispatcher._dispatch_tokens = Mock(return_value=(x, received, object()))
    dispatcher.dispatch(x, scores, ids, demand, score_before_experts=False)
    args = dispatcher._dispatch_tokens.call_args.args
    if enabled:
        torch.testing.assert_close(args[1], torch.tensor([[0], [-1], [-1], [-1]]))
        torch.testing.assert_close(args[2], torch.tensor([[0.9], [0], [0], [0]]))
        assert observed[0].capacity_keep_mask.sum() == 1
    else:
        assert args[1] is ids and args[2] is scores
        assert observed[0].capacity_keep_mask is None
    assert observed[0].local_tokens_per_expert is demand
    assert observed[0].global_tokens_per_local_expert is received
    torch.testing.assert_close(ids, torch.zeros_like(ids))
    torch.testing.assert_close(
        scores.detach(), torch.tensor([[0.9], [0.8], [0.7], [0.6]])
    )


def _group_capacity_worker(rank, store_path):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=2
    )
    try:
        for num_tokens, relative in [(8, True), (8, False), (9, False)]:
            n = num_tokens if rank == 1 else 0
            scores = torch.ones(n, 1)
            ids = torch.zeros(n, 1, dtype=torch.long)
            policy = (
                TokenDropConfig(expert_capacity_factor=1.0, drop_by_group_capacity=True)
                if relative
                else TokenDropConfig(
                    strategy="fixed_capacity",
                    expert_capacity_tokens=4,
                    drop_by_group_capacity=True,
                )
            )
            for branchless_capacity_drop in (False, True):
                keep = select_capacity_routes(
                    scores,
                    ids,
                    num_experts=4,
                    policy=replace(
                        policy, branchless_capacity_drop=branchless_capacity_drop
                    ),
                    ep_group=dist.group.WORLD,
                )
                expected = 0 if rank == 0 else (8 if num_tokens == 8 else 2)
                assert int(keep.sum()) == expected
    finally:
        dist.destroy_process_group()


def test_group_capacity_with_real_collective_and_empty_source(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _group_capacity_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True
    )


@pytest.mark.parametrize("num_tokens", [0, 4])
@pytest.mark.parametrize("drop_by_group_capacity", [False, True])
def test_conditional_skips_selection_and_preserves_inactive_routes(
    monkeypatch, num_tokens, drop_by_group_capacity
):
    ids = torch.tensor([[0, -1], [0, 1], [1, 0], [-1, 1]])[:num_tokens]
    scores = torch.tensor([[0.9, 0.5], [0.8, 0.0], [0.7, 0.0], [0.4, 0.6]])[:num_tokens]
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=2,
        drop_by_group_capacity=drop_by_group_capacity,
        branchless_capacity_drop=False,
    )
    sort = Mock(side_effect=AssertionError("No overload: sorting must be skipped"))
    monkeypatch.setattr(torch, "argsort", sort)
    monkeypatch.setattr(torch, "topk", sort)
    keep = select_capacity_routes(scores, ids, num_experts=2, policy=policy)
    torch.testing.assert_close(keep, (ids >= 0) & (scores != 0))
    sort.assert_not_called()


@pytest.mark.parametrize("drop_by_group_capacity", [False, True])
def test_device_only_selection_does_not_read_tensor_scalars(drop_by_group_capacity):
    class RejectScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("Device-only selection must not read GPU scalars")
            return func(*args, **(kwargs or {}))

    ids = torch.tensor([[0, 1], [0, 1]])
    scores = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    for capacity in (1, 2):
        policy = TokenDropConfig(
            strategy="fixed_capacity",
            expert_capacity_tokens=capacity,
            drop_by_group_capacity=drop_by_group_capacity,
            branchless_capacity_drop=True,
        )
        with RejectScalarRead():
            keep = select_capacity_routes(scores, ids, num_experts=2, policy=policy)
        torch.testing.assert_close(keep, reference(scores, ids, 2, capacity))


@pytest.mark.parametrize("drop_by_group_capacity", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("capacity", [3, 16])
def test_execution_modes_have_identical_training_updates_without_ties(
    drop_by_group_capacity, recompute, capacity
):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(8, 4, generator=generator)
    initial_gate = torch.randn(4, 4, generator=generator)
    initial_experts = torch.randn(4, 4, 4, generator=generator)
    balanced_ids = torch.tensor([[0, 1], [2, 3]] * 2)
    overloaded_ids = torch.tensor([[0, 1]] * 8)

    def train(branchless_capacity_drop):
        gate = initial_gate.clone().requires_grad_()
        experts = initial_experts.clone().requires_grad_()
        optimizer = torch.optim.SGD([gate, experts], lr=0.01)
        policy = TokenDropConfig(
            strategy="fixed_capacity",
            expert_capacity_tokens=capacity,
            drop_by_group_capacity=drop_by_group_capacity,
            branchless_capacity_drop=branchless_capacity_drop,
        )
        history = []
        for step in range(10):
            ids = balanced_ids if step % 2 == 0 else overloaded_ids

            def forward(x, gate, experts):
                scores = (x @ gate).softmax(-1).gather(1, ids)
                keep = select_capacity_routes(scores, ids, num_experts=4, policy=policy)
                routed = torch.einsum("td,tkdf->tkf", x, experts[ids])
                return (routed * (scores * keep).unsqueeze(-1)).sum(1)

            optimizer.zero_grad()
            inputs = x[: ids.shape[0]]
            output = (
                checkpoint(forward, inputs, gate, experts, use_reentrant=False)
                if recompute
                else forward(inputs, gate, experts)
            )
            loss = output.square().mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([gate, experts], 1.0)
            optimizer.step()
            history.append(
                tuple(
                    value.detach().clone()
                    for value in (
                        loss,
                        grad_norm,
                        gate.grad,
                        experts.grad,
                        gate,
                        experts,
                    )
                )
            )
        return history

    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        device_history = train(True)
        cpu_history = train(False)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    for device_step, cpu_step in zip(device_history, cpu_history):
        for device_value, cpu_value in zip(device_step, cpu_step):
            torch.testing.assert_close(device_value, cpu_value, rtol=0, atol=0)


@pytest.mark.parametrize("branchless_capacity_drop", [False, True])
@pytest.mark.parametrize("local_overload", [False, True])
@pytest.mark.parametrize("global_overload", [False, True])
def test_group_gate_requires_both_local_and_global_overload(
    monkeypatch, branchless_capacity_drop, local_overload, global_overload
):
    num_tokens = 4 if local_overload else 1
    scores = torch.arange(num_tokens, 0, -1, dtype=torch.float32).view(-1, 1)
    ids = torch.zeros(num_tokens, 1, dtype=torch.long)
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=4,
        drop_by_group_capacity=True,
        branchless_capacity_drop=branchless_capacity_drop,
    )
    # The remote source overloads destination 1; local routes go to destination 0.
    other = torch.tensor([0, 0, 9 if global_overload else 0, 0])
    collective = Mock(side_effect=lambda tensor, reduce_op, group: tensor + other)
    monkeypatch.setattr(
        "torchtitan.models.common.moe.dispatch.capacity.funcol.all_reduce", collective
    )
    import torchtitan.models.common.moe.dispatch.capacity as capacity_module

    selector = Mock(wraps=capacity_module._sorted_capacity_mask)
    monkeypatch.setattr(capacity_module, "_sorted_capacity_mask", selector)
    keep = select_capacity_routes(
        scores,
        ids,
        num_experts=4,
        policy=policy,
        ep_group=SimpleNamespace(size=lambda: 2),
    )
    collective.assert_called_once()
    assert bool(selector.call_count) == (
        not branchless_capacity_drop and local_overload and global_overload
    )
    assert int(keep.sum()) == (2 if local_overload and global_overload else num_tokens)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("capacity", [1, 3])
def test_branchless_uses_one_stable_sort_even_without_overload(
    monkeypatch, dtype, capacity
):
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
    scores = torch.tensor([[0.8, 0.2], [0.4, 0.6], [0.5, 0.5]], dtype=dtype)
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=capacity,
        branchless_capacity_drop=True,
    )
    sort = Mock(wraps=torch.argsort)
    monkeypatch.setattr(torch, "argsort", sort)
    monkeypatch.setattr(
        torch, "topk", Mock(side_effect=AssertionError("Compact pairs need no topk"))
    )
    keep = select_capacity_routes(scores, ids, num_experts=2, policy=policy)
    sort.assert_called_once()
    assert sort.call_args.kwargs["stable"]
    torch.testing.assert_close(keep, reference(scores, ids, 2, capacity))


@pytest.mark.parametrize("branchless_capacity_drop", [False, True])
@pytest.mark.parametrize("drop_by_group_capacity", [False, True])
@pytest.mark.parametrize("overloaded", [False, True])
def test_only_conditional_mode_reads_one_overload_scalar(
    branchless_capacity_drop, drop_by_group_capacity, overloaded
):
    class RecordScalarReads(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.num_scalar_reads = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                self.num_scalar_reads += 1
            return func(*args, **(kwargs or {}))

    ids = torch.tensor([[0, 1], [0, 1]])
    scores = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=1 if overloaded else 2,
        drop_by_group_capacity=drop_by_group_capacity,
        branchless_capacity_drop=branchless_capacity_drop,
    )
    with RecordScalarReads() as trace:
        select_capacity_routes(scores, ids, num_experts=2, policy=policy)
    assert trace.num_scalar_reads == (0 if branchless_capacity_drop else 1)


@pytest.mark.parametrize("branchless_capacity_drop", [False, True])
def test_float64_selection_preserves_distinct_scores(branchless_capacity_drop):
    # Both values round to the same float32 value; narrowing would keep token 0.
    scores = torch.tensor([[0.5], [0.5 + 1e-10]], dtype=torch.float64)
    ids = torch.zeros(2, 1, dtype=torch.long)
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=1,
        branchless_capacity_drop=branchless_capacity_drop,
    )
    keep = select_capacity_routes(scores, ids, num_experts=2, policy=policy)
    torch.testing.assert_close(keep, torch.tensor([[False], [True]]))


@pytest.mark.parametrize("branchless_capacity_drop", [False, True])
def test_signed_scores_and_inactive_slots(branchless_capacity_drop):
    scores = torch.tensor([[-0.9, 2.0], [-0.3, 0.0], [-0.5, -0.1]])
    ids = torch.tensor([[0, -1], [0, 1], [0, 1]])
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=1,
        branchless_capacity_drop=branchless_capacity_drop,
    )
    keep = select_capacity_routes(scores, ids, num_experts=2, policy=policy)
    torch.testing.assert_close(keep, reference(scores, ids, 2, 1))


@pytest.mark.parametrize("num_tokens,capacity", [(0, 0), (0, 3), (4, 8), (4, 16)])
@pytest.mark.parametrize("group_gate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_branchless_static_no_drop_skips_selection_after_collective(
    monkeypatch, num_tokens, capacity, group_gate, dtype
):
    class RejectScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("The static shortcut must not read tensor values")
            return func(*args, **(kwargs or {}))

    scores = torch.tensor(
        [[0.9, 0.5], [0.8, 0.0], [0.7, 0.0], [0.4, 0.6]], dtype=dtype
    )[:num_tokens]
    ids = torch.tensor([[0, -1], [0, 1], [1, 0], [-1, 1]])[:num_tokens]
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=max(1, 2 * capacity),
        drop_by_group_capacity=group_gate,
        branchless_capacity_drop=True,
    )
    sort = Mock(side_effect=AssertionError("A sufficient static quota needs no sort"))
    monkeypatch.setattr(torch, "argsort", sort)
    monkeypatch.setattr(torch, "topk", sort)
    # A peer can overload a destination even though this source cannot drop.
    other = torch.tensor([0, 0, 1000, 0])
    collective = Mock(side_effect=lambda tensor, reduce_op, group: tensor + other)
    monkeypatch.setattr(
        "torchtitan.models.common.moe.dispatch.capacity.funcol.all_reduce", collective
    )
    with RejectScalarRead():
        keep = select_capacity_routes(
            scores,
            ids,
            num_experts=4,
            policy=policy,
            ep_group=SimpleNamespace(size=lambda: 2),
        )
    torch.testing.assert_close(keep, (ids >= 0) & (scores != 0), rtol=0, atol=0)
    assert collective.call_count == int(group_gate)
    sort.assert_not_called()
