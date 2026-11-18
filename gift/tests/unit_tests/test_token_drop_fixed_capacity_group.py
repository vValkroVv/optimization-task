"""The llm-foundry setup ``dispatcher_balancing_strategy: fixed_capacity``,
``expert_capacity_tokens: N``, ``drop_by_group_capacity: true``.

In torchtitan that is ``TokenDropConfig(strategy="fixed_capacity",
expert_capacity_tokens=N, drop_by_group_capacity=True)``: ``N`` is the quota
of one expert over the whole EP group (``N // ep_size`` per source), and the
gate clips only when some EP rank's total received load exceeds
``local quota * ep_size * num_local_experts``. These tests pin that semantic at
the policy, dispatcher and monitoring levels, on CPU.
"""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchtitan.callbacks.sources.dispatcher_source import (
    DispatcherSource,
    MaxVioWindow,
)
from torchtitan.config.moe import TokenDropConfig
from torchtitan.models.common.moe.dispatch.base import BaseEPTokenDispatcher
from torchtitan.models.common.moe.dispatch import capacity
from torchtitan.models.common.moe.dispatch.capacity import select_capacity_routes
from torchtitan.models.common.moe.dispatch.deepep import DeepEPTokenDispatcher
from torchtitan.protocols.moe_monitoring import DispatchObservation
from tests.unit_tests.callbacks.trainer_stub import snapshot


def fixed_group(expert_capacity_tokens: int) -> TokenDropConfig:
    return TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=expert_capacity_tokens,
        drop_by_group_capacity=True,
    )


def _group(size: int):
    return SimpleNamespace(size=lambda: size)


def _patch_all_reduce(monkeypatch, other_source_counts: torch.Tensor) -> Mock:
    """Stand in for EP SUM while checking the destination-group payload."""
    observations = []

    def all_reduce(group_loads, reduce_op, group):
        assert reduce_op == "sum"
        assert group_loads.dtype == torch.int64
        assert group_loads.shape == (group.size(),)
        assert other_source_counts.numel() % group.size() == 0
        other_group_loads = other_source_counts.view(group.size(), -1).sum(1)
        global_group_loads = group_loads + other_group_loads
        observations.append((group_loads.clone(), global_group_loads.clone()))
        return global_group_loads

    collective = Mock(side_effect=all_reduce)
    collective.observations = observations
    monkeypatch.setattr(
        "torchtitan.models.common.moe.dispatch.capacity.funcol.all_reduce", collective
    )
    return collective


@pytest.mark.parametrize(
    "route_ids,scores,other_source_counts,expected_keep",
    [
        ([[0], [1], [2]], [0.9, 0.8, 0.7], [0, 0, 0, 0], [True, True, True]),
        (
            [[0], [0], [0], [0], [0], [0], [1], [2]],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0, 0, 0, 0],
            [True, True, True, True, True, True, True, True],
        ),
        (
            [[0], [0], [0], [0], [1], [1], [1], [1]],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [9, 0, 0, 0],
            [True, True, True, True, True, True, True, True],
        ),
        (
            [[0], [0], [0], [0], [0], [0], [1], [2]],
            [0.5] * 8,
            [10, 0, 0, 0],
            [True, True, True, True, False, False, True, True],
        ),
        (
            [[0], [0], [0], [0], [0], [0], [1], [2]],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0, 0, 20, 0],
            [True, True, True, True, False, False, True, True],
        ),
    ],
    ids=[
        "no-overload",
        "local-only",
        "global-only",
        "tied-overload",
        "other-group-overload",
    ],
)
def test_group_counter_reduction_matches_original_gate_and_mask(
    monkeypatch, route_ids, scores, other_source_counts, expected_keep
):
    """Moving the expert sum before SUM preserves counts, gate, and stable mask."""
    ids = torch.tensor(route_ids, dtype=torch.int64)
    scores = torch.tensor(scores, dtype=torch.float32).view_as(ids)
    other_source_counts = torch.tensor(other_source_counts, dtype=torch.int64)
    collective = _patch_all_reduce(monkeypatch, other_source_counts)
    sorted_calls = []
    sorted_capacity_mask = capacity._sorted_capacity_mask

    def record_selection(*args, **kwargs):
        sorted_calls.append(True)
        return sorted_capacity_mask(*args, **kwargs)

    monkeypatch.setattr(capacity, "_sorted_capacity_mask", record_selection)
    keep = select_capacity_routes(
        scores, ids, num_experts=4, policy=fixed_group(8), ep_group=_group(2)
    )

    capacity_per_expert = 4
    local_counts = torch.bincount(ids.flatten(), minlength=4)
    original_global_counts = local_counts + other_source_counts
    expected_local_group_loads = local_counts.view(2, 2).sum(1)
    expected_global_group_loads = original_global_counts.view(2, 2).sum(1)
    expected_needs_drop = bool(
        (local_counts > capacity_per_expert).any()
        and (expected_global_group_loads > capacity_per_expert * 4).any()
    )

    collective.assert_called_once()
    actual_local_group_loads, actual_global_group_loads = collective.observations[0]
    torch.testing.assert_close(
        actual_local_group_loads, expected_local_group_loads, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_global_group_loads, expected_global_group_loads, rtol=0, atol=0
    )
    assert bool(sorted_calls) is expected_needs_drop
    torch.testing.assert_close(
        keep, torch.tensor(expected_keep, dtype=torch.bool).view_as(ids), rtol=0, atol=0
    )


def test_ep_all_reduce_receives_eight_int64_group_counts_for_128_experts(monkeypatch):
    num_experts = 128
    ep_size = 8
    collective = _patch_all_reduce(
        monkeypatch, torch.zeros(num_experts, dtype=torch.int64)
    )
    keep = select_capacity_routes(
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int64),
        num_experts=num_experts,
        policy=fixed_group(2048),
        ep_group=_group(ep_size),
    )

    payload = collective.call_args.args[0]
    assert payload.dtype == torch.int64
    assert payload.numel() == ep_size
    assert payload.numel() != num_experts
    assert keep.all()


def test_config_round_trip_and_per_source_quota():
    policy = fixed_group(2048)
    assert TokenDropConfig(**asdict(policy)) == policy
    assert policy.drop_by_group_capacity
    # The quota is over the EP group: N // ep_size per source, whatever T is.
    for ep_size, expected in ((1, 2048), (2, 1024), (4, 512), (8, 256)):
        assert (
            policy.source_capacity(
                num_tokens=8192, top_k=4, num_experts=32, ep_size=ep_size
            )
            == expected
        )
    # Floor division: an odd quota loses the remainder per source.
    assert (
        fixed_group(7).source_capacity(num_tokens=8, top_k=2, num_experts=4, ep_size=2)
        == 3
    )
    with pytest.raises(ValueError, match="requires fixed_capacity"):
        TokenDropConfig(
            strategy="capacity_factor",
            expert_capacity_factor=1.0,
            expert_capacity_tokens=2048,
            drop_by_group_capacity=True,
        )


def test_gate_skips_clipping_while_every_destination_is_within_budget(monkeypatch):
    """A single overflowing expert does not trigger drops: only a rank's total does."""
    # 4 experts over EP2: destination 0 owns experts 0-1, destination 1 owns 2-3.
    # N=8 -> quota 4 per source and expert; destination budget = (4+4) * 2 = 16.
    ids = torch.tensor([[0], [0], [0], [0], [0], [0], [1], [2]])
    scores = torch.tensor([[0.9], [0.8], [0.7], [0.6], [0.5], [0.4], [0.3], [0.2]])
    # The other source sends nothing: destination 0 receives 7 < 16.
    collective = _patch_all_reduce(monkeypatch, torch.tensor([0, 0, 0, 0]))
    keep = select_capacity_routes(
        scores, ids, num_experts=4, policy=fixed_group(8), ep_group=_group(2)
    )
    assert keep.all(), "expert 0 exceeds its quota (6 > 4) but the gate is closed"
    collective.assert_called_once()


def test_gate_clips_every_expert_once_a_destination_is_over_budget(monkeypatch):
    ids = torch.tensor([[0], [0], [0], [0], [0], [0], [1], [2]])
    scores = torch.tensor([[0.9], [0.8], [0.7], [0.6], [0.5], [0.4], [0.3], [0.2]])
    # The other source pushes destination 0 over budget: 7 + 10 = 17 > 16.
    _patch_all_reduce(monkeypatch, torch.tensor([10, 0, 0, 0]))
    keep = select_capacity_routes(
        scores, ids, num_experts=4, policy=fixed_group(8), ep_group=_group(2)
    )
    # Expert 0 keeps its 4 best routes; experts 1 and 2 are under quota anyway.
    expected = torch.tensor([[True]] * 4 + [[False]] * 2 + [[True]] * 2)
    torch.testing.assert_close(keep, expected)


def test_gate_matches_strict_clipping_when_the_overloaded_rank_is_elsewhere(
    monkeypatch,
):
    """Overload on destination 1 clips this source's expert 0 too (llm-foundry
    semantic: the gate is global, the clipping is per local expert column)."""
    ids = torch.tensor([[0], [0], [0], [0], [0], [0], [1], [2]])
    scores = torch.tensor([[0.9], [0.8], [0.7], [0.6], [0.5], [0.4], [0.3], [0.2]])
    _patch_all_reduce(monkeypatch, torch.tensor([0, 0, 20, 0]))
    gated = select_capacity_routes(
        scores, ids, num_experts=4, policy=fixed_group(8), ep_group=_group(2)
    )
    strict = select_capacity_routes(
        scores,
        ids,
        num_experts=4,
        policy=TokenDropConfig(strategy="fixed_capacity", expert_capacity_tokens=8),
        ep_group=_group(2),
    )
    torch.testing.assert_close(gated, strict)
    assert int(gated.sum()) == 6


def test_dispatcher_applies_fixed_group_policy_before_transport(monkeypatch):
    dispatcher = object.__new__(DeepEPTokenDispatcher)
    # Quota 2 per source over EP2.
    BaseEPTokenDispatcher.__init__(
        dispatcher, DeepEPTokenDispatcher.Config(token_drop=fixed_group(4))
    )
    dispatcher.ep_mesh = SimpleNamespace(get_group=lambda: _group(2))
    observed = []
    dispatcher.dispatch_monitor = observed.append
    x = torch.randn(4, 3)
    scores = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    demand = torch.tensor([4, 4])
    dispatcher._dispatch_tokens = Mock(return_value=(x, torch.tensor([2]), object()))
    # Destination 0 (expert 0) receives 4 against a budget of (2+2)*1 = 4:
    # not over budget, so nothing is dropped.
    _patch_all_reduce(monkeypatch, torch.tensor([0, 0]))
    dispatcher.dispatch(x, scores, ids, demand, score_before_experts=False)
    args = dispatcher._dispatch_tokens.call_args.args
    torch.testing.assert_close(args[1], ids)
    torch.testing.assert_close(args[2], scores)
    assert observed[-1].capacity_keep_mask.all()

    # One more route from the other source tips destination 0 over budget:
    # every expert is clipped to its quota of 2, best scores first.
    _patch_all_reduce(monkeypatch, torch.tensor([1, 0]))
    dispatcher.dispatch(x, scores, ids, demand, score_before_experts=False)
    args = dispatcher._dispatch_tokens.call_args.args
    torch.testing.assert_close(
        args[1], torch.tensor([[0, -1], [0, -1], [-1, 1], [-1, 1]])
    )
    torch.testing.assert_close(
        args[2], torch.tensor([[0.9, 0.0], [0.8, 0.0], [0.0, 0.3], [0.0, 0.4]])
    )
    keep = observed[-1].capacity_keep_mask
    assert int(keep.sum()) == 4
    assert observed[-1].local_tokens_per_expert is demand, "router demand unchanged"


def test_monitoring_reports_gated_fixed_capacity_drops():
    """The metrics of the run above: 8 requested, 4 retained, no token lost."""
    source = DispatcherSource(DispatcherSource.Config())
    source.active = True
    source._needs_sums = {MaxVioWindow.global_}
    unit = SimpleNamespace(
        name="layers.1.moe", dispatcher=object(), router=SimpleNamespace(num_experts=2)
    )
    source.units = [unit]
    accum = source._allocate(unit, ep_size=2, device="cpu")
    source._accum[unit.name] = accum
    state = snapshot(
        SimpleNamespace(
            parallel_dims=SimpleNamespace(get_optional_mesh=lambda name: None)
        )
    )
    source.batch_start(state)
    scores = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    keep = torch.tensor([[True, False], [True, False], [False, True], [False, True]])
    source._observe(accum)(
        DispatchObservation(
            local_tokens_per_expert=torch.tensor([4, 4]),
            global_tokens_per_local_expert=torch.tensor([2]),
            routed_input=torch.empty(2, 3),
            topk_expert_ids_TK=torch.tensor([[0, -1], [0, -1], [-1, 1], [-1, 1]]),
            requested_topk_scores=scores,
            capacity_keep_mask=keep,
        )
    )
    source.batch_end(state)
    metrics = {r.name: r.value for r in source.drain()}
    prefix = "moe/token_dispatcher/layers.1.moe/token_drop/"
    assert metrics[prefix + "requested_assignments"] == 8
    assert metrics[prefix + "retained_assignments"] == 4
    assert metrics[prefix + "assignment_drop_fraction"] == 0.5
    assert metrics[prefix + "fully_dropped_token_fraction"] == 0.0
    dropped_mass = 0.1 + 0.2 + 0.7 + 0.6
    assert metrics[prefix + "dropped_score_mass_fraction"] == pytest.approx(
        dropped_mass / 4.0
    )


def _fixed_group_worker(rank, store_path):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=2
    )
    try:
        # 4 experts over EP2; N=8 -> quota 4 per source; destination budget 16.
        policy = fixed_group(8)
        # Case 1: rank 0 sends 6 to expert 0, rank 1 sends 6 to expert 1: destination
        # 0 receives 12 <= 16, so both ranks keep everything despite the overflow.
        ids = torch.full((6, 1), rank, dtype=torch.long)
        keep = select_capacity_routes(
            torch.rand(6, 1) + 0.1,
            ids,
            num_experts=4,
            policy=policy,
            ep_group=dist.group.WORLD,
        )
        assert int(keep.sum()) == 6
        # Case 2: rank 0 sends 9 to expert 0 and rank 1 sends 9 to expert 1:
        # 18 > 16, so every source clips to its quota of 4.
        ids = torch.full((9, 1), rank, dtype=torch.long)
        keep = select_capacity_routes(
            torch.rand(9, 1) + 0.1,
            ids,
            num_experts=4,
            policy=policy,
            ep_group=dist.group.WORLD,
        )
        assert int(keep.sum()) == 4
        # Case 3: an empty source (the first-group router seen from rank 1)
        # still takes part in the collective and keeps nothing.
        n = 20 if rank == 0 else 0
        keep = select_capacity_routes(
            torch.rand(n, 1) + 0.1,
            torch.zeros(n, 1, dtype=torch.long),
            num_experts=4,
            policy=policy,
            ep_group=dist.group.WORLD,
        )
        assert int(keep.sum()) == (4 if rank == 0 else 0)
    finally:
        dist.destroy_process_group()


def test_fixed_group_with_real_collective(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_fixed_group_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)
