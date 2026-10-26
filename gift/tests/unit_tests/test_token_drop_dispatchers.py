"""Real Local/Gloo all-to-all transport and gradients after capacity selection."""

from datetime import timedelta
from itertools import product

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.config.moe import TokenDropConfig
from torchtitan.models.common.moe.dispatch import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    dispatcher_for,
)


def source_routes(num_tokens, rank, first_group=False):
    ids = torch.tensor([0, 1] if first_group else [0, 2]).expand(num_tokens, 2).clone()
    scores = (
        torch.arange(1, num_tokens * 2 + 1).reshape(num_tokens, 2).float() + rank
    ) / 128
    return ids, scores


def reference_keep(ids, scores, policy, source_sizes, rank, first_group=False):
    valid = (ids >= 0) & (scores != 0)
    if policy is None:
        return valid
    ep_size = len(source_sizes)
    quotas = [
        policy.source_capacity(
            num_tokens=n, top_k=2, num_experts=4, ep_size=ep_size, max_num_tokens=8
        )
        for n in source_sizes
    ]
    capacity = quotas[rank]
    # Independently compute the Foundry gate from the known global routing map.
    global_counts = torch.zeros(4, dtype=torch.long)
    for r, n in enumerate(source_sizes):
        other_ids, other_scores = source_routes(n, r, first_group)
        global_counts += torch.bincount(other_ids[other_scores != 0], minlength=4)
    if policy.drop_by_group_capacity:
        loads = global_counts.view(ep_size, 4 // ep_size).sum(1)
        if not bool((loads > capacity * 4).any()):
            return valid
    keep = torch.zeros_like(valid)
    for expert in range(4):
        positions = [
            (t, k)
            for t in range(ids.shape[0])
            for k in range(2)
            if valid[t, k] and ids[t, k] == expert
        ]
        positions.sort(key=lambda p: -float(scores.detach()[p]))
        for pos in positions[:capacity]:
            keep[pos] = True
    return keep


def check_transport(
    dispatcher, source_sizes, rank, policy, score_before, first_group=False
):
    n = source_sizes[rank]
    ids, scores = source_routes(n, rank, first_group)
    scores.requires_grad_()
    x = (torch.arange(n * 3).reshape(n, 3).float() / 16).requires_grad_()
    counts = torch.bincount(ids.reshape(-1), minlength=4)
    original_counts = counts.clone()
    observations = []
    dispatcher.dispatch_monitor = observations.append
    routed, sizes, metadata = dispatcher.dispatch(
        x, scores, ids, counts, score_before_experts=score_before
    )
    ep_size = len(source_sizes)
    factors = torch.arange(
        rank * (4 // ep_size) + 1, (rank + 1) * (4 // ep_size) + 1
    ).repeat_interleave(sizes)
    assert routed.shape[0] == sizes.sum()
    output = dispatcher.combine(routed * factors[:, None], metadata, x)
    output.sum().backward()
    keep = reference_keep(ids, scores, policy, source_sizes, rank, first_group)
    xr = x.detach().clone().requires_grad_()
    sr = scores.detach().clone().requires_grad_()
    expected = (xr[:, None, :] * (sr * keep * (ids + 1))[:, :, None]).sum(1)
    expected.sum().backward()
    for actual, ref in ((output, expected), (x.grad, xr.grad), (scores.grad, sr.grad)):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    torch.testing.assert_close(counts, original_counts, rtol=0, atol=0)
    observation = observations[0]
    assert observation.local_tokens_per_expert is counts
    if policy is not None:
        torch.testing.assert_close(observation.capacity_keep_mask, keep)
        assert observation.requested_topk_scores is scores


def policies(branchless, group_gate):
    return [
        None,
        TokenDropConfig(
            expert_capacity_factor=0.5,
            branchless_capacity_drop=branchless,
            drop_by_group_capacity=group_gate,
        ),
        TokenDropConfig(
            expert_capacity_factor=1,
            branchless_capacity_drop=branchless,
            drop_by_group_capacity=group_gate,
        ),
        TokenDropConfig(
            expert_capacity_factor=0.01,
            branchless_capacity_drop=branchless,
            drop_by_group_capacity=group_gate,
        ),
        TokenDropConfig(
            strategy="fixed_capacity",
            expert_capacity_tokens=3,
            branchless_capacity_drop=branchless,
            drop_by_group_capacity=group_gate,
        ),
        TokenDropConfig(
            strategy="peak_capacity_factor",
            expert_capacity_factor=0.5,
            branchless_capacity_drop=branchless,
            drop_by_group_capacity=group_gate,
        ),
    ]


@pytest.mark.parametrize(
    "branchless,group_gate,score_before", product((False, True), repeat=3)
)
def test_local_capacity_dispatch_and_gradient(branchless, group_gate, score_before):
    for n in (0, 1, 8):
        for policy in policies(branchless, group_gate):
            dispatcher = LocalTokenDispatcher(
                LocalTokenDispatcher.Config(token_drop=policy).with_capacity_context(
                    num_tokens=8, top_k=2, num_experts=4, ep_size=1
                )
            )
            check_transport(dispatcher, [n], 0, policy, score_before)


def all_to_all_worker(rank, store_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        for source_sizes in ([8, 8], [4, 8], [0, 8], [0, 0], [1, 1]):
            for branchless, group_gate, score_before, first_group in product(
                (False, True), repeat=4
            ):
                for policy in policies(branchless, group_gate):
                    dispatcher = AllToAllTokenDispatcher(
                        AllToAllTokenDispatcher.Config(
                            token_drop=policy
                        ).with_capacity_context(
                            num_tokens=8, top_k=2, num_experts=4, ep_size=2
                        )
                    )
                    dispatcher.wire_meshes(ep_mesh=mesh)
                    check_transport(
                        dispatcher,
                        source_sizes,
                        rank,
                        policy,
                        score_before,
                        first_group,
                    )
    finally:
        dist.destroy_process_group()


def test_all_to_all_with_real_collectives_empty_and_unequal_sources(tmp_path):
    mp.spawn(all_to_all_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)


def test_backend_substitution_preserves_capacity_policy():
    from types import SimpleNamespace

    policy = TokenDropConfig(
        strategy="peak_capacity_factor", expert_capacity_factor=0.5
    )
    dispatcher = (
        LocalTokenDispatcher.Config(token_drop=policy)
        .with_capacity_context(num_tokens=8, top_k=2, num_experts=4, ep_size=2)
        .build()
    )
    converted = dispatcher_for(dispatcher, ep_mesh=SimpleNamespace(size=lambda: 2))
    assert isinstance(converted, AllToAllTokenDispatcher)
    assert converted.token_drop == dispatcher.token_drop
    restored = dispatcher_for(converted, ep_mesh=None)
    assert isinstance(restored, LocalTokenDispatcher)
    assert restored.token_drop == dispatcher.token_drop
    assert restored._capacity_num_tokens == 8
