"""Router choices and EP overload decisions survive selective recomputation."""

from copy import deepcopy
from datetime import timedelta
from itertools import product
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.config.moe import TokenDropConfig
from torchtitan.distributed.activation_checkpoint import (
    _apply_full_ac,
    _apply_op_sac,
    _get_save_ops,
)
from torchtitan.models.common.moe.dispatch.capacity import select_capacity_routes
from torchtitan.models.common.moe.router import TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import Linear
from torchtitan.ops.router import router_topk_indices


def _checkpoint(module, mode):
    config = ActivationCheckpointConfig(mode=mode, early_stop=False)
    if mode == "selective":
        return _apply_op_sac(module, config)
    if mode == "full":
        return _apply_full_ac(module, config)
    return module


class RouterSelections(nn.Module):
    def forward(self, scores):
        indices = router_topk_indices(scores, 2)
        selected = scores.gather(-1, indices)
        unrelated = torch.topk(scores, 1, dim=-1).values
        return selected.sin().sum() + unrelated.cos().sum()


@pytest.mark.parametrize(
    "mode,expected_topk_calls", [("none", 2), ("full", 4), ("selective", 3)]
)
def test_sac_saves_router_indices_without_saving_unrelated_topk(
    mode, expected_topk_calls
):
    scores = torch.arange(12).float().reshape(3, 4).requires_grad_()
    expected_scores = scores.detach().clone().requires_grad_()
    expected = RouterSelections()(expected_scores)
    expected.backward()
    model = _checkpoint(RouterSelections(), mode)
    with patch("torch.topk", wraps=torch.topk) as topk:
        result = model(scores)
        result.backward()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores.grad, expected_scores.grad, rtol=0, atol=0)
    assert topk.call_count == expected_topk_calls
    saved = _get_save_ops()
    assert torch.ops.aten.topk.default not in saved
    assert torch.ops.torchtitan.router_topk_indices.default in saved
    assert torch.ops._c10d_functional.all_reduce.default in saved


@pytest.mark.parametrize("sorted", [False, True])
def test_router_operator_fake_autograd_and_fullgraph(sorted):
    scores = torch.tensor(
        [[0.5, 0.5, 0.1, 0.9], [0.2, 0.4, 0.8, 0.6]], requires_grad=True
    )
    torch.library.opcheck(router_topk_indices, (scores, 2, sorted))

    def route(values):
        ids = router_topk_indices(values, 2, sorted)
        return values.gather(-1, ids)

    compiled = torch.compile(route, fullgraph=True, backend="aot_eager")
    result = compiled(scores)
    expected = scores.gather(-1, torch.topk(scores, 2, sorted=sorted).indices)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(result.sum(), scores)[0],
        torch.autograd.grad(expected.sum(), scores)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("mode", ["none", "full", "selective"])
@pytest.mark.parametrize("compiled", [False, True])
def test_router_preserves_choices_scores_and_gradients(forced, grouped, mode, compiled):
    torch.manual_seed(42)
    router = TokenChoiceTopKRouter.Config(
        gate=Linear.Config(in_features=4, out_features=8, bias=False),
        top_k=2,
        num_expert_groups=2 if grouped else None,
        num_limited_groups=1 if grouped else None,
        route_norm=True,
        route_scale=2.0,
        _debug_force_load_balance=forced,
    ).build()
    with torch.no_grad():
        router.gate.weight.copy_(torch.randn_like(router.gate.weight))
    reference = deepcopy(router)
    x = torch.randn(2, 3, 4, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    bias = torch.linspace(-0.2, 0.2, 8)

    def native_topk(scores, k, sorted=False):
        return torch.topk(scores, k, dim=-1, sorted=sorted).indices

    rng_state = torch.get_rng_state()
    with patch("torchtitan.models.common.moe.router.router_topk_indices", native_topk):
        expected = reference(xr, bias)
        expected_loss = expected[0].square().sum() + expected[2].square().sum()
        expected_loss.backward()
    model = _checkpoint(router, mode)
    if compiled:
        model = torch.compile(model, backend="aot_eager", fullgraph=True)
    torch.set_rng_state(rng_state)
    actual = model(x, bias)
    (actual[0].square().sum() + actual[2].square().sum()).backward()
    for result, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, wanted, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        router.gate.weight.grad, reference.gate.weight.grad, rtol=0, atol=0
    )


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("route_norm", [False, True])
def test_forced_routing_samples_all_experts_and_preserves_gate_weights(
    grouped, route_norm
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        router = TokenChoiceTopKRouter.Config(
            gate=Linear.Config(in_features=4, out_features=8, bias=False),
            top_k=2,
            num_expert_groups=2 if grouped else None,
            num_limited_groups=1 if grouped else None,
            route_norm=route_norm,
            route_scale=2.0,
            _debug_force_load_balance=True,
        ).build()
        with torch.no_grad():
            router.gate.weight.normal_()
        x = torch.randn(2, 4096, 4, requires_grad=True)
        rng_state = torch.get_rng_state()
        weights, ids, _, _ = router(x)
        torch.set_rng_state(rng_state)
        _, biased_ids, _, _ = router(x, torch.arange(8).float() * 1000)
        torch.testing.assert_close(ids, biased_ids, rtol=0, atol=0)
        assert (ids[..., 0] != ids[..., 1]).all()
        same_group = ids[..., 0] // 4 == ids[..., 1] // 4
        if grouped:
            assert same_group.all()
        else:
            assert not same_group.all()
        counts = torch.bincount(ids.flatten(), minlength=8).float()
        # Wide bound on a fixed sample; detects missing or strongly favored experts.
        assert ((counts / counts.mean() - 1).abs() < 0.1).all()

        expected = torch.sigmoid(router.gate(x.float())).gather(-1, ids)
        if route_norm:
            expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
        expected = expected * 2.0
        torch.testing.assert_close(weights, expected, rtol=0, atol=0)
        inputs = (x, router.gate.weight)
        actual_grads = torch.autograd.grad(weights.square().sum(), inputs)
        expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
        for actual, reference in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("preserve_rng_state", [False, True])
def test_full_ac_requires_rng_preservation_for_forced_router(
    forced, preserve_rng_state
):
    router = TokenChoiceTopKRouter.Config(
        gate=Linear.Config(in_features=4, out_features=8, bias=False),
        top_k=2,
        _debug_force_load_balance=forced,
    ).build()
    block = nn.Sequential(router)
    config = ActivationCheckpointConfig(
        mode="full", preserve_rng_state=preserve_rng_state
    )
    if forced and not preserve_rng_state:
        with pytest.raises(ValueError, match="requires preserve_rng_state=True"):
            _apply_full_ac(block, config)
    else:
        _apply_full_ac(block, config)


class CapacitySelection(nn.Module):
    def __init__(self, policy, group):
        super().__init__()
        self.policy = policy
        self.group = group

    def forward(self, x, scores, ids):
        keep = select_capacity_routes(
            scores, ids, num_experts=4, policy=self.policy, ep_group=self.group
        )
        return (x[:, None, :] * (scores * keep)[:, :, None]).sin().sum()


def _distributed_checks(rank, store_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        # Token dimensions stay sharded; an expert-sharded input is redistributed before selection.
        values = torch.arange(48).float().reshape(2, 4, 6)
        for placement, compiled in product(
            (Replicate(), Shard(0), Shard(1), Shard(2)), (False, True)
        ):
            scores = distribute_tensor(values, mesh, [placement])
            fn = (
                torch.compile(router_topk_indices, backend="aot_eager", fullgraph=True)
                if compiled
                else router_topk_indices
            )
            actual = fn(scores, 2)
            expected = torch.topk(values, 2, sorted=False).indices
            torch.testing.assert_close(actual.full_tensor(), expected, rtol=0, atol=0)

        for branchless, mode, compiled in product(
            (False, True), ("none", "full", "selective"), (False, True)
        ):
            policy = TokenDropConfig(
                strategy="fixed_capacity",
                expert_capacity_tokens=2,
                drop_by_group_capacity=True,
                branchless_capacity_drop=branchless,
            )
            module = CapacitySelection(policy, mesh.get_group())
            model = _checkpoint(module, mode)
            if compiled:
                model = torch.compile(model, backend="aot_eager", fullgraph=True)
            for empty_source, overloaded in (
                (False, False),
                (False, True),
                (True, True),
            ):
                if empty_source and compiled and mode == "full":
                    # Full AC can prune unused communication from the empty
                    # rank's backward. SAC must preserve the cached decision.
                    continue
                n = 0 if empty_source and rank == 0 else 4
                ids = torch.tensor(
                    [[0, 1]] * 4 if overloaded else [[0, 1], [2, 3], [-1, -1], [-1, -1]]
                )[:n]
                scores = torch.full((n, 2), 0.5, requires_grad=True)
                x = torch.arange(n * 3).reshape(n, 3).float().requires_grad_()
                xr, sr = (v.detach().clone().requires_grad_() for v in (x, scores))
                expected = module(xr, sr, ids)
                expected_grads = torch.autograd.grad(expected, (xr, sr))
                # Warm tracing independently of the measured forward/backward.
                if compiled:
                    torch.autograd.grad(model(x, scores, ids), (x, scores))
                group = mesh.get_group()
                sequence_before = group._get_sequence_number_for_group()
                result = model(x, scores, ids)
                grads = torch.autograd.grad(result, (x, scores))
                calls = group._get_sequence_number_for_group() - sequence_before
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                for actual, reference in zip(grads, expected_grads, strict=True):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                # The process-group sequence advances only for real collectives,
                # unlike profiler events emitted when SAC returns a cached tensor.
                assert calls == (2 if mode == "full" else 1), (
                    rank,
                    branchless,
                    mode,
                    compiled,
                    empty_source,
                    overloaded,
                    calls,
                )
    finally:
        dist.destroy_process_group()


def test_dtensor_routing_and_sac_issue_one_real_group_collective(tmp_path):
    mp.spawn(_distributed_checks, args=(str(tmp_path / "store"),), nprocs=2, join=True)
