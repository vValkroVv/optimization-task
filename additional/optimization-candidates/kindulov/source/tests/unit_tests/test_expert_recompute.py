"""Local expert recomputation preserves outputs, gradients and clip counters."""

import copy

import pytest
import torch

from torchtitan.models.common.moe.experts.gmm import GroupedExperts


def _grouped_mm(x, weights, *, offs):
    starts = [0] + offs.tolist()
    return torch.cat(
        [x[starts[i] : starts[i + 1]] @ weights[i] for i in range(len(weights))]
    )


@pytest.mark.parametrize("frozen_input", [False, True])
@pytest.mark.parametrize("limit", [0.0, 1.0])
def test_expert_recompute_exact(monkeypatch, frozen_input, limit):
    monkeypatch.setattr(torch, "_grouped_mm", _grouped_mm, raising=False)
    torch.manual_seed(42)
    baseline = GroupedExperts.Config(
        dim=8, hidden_dim=16, num_experts=3, swiglu_limit=limit
    ).build()
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.normal_(std=0.3)
    baseline.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64)
    recomputed = copy.deepcopy(baseline)
    recomputed.recompute = True
    x = torch.randn(7, 8, dtype=torch.bfloat16).requires_grad_(not frozen_input)
    other = x.detach().clone().requires_grad_(not frozen_input)
    sizes = torch.tensor([3, 0, 4])
    expected = baseline._experts_forward(x, sizes)
    actual = recomputed._experts_forward(other, sizes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    counts = recomputed.swiglu_clip_counts.clone()
    expected.float().square().sum().backward()
    actual.float().square().sum().backward()
    for p, q in zip(baseline.parameters(), recomputed.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    if not frozen_input:
        torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
    assert torch.equal(counts, recomputed.swiglu_clip_counts)
    assert torch.equal(baseline.swiglu_clip_counts, counts)


def test_expert_recompute_retains_fewer_intermediates(monkeypatch):
    monkeypatch.setattr(torch, "_grouped_mm", _grouped_mm, raising=False)
    module = GroupedExperts.Config(dim=8, hidden_dim=32, num_experts=3).build()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.1)
    x = torch.randn(64, 8, dtype=torch.bfloat16, requires_grad=True)
    sizes = torch.tensor([20, 0, 44])

    def retained(recompute):
        module.recompute = recompute
        saved = []

        def pack(tensor):
            saved.append(tensor.numel() * tensor.element_size())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = module._experts_forward(x, sizes)
        assert output.requires_grad
        return sum(saved)

    assert retained(True) < retained(False)


@pytest.mark.parametrize("coefs", [(0.03, 0.0), (0.0, 0.07), (0.03, 0.07)])
@pytest.mark.parametrize("frozen_input", [False, True])
def test_expert_recompute_preserves_z_losses(monkeypatch, coefs, frozen_input):
    from torchtitan.models.common.regularization import wire_aux_loss

    monkeypatch.setattr(torch, "_grouped_mm", _grouped_mm, raising=False)
    torch.manual_seed(42)
    baseline = GroupedExperts.Config(
        dim=8,
        hidden_dim=16,
        num_experts=3,
        swiglu_limit=1.0,
        w1_hidden_z_loss_coef=coefs[0],
        w3_hidden_z_loss_coef=coefs[1],
    ).build()
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.normal_(std=0.3)
    baseline.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64)
    replay = copy.deepcopy(baseline)
    replay.recompute = True
    states = [
        wire_aux_loss(module, device=torch.device("cpu"))
        for module in (baseline, replay)
    ]
    for state in states:
        state.scale.fill_(0.125)
    for sizes in (torch.tensor([3, 0, 4]), torch.tensor([0, 7, 0])):
        x = torch.randn(7, 8, dtype=torch.bfloat16).requires_grad_(not frozen_input)
        y = x.detach().clone().requires_grad_(not frozen_input)
        expected = baseline._experts_forward(x, sizes)
        actual = replay._experts_forward(y, sizes)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
        if not frozen_input:
            torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)
        for a, b in zip(baseline.parameters(), replay.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        torch.testing.assert_close(states[0].totals, states[1].totals, rtol=0, atol=0)
        assert torch.all(states[0].totals > 0)
        assert torch.equal(baseline.swiglu_clip_counts, replay.swiglu_clip_counts)


@pytest.mark.parametrize("score_before", [False, True])
def test_recompute_preserves_router_gradient_and_dispatch_count(
    monkeypatch, score_before
):
    from torchtitan.models.common.moe.losses import router_z_loss
    from torchtitan.models.common.regularization import AddAuxiliaryLoss

    monkeypatch.setattr(torch, "_grouped_mm", _grouped_mm, raising=False)
    torch.manual_seed(42)
    eager = GroupedExperts.Config(
        dim=8,
        hidden_dim=16,
        num_experts=3,
        score_before_experts=score_before,
    ).build()
    with torch.no_grad():
        for parameter in eager.parameters():
            parameter.normal_(std=0.2)
    replay = copy.deepcopy(eager)
    replay.recompute = True
    results = []
    for module in (eager, replay):
        calls = []
        dispatch = module.token_dispatcher.dispatch

        def counted(*args, **kwargs):
            calls.append(1)
            return dispatch(*args, **kwargs)

        monkeypatch.setattr(module.token_dispatcher, "dispatch", counted)
        x = (
            torch.linspace(-1, 1, 32)
            .reshape(1, 4, 8)
            .to(torch.bfloat16)
            .requires_grad_()
        )
        router_logits = torch.linspace(-1, 1, 12).reshape(1, 4, 3).requires_grad_()
        logits = AddAuxiliaryLoss.apply(
            router_logits, router_z_loss(router_logits, coef=0.03)
        )
        scores, ids = logits.softmax(-1).topk(2, dim=-1)
        counts = torch.bincount(ids.flatten(), minlength=3)
        output = module(x, scores, ids, counts)
        output.float().square().sum().backward()
        assert len(calls) == 1
        results.append((output.detach(), x.grad, router_logits.grad))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(eager.parameters(), replay.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped GEMM")
def test_expert_recompute_cuda_gradients():
    torch.manual_seed(42)
    eager = GroupedExperts.Config(dim=32, hidden_dim=64, num_experts=2).build().cuda()
    with torch.no_grad():
        for parameter in eager.parameters():
            parameter.normal_(std=0.1)
    replay = copy.deepcopy(eager)
    replay.recompute = True
    x = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    sizes = torch.tensor([32, 32], device="cuda", dtype=torch.int32)
    expected = eager._experts_forward(x, sizes)
    actual = replay._experts_forward(y, sizes)
    expected.float().square().sum().backward()
    actual.float().square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for a, b in zip(eager.parameters(), replay.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_expert_recompute_rejects_fp8():
    from torchtitan.config.schema.models.common.moe.dispatch.deepep import (
        DeepEPTokenDispatcherConfig,
    )

    config = GroupedExperts.Config(
        dim=8,
        hidden_dim=16,
        num_experts=2,
        recompute=True,
        token_dispatcher=DeepEPTokenDispatcherConfig(),
    )
    config.use_float8_grouped_gemm = True
    with pytest.raises(ValueError, match="BF16 grouped GEMM only"):
        config.build()


def test_compiled_expert_recompute_preserves_z_loss_and_counts(monkeypatch):
    from torchtitan.models.common.regularization import wire_aux_loss

    # A differentiable, capture-safe grouped-MM reference with two equal groups.
    # This verifies checkpoint/aux-loss compilation, not the CUDA GEMM kernel.
    def grouped_mm(x, weights, *, offs):
        return torch.bmm(x.reshape(2, 3, -1), weights).reshape(6, -1)

    monkeypatch.setattr(torch, "_grouped_mm", grouped_mm, raising=False)
    torch.manual_seed(42)
    eager = GroupedExperts.Config(
        dim=8,
        hidden_dim=16,
        num_experts=2,
        swiglu_limit=1.0,
        w1_hidden_z_loss_coef=0.03,
        w3_hidden_z_loss_coef=0.07,
    ).build()
    with torch.no_grad():
        for parameter in eager.parameters():
            parameter.normal_(std=0.3)
    eager.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64)
    replay = copy.deepcopy(eager)
    replay.recompute = True
    states = [
        wire_aux_loss(module, device=torch.device("cpu")) for module in (eager, replay)
    ]
    for state in states:
        state.scale.fill_(0.125)
    functions = [
        torch.compile(module._experts_forward, backend="aot_eager", fullgraph=True)
        for module in (eager, replay)
    ]
    for _ in range(2):
        x = torch.randn(6, 8, dtype=torch.bfloat16, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        sizes = torch.tensor([3, 3])
        expected = functions[0](x, sizes)
        actual = functions[1](y, sizes)
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        for a, b in zip(eager.parameters(), replay.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        torch.testing.assert_close(states[0].totals, states[1].totals, rtol=0, atol=0)
        assert torch.all(states[0].totals > 0)
        assert torch.equal(eager.swiglu_clip_counts, replay.swiglu_clip_counts)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_expert_recompute_requires_bool(value):
    with pytest.raises(ValueError, match="recompute must be a bool"):
        GroupedExperts.Config(dim=8, hidden_dim=16, num_experts=2, recompute=value)
