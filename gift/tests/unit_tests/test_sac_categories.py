"""Category SAC preserves per-block policy and route decisions."""

from copy import deepcopy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from torchtitan.config import ActivationCheckpointConfig as ACConfig
from torchtitan.distributed.activation_checkpoint import (
    _apply_ac_to_block,
    _sac_context_fn,
    apply_ac,
    parse_save_spec,
    resolve_ac_mode,
)
from torchtitan.ops.router import router_topk_indices
from torchtitan.protocols.blocks import Block
from torchtitan.protocols.module import Module, ModuleDict


def test_exact_crop_modes_first_match_wins():
    config = ACConfig(
        selective_save=[],
        module_modes=[
            "layers.[01]=selective",
            "layers.5=save:experts_up+comm",
            "layers.*=save:experts_up",
        ],
    )
    assert [resolve_ac_mode(f"layers.{i}", config) for i in range(8)] == [
        "selective",
        "selective",
        "save:experts_up",
        "save:experts_up",
        "save:experts_up",
        "save:experts_up+comm",
        "save:experts_up",
        "save:experts_up",
    ]
    assert config.selective_save == []


@pytest.mark.parametrize("spec", ["unknown", "mm+mm_half", "experts+experts_up"])
def test_invalid_save_sets_fail(spec):
    with pytest.raises(ValueError):
        parse_save_spec(spec)


def test_policy_keeps_only_up_experts_and_route_decisions():
    from torch.utils.checkpoint import CheckpointPolicy

    with patch(
        "torchtitan.distributed.activation_checkpoint.create_selective_checkpoint_contexts",
        side_effect=lambda policy: policy,
    ):
        policy = _sac_context_fn(("experts_up",), ())
    ctx = SimpleNamespace(is_recompute=False)
    up = (torch.empty(4, 16), torch.empty(2, 16, 8))
    down = (torch.empty(4, 8), torch.empty(2, 8, 16))
    assert (
        policy(ctx, torch.ops.aten._grouped_mm.default, *up)
        == CheckpointPolicy.MUST_SAVE
    )
    assert (
        policy(ctx, torch.ops.aten._grouped_mm.default, *down)
        == CheckpointPolicy.PREFER_RECOMPUTE
    )
    assert (
        policy(ctx, torch.ops.torchtitan.router_topk_indices.default)
        == CheckpointPolicy.MUST_SAVE
    )
    assert policy(ctx, torch.ops.aten.topk.default) == CheckpointPolicy.PREFER_RECOMPUTE
    assert policy(ctx, torch.ops.aten.max.default) == CheckpointPolicy.MUST_SAVE


class ProjectionBlock(Block):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.projection(x).sigmoid()


class TwoBlocks(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleDict({str(i): ProjectionBlock() for i in range(2)})

    def forward(self, x):
        for block in self.layers.values():
            x = block(x)
        return x


def test_module_modes_apply_and_preserve_loss_gradients_and_state():
    torch.manual_seed(42)
    reference = TwoBlocks()
    model = deepcopy(reference)
    apply_ac(model, ACConfig(module_modes=["layers.0=none", "layers.1=save:mm"]))
    assert type(model.layers["0"]) is ProjectionBlock
    assert hasattr(model.layers["1"], "_checkpoint_wrapped_module")
    assert model.state_dict().keys() == reference.state_dict().keys()
    x = torch.randn(3, 4)
    losses, gradients = [], []
    for candidate in (reference, model):
        value = x.clone().requires_grad_()
        loss = candidate(value).square().mean()
        loss.backward()
        losses.append(loss.detach())
        gradients.append([value.grad, *(p.grad for p in candidate.parameters())])
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for left, right in zip(*gradients):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_recompute_all_keeps_topk_routes():
    calls = []

    class Routed(torch.nn.Module):
        def forward(self, x):
            # A changed route on recompute would change the gradient.
            ids = router_topk_indices(x + len(calls) * x.new_tensor([2.0, 0.0]), 1)
            calls.append(ids.detach().clone())
            return x.gather(-1, ids).sigmoid()

    model = _apply_ac_to_block(Routed(), ACConfig(selective_save=[]), mode="selective")
    x = torch.tensor([[0.1, 0.9]], requires_grad=True)
    model(x).sum().backward()
    assert len(calls) == 2
    assert torch.equal(calls[0], calls[1])
    assert x.grad[0, 0] == 0


def test_compiled_policies_have_separate_caches():
    if getattr(sys.modules.get("triton"), "__version__", None) == "0.0.0":
        pytest.skip(
            "Unit-test Triton stub blocks Dynamo; run this module with --noconftest"
        )
    import torchtitan.distributed.compile  # noqa: F401

    compilations = 0

    def backend(graph, inputs):
        nonlocal compilations
        compilations += 1
        return torch._dynamo.lookup_backend("aot_eager")(graph, inputs)

    def run(mode):
        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16, bias=False),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, 8, bias=False),
        )
        model = _apply_ac_to_block(model, ACConfig(selective_save=[]), mode=mode)
        model.compile(backend=backend, fullgraph=True)
        x = torch.randn(4, 8, requires_grad=True)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: (saved.append(tuple(tensor.shape)), tensor)[1], lambda t: t
        ):
            output = model(x)
        output.square().mean().backward()
        return saved, output.detach(), [x.grad, *(p.grad for p in model.parameters())]

    try:
        for modes in (("selective", "save:mm"), ("save:mm", "selective")):
            torch._dynamo.reset()
            first, second = (run(mode) for mode in modes)
            assert first[0] != second[0]
            torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
            for left, right in zip(first[2], second[2]):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            before = compilations
            assert run(modes[-1])[0] == second[0]
            assert compilations == before
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"selective_save": ["unknown"]},
        {"module_modes": ["layers.*=invalid"]},
        {"module_modes": ["layers.*"]},
        {"module_modes": ["=full"]},
        {"mode": "none", "module_modes": ["layers.*=full"]},
        {"mode": "memory_budget", "module_modes": ["layers.*=full"]},
    ],
)
def test_invalid_config_fails_before_model_wrapping(kwargs):
    with pytest.raises(ValueError):
        ACConfig(**kwargs)
