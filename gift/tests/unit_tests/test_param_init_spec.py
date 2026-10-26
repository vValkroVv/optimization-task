"""``InitSpec`` resolves to the ``nn.init`` call it names, placement specs to DTensor
placements, and a z-loss spec attaches."""

import math

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Partial, Replicate, Shard

from torchtitan.config.schema.models.common.attention.linear.param_init import (
    GatedDeltaALog,
    GatedDeltaDtBias,
)
from torchtitan.config.schema.models.common.regularization import HiddenZLoss
from torchtitan.config.schema.protocols import sharding as spec
from torchtitan.config.schema.protocols.module import (
    KaimingUniform,
    Normal,
    Ones,
    Skip,
    TruncNormal,
    XavierUniform,
    Zeros,
)
from torchtitan.models.common.norms import LayerNorm
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.module import resolve_param_init
from torchtitan.protocols.sharding import resolve_placement


def _init(fn, shape=(8, 16), seed=0):
    torch.manual_seed(seed)
    p = nn.Parameter(torch.empty(shape))
    fn(p)
    return p.detach()


def test_specs_match_their_nn_init_calls():
    cases = [
        (TruncNormal(std=0.02), lambda p: nn.init.trunc_normal_(p, std=0.02)),
        (
            TruncNormal(std=0.5, a=-1.5, b=1.5),
            lambda p: nn.init.trunc_normal_(p, std=0.5, a=-1.5, b=1.5),
        ),
        (
            TruncNormal(std=0.02, mean=0.1),
            lambda p: nn.init.trunc_normal_(p, mean=0.1, std=0.02),
        ),
        (Normal(std=0.006), lambda p: nn.init.normal_(p, std=0.006)),
        (Normal(std=0.1, mean=0.2), lambda p: nn.init.normal_(p, mean=0.2, std=0.1)),
        (Zeros(), nn.init.zeros_),
        (Ones(), nn.init.ones_),
        (XavierUniform(), nn.init.xavier_uniform_),
        (
            KaimingUniform(a=math.sqrt(5)),
            lambda p: nn.init.kaiming_uniform_(p, a=math.sqrt(5)),
        ),
    ]
    for s, reference in cases:
        torch.testing.assert_close(_init(resolve_param_init(s)), _init(reference))


def test_skip_leaves_the_parameter_alone():
    p = nn.Parameter(torch.full((4,), 7.0))
    resolve_param_init(Skip())(p)
    assert torch.equal(p, torch.full((4,), 7.0))


def test_family_specs_resolve_through_the_runtime_mirror():
    a_log = _init(resolve_param_init(GatedDeltaALog()), shape=(32,))
    assert a_log.min() >= math.log(1.0) and a_log.max() <= math.log(16.0)
    dt_bias = _init(resolve_param_init(GatedDeltaDtBias()), shape=(32,))
    assert torch.isfinite(dt_bias).all()


def test_callables_are_no_longer_initializers():
    linear = Linear.Config(
        in_features=4, out_features=4, param_init={"weight": nn.init.zeros_}
    ).build()
    with pytest.raises(TypeError):
        linear.init_states()


def test_fused_w13_spec_initializes_gate_and_up_slots():
    from torchtitan.config.schema.models.common.param_init import FusedW13Init

    # (E, 2, F, D): slot 0 of dim 1 is gate (w1), slot 1 is up (w3).
    p = nn.Parameter(torch.empty(2, 2, 3, 4))
    resolve_param_init(FusedW13Init(gate=Zeros(), up=Ones()))(p)
    assert torch.equal(p[:, 0], torch.zeros(2, 3, 4))
    assert torch.equal(p[:, 1], torch.ones(2, 3, 4))


def test_placement_specs_resolve_to_dtensor_placements():
    assert resolve_placement(spec.Shard(1)) == Shard(1)
    assert resolve_placement(spec.Replicate()) == Replicate()
    assert resolve_placement(spec.Partial()) == Partial()
    assert resolve_placement(spec.Partial("avg")) == Partial("avg")


def test_hidden_z_loss_spec_attaches_a_forward_hook():
    norm = LayerNorm.Config(
        normalized_shape=8, hidden_z_loss=HiddenZLoss(coef=1e-3)
    ).build()
    assert len(norm._forward_hooks) == 1
    silent = LayerNorm.Config(
        normalized_shape=8, hidden_z_loss=HiddenZLoss(coef=0.0)
    ).build()
    assert len(silent._forward_hooks) == 0
