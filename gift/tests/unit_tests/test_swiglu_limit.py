"""DeepSeek-style SwiGLU clipping (``swiglu_limit``) on dense FFNs and experts.

The bf16 paths are checked against the clamp spelled out here; the fp8 kernels
take the same limit and are covered in ``test_fp8_swiglu_clip_zloss.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.unit_tests.config_helpers import load_model_spec
from torchtitan.models.common.ffn import FeedForward
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.moe.dispatch import LocalTokenDispatcher
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.gift_moe._base.hf_config import build_hf_config

DIM, HIDDEN = 16, 32


def _clipped_swiglu(gate, up, limit):
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return F.silu(gate) * up


def _feed_forward(limit: float) -> FeedForward:
    ffn = FeedForward.Config(
        dim=DIM,
        hidden_dim=HIDDEN,
        w1=Linear.Config(in_features=DIM, out_features=HIDDEN, bias=False),
        w2=Linear.Config(in_features=HIDDEN, out_features=DIM, bias=False),
        w3=Linear.Config(in_features=DIM, out_features=HIDDEN, bias=False),
        swiglu_limit=limit,
    ).build()
    torch.manual_seed(0)
    for p in ffn.parameters():
        torch.nn.init.normal_(p)
    return ffn


@pytest.mark.parametrize("limit", [0.0, 0.5])
def test_feed_forward_clips_gate_and_up(limit):
    ffn = _feed_forward(limit)
    x = 3 * torch.randn(4, DIM)
    gate, up = x @ ffn.w1.weight.t(), x @ ffn.w3.weight.t()
    # The input is scaled so the limit actually bites.
    assert (gate > 0.5).any() and (up.abs() > 0.5).any()
    expected = _clipped_swiglu(gate, up, limit) @ ffn.w2.weight.t()
    torch.testing.assert_close(ffn(x), expected)


def test_negative_limit_is_a_config_error():
    with pytest.raises(ValueError, match="swiglu_limit"):
        _feed_forward(-1.0)
    with pytest.raises(ValueError, match="swiglu_limit"):
        GroupedExperts.Config(
            dim=DIM, hidden_dim=HIDDEN, num_experts=2, swiglu_limit=-1
        )


# Every SwiGLU of the model: the dense FFN (MTP heads included), the routed and
# the shared experts.
_DENSE_7 = {"feed_forward": {"swiglu_limit": 7.0}}
_MOE_7 = {
    "moe": {
        "experts": {"swiglu_limit": 7.0},
        "shared_experts": {"swiglu_limit": 7.0},
    }
}
_SWIGLU_LIMIT_7 = {
    "config": {
        "mtp": {"layers": [{"block": "mtp"}]},
        "blocks": {
            "gdn_dense": _DENSE_7,
            "gdn_moe": _MOE_7,
            "mla_moe": _MOE_7,
            "mtp": _DENSE_7,
        },
    }
}


def test_yaml_sets_the_limit_on_every_swiglu_and_exports_it():
    cfg = load_model_spec("gift_moe/gift_3_5/debugmodel.yaml", _SWIGLU_LIMIT_7).model
    ffns = [c for _f, c, _p, _a in cfg.traverse(FeedForward.Config)]
    experts = [c for _f, c, _p, _a in cfg.traverse(GroupedExperts.Config)]
    assert ffns and experts
    assert all(c.swiglu_limit == 7.0 for c in ffns + experts)
    assert build_hf_config(cfg)["swiglu_limit"] == 7.0


def test_default_is_off_and_exported_as_zero():
    cfg = load_model_spec("gift_moe/gift_3_5/debugmodel.yaml").model
    assert build_hf_config(cfg)["swiglu_limit"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch._grouped_mm")
@pytest.mark.parametrize("limit", [0.0, 0.5])
def test_bf16_experts_clip_and_count_gate_and_up(limit):
    num_experts, dim, hidden = 2, 128, 256
    experts = GroupedExperts(
        GroupedExperts.Config(
            dim=dim,
            hidden_dim=hidden,
            num_experts=num_experts,
            token_dispatcher=LocalTokenDispatcher.Config(),
            swiglu_limit=limit,
        )
    ).cuda()
    torch.manual_seed(0)
    for p in experts.parameters():
        torch.nn.init.normal_(p, std=0.1)
    experts.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64, device="cuda")
    counts = torch.tensor([16, 32], dtype=torch.int32, device="cuda")
    x = torch.randn(int(counts.sum()), dim, device="cuda")

    out = experts._experts_forward(x, counts)

    w13 = experts.w13_E2FD.bfloat16().float()
    w2 = experts.w2_EDF.bfloat16().float()
    rows = x.bfloat16().float().split(counts.tolist())
    expected = torch.cat(
        [
            _clipped_swiglu(r @ w13[e, 0].t(), r @ w13[e, 1].t(), limit) @ w2[e].t()
            for e, r in enumerate(rows)
        ]
    )
    # bf16 GEMMs and SwiGLU against an fp32 reference: compare by norm.
    rel_error = (out.float() - expected).norm() / expected.norm()
    assert rel_error < 5e-3, rel_error

    # The clamp counts, against the same grouped GEMM the experts run: the fp32
    # reference above could round a value at the limit to the other side.
    h = torch._grouped_mm(
        x.bfloat16(),
        experts.w13_E2FD.view(num_experts, 2 * hidden, dim)
        .bfloat16()
        .transpose(-2, -1),
        offs=torch.cumsum(counts, dim=0, dtype=torch.int32),
    )
    gate, up = h.chunk(2, dim=-1)
    if limit > 0:
        expected_counts = [
            int(gate.ge(limit).sum()),
            int((up.le(-limit) | up.ge(limit)).sum()),
            gate.numel(),
        ]
        assert 0 < expected_counts[0] and 0 < expected_counts[1]
    else:
        # Nothing is clamped, so the slot is never written.
        expected_counts = [0, 0, 0]
    assert experts.swiglu_clip_counts.tolist() == expected_counts
