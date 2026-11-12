"""Router z-loss + hidden z-loss regularizers (CPU).

Covers the router z-loss formula, the grad-only splice (forward value unchanged,
gradient injected), the no-op-when-off invariant, and that the router surfaces
pre-activation logits.
"""

from dataclasses import dataclass

import pytest
import torch

from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.schema.models.common.moe.const import ScoreFunc
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.models.common.attention import MLA, MaskType, ScaledDotProductAttention
from torchtitan.models.common.decoder_blocks.transformer_block import (
    ResidualTransformerBlock,
)
from torchtitan.models.common.embeddings.rope import ComplexRoPE
from torchtitan.models.common.ffn import FeedForward
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.moe.experts.gmm import _routed_row_mask
from torchtitan.models.common.moe.losses import router_z_loss
from torchtitan.models.common.moe.router import TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.norms import RMSNorm
from torchtitan.models.common.regularization import (
    declare_fused_aux_loss,
    fused_aux_loss_state,
    HiddenZLoss,
    set_hidden_z_loss,
    suppress_aux_recording,
    wire_aux_loss,
)


def test_router_z_loss_formula():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 8)
    coef = 1e-3
    expected = (coef * torch.logsumexp(logits.float(), dim=-1).square()).mean()
    assert torch.allclose(router_z_loss(logits, coef=coef), expected)


def test_router_returns_pre_activation_logits():
    router = TokenChoiceTopKRouter.Config(
        gate=Linear.Config(
            in_features=16,
            out_features=8,
            bias=False,
            param_init={},
        ),
        top_k=3,
        score_func=ScoreFunc.SOFTMAX,
    ).build()
    router.eval()
    x = torch.randn(2, 5, 16)
    out = router(x)
    assert len(out) == 4, "router must return a 4-tuple including logits"
    _topk_scores, _topk_ids, scores_BLE, logits_BLE = out
    # logits are the pre-activation gate output (float32), scores are post-softmax.
    assert logits_BLE.dtype == torch.float32
    assert torch.allclose(torch.softmax(logits_BLE, dim=-1), scores_BLE, atol=1e-5)
    assert not torch.allclose(logits_BLE, scores_BLE)


# --- block-level: the hidden z-loss hooks are wired into ResidualTransformerBlock ---

_DIM = 64
_N_HEADS = 4
_QK_NOPE = 16
_QK_ROPE = 16
_V_HEAD = 16
_KV_LORA = 32
_SEQ_LEN = 8


def _block_config(
    coef: float, *, q_lora_rank: int = 0
) -> ResidualTransformerBlock.Config:
    qk_head_dim = _QK_NOPE + _QK_ROPE
    mla_cfg = MLA.Config(
        dim=_DIM,
        n_heads=_N_HEADS,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=_KV_LORA,
        qk_nope_head_dim=_QK_NOPE,
        qk_rope_head_dim=_QK_ROPE,
        v_head_dim=_V_HEAD,
        wq=(
            Linear.Config(in_features=_DIM, out_features=_N_HEADS * qk_head_dim)
            if q_lora_rank == 0
            else None
        ),
        kv_norm=RMSNorm.Config(normalized_shape=_KV_LORA),
        wkv_a=Linear.Config(in_features=_DIM, out_features=_KV_LORA + _QK_ROPE),
        wkv_b=Linear.Config(
            in_features=_KV_LORA, out_features=_N_HEADS * (_QK_NOPE + _V_HEAD)
        ),
        wo=Linear.Config(in_features=_N_HEADS * _V_HEAD, out_features=_DIM),
        inner_attention=ScaledDotProductAttention.Config(),
        mask_type=MaskType.CAUSAL,
    )
    for projection in (
        mla_cfg.wq,
        mla_cfg.wq_a,
        mla_cfg.wq_b,
        mla_cfg.wkv_a,
        mla_cfg.wkv_b,
        mla_cfg.wo,
        mla_cfg.q_norm,
        mla_cfg.kv_norm,
    ):
        if projection is not None:
            projection.param_init = {}
    config = ResidualTransformerBlock.Config(
        attention=mla_cfg,
        attention_norm=RMSNorm.Config(normalized_shape=_DIM),
        ffn_norm=RMSNorm.Config(normalized_shape=_DIM),
        feed_forward=FeedForward.Config(
            dim=_DIM,
            hidden_dim=128,
            w1=LinearConfig(param_init={}),
            w2=LinearConfig(param_init={}),
            w3=LinearConfig(param_init={}),
        ),
    )
    if coef:
        set_hidden_z_loss(config, {"attention": coef, "feed_forward": coef})
    return config


def test_block_hidden_z_loss_is_grad_only():
    torch.manual_seed(0)
    block = _block_config(1e-2).build()
    rope = ComplexRoPE.Config(dim=_QK_ROPE, max_seq_len=_SEQ_LEN).build().cache
    x = torch.randn(2, _SEQ_LEN, _DIM)

    # Eval => hooks are no-ops. Train => hooks splice grad but leave forward value
    # unchanged. The two forward values must be bit-identical (no dropout in MLA).
    block.eval()
    with torch.no_grad():
        y_eval = block(x, rope)

    block.train()
    x_on = x.clone().requires_grad_(True)
    y_on = block(x_on, rope)
    assert torch.equal(y_eval, y_on)
    y_on.sum().backward()
    grad_on = x_on.grad.clone()
    y_on_state = block.state_dict()

    # Same weights, no z-loss configured: forward identical, but the injected
    # gradient is gone, so the input gradient must differ.
    block = _block_config(0.0).build()
    block.load_state_dict(y_on_state)
    x_off = x.clone().requires_grad_(True)
    y_off = block(x_off, rope)
    assert torch.equal(y_eval, y_off)
    y_off.sum().backward()
    assert not torch.allclose(grad_on, x_off.grad)


def test_expert_z_loss_is_grad_only_and_skips_padding():
    """gate/up z-loss on the fused w13 output: value unchanged, padding untouched."""
    torch.manual_seed(0)
    coefs = (1e-2, 3e-2)
    experts = GroupedExperts.Config(
        dim=8,
        hidden_dim=4,
        num_experts=2,
        w1_hidden_z_loss_coef=coefs[0],
        w3_hidden_z_loss_coef=coefs[1],
    ).build()
    experts.train()
    group_sizes = torch.tensor([3, 3])
    num_tokens = torch.tensor([2, 3])  # row 2 is padding
    h = torch.randn(6, 8, dtype=torch.float64)

    x = h.clone().requires_grad_(True)
    y = experts._apply_hidden_z_loss(
        x, group_sizes_E=group_sizes, num_tokens_E=num_tokens
    )
    assert torch.equal(y, h)
    y.sum().backward()

    mask = torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.float64)
    x_ref = h.clone().requires_grad_(True)
    loss = sum(
        coef * (torch.logsumexp(half.abs(), dim=-1).square() * mask).sum() / mask.sum()
        for coef, half in zip(coefs, x_ref.chunk(2, dim=-1))
    )
    (x_ref.sum() + loss).backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(x.grad[2], torch.ones(8, dtype=torch.float64))


def test_routed_row_mask_skips_group_padding_and_tail():
    # Groups of 4, 4, 0 and 4 rows (tokens 3, 0, 0, 4) and a 2-row tail.
    mask = _routed_row_mask(
        14, torch.tensor([4, 4, 0, 4], dtype=torch.int32), torch.tensor([3, 0, 0, 4])
    )
    expected = [1, 1, 1, 0] + [0, 0, 0, 0] + [1, 1, 1, 1] + [0, 0]
    assert mask.tolist() == [bool(v) for v in expected]


def test_expert_projection_z_loss_patterns():
    """Routed-expert gate/up are addressed as ``<experts>.w1`` / ``<experts>.w3``."""

    @dataclass
    class _Holder:
        experts: GroupedExperts.Config

    def holder():
        return _Holder(
            experts=GroupedExperts.Config(dim=8, hidden_dim=16, num_experts=2)
        )

    cfg = holder()
    set_hidden_z_loss(cfg, {"experts.w1": 1e-3, "experts.w3": 2e-3})
    assert (cfg.experts.w1_hidden_z_loss_coef, cfg.experts.w3_hidden_z_loss_coef) == (
        1e-3,
        2e-3,
    )
    # The experts' own output hook is a separate, untouched knob.
    assert cfg.experts.hidden_z_loss is None

    with pytest.raises(ValueError, match="match no module"):
        set_hidden_z_loss(holder(), {"experts.w2": 1e-3})


# --- the w1/w3 z-loss of the dense FFN: a hook, or the fused fp8 SwiGLU ---


def _ffn_with_z_loss(*, fp8: bool, fused: bool, w3_on: str = "output") -> FeedForward:
    """An FFN with an output z-loss on w1, and a z-loss on the ``w3_on`` of w3."""
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    linear = FP8Linear.Config if fp8 else LinearConfig
    config = FeedForward.Config(
        dim=16,
        hidden_dim=32,
        w1=linear(param_init={}, hidden_z_loss=HiddenZLoss(coef=1e-2)),
        w2=linear(param_init={}),
        w3=linear(param_init={}, hidden_z_loss=HiddenZLoss(coef=2e-2, on=w3_on)),
    )
    config.dense_fused_swiglu_quant = fused
    return FeedForward(config)


@pytest.mark.parametrize(
    "fp8, fused", [(False, True), (True, False)], ids=["bf16", "fp8_bf16_swiglu"]
)
def test_a_bf16_swiglu_keeps_the_z_loss_hooks(fp8, fused):
    ffn = _ffn_with_z_loss(fp8=fp8, fused=fused)
    assert ffn._hidden_z_loss_coefs == (0.0, 0.0)
    for projection in (ffn.w1, ffn.w3):
        assert len(projection._forward_hooks) == 1
        assert projection._aux_loss_kinds == ("hidden_z_loss",)


def test_the_fused_fp8_swiglu_takes_the_output_z_loss_from_the_hooks():
    """The kernel applies the output z-loss: no hook, a declared splice, and the
    coefficient. An input z-loss is on the FFN input, thus it keeps its hook."""
    ffn = _ffn_with_z_loss(fp8=True, fused=True)
    assert ffn._hidden_z_loss_coefs == (1e-2, 2e-2)
    for projection in (ffn.w1, ffn.w3):
        assert not projection._forward_hooks
        assert not projection._forward_pre_hooks
        assert projection._aux_loss_kinds == ("hidden_z_loss",)

    ffn = _ffn_with_z_loss(fp8=True, fused=True, w3_on="input")
    assert ffn._hidden_z_loss_coefs == (1e-2, 0.0)
    assert len(ffn.w3._forward_pre_hooks) == 1


def _fp8_ffn_config(
    *, shared_quantizer: bool, fused: bool, **z_loss: HiddenZLoss
) -> FeedForward.Config:
    """The config of an fp8 FFN, with a z-loss spec for each named projection."""
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    config = FeedForward.Config(
        dim=16,
        hidden_dim=32,
        **{
            name: FP8Linear.Config(param_init={}, hidden_z_loss=z_loss.get(name))
            for name in ("w1", "w2", "w3")
        },
    )
    config.use_shared_quantizer = shared_quantizer
    config.dense_fused_swiglu_quant = fused
    return config


@pytest.mark.parametrize(
    "name, shared_quantizer, fused",
    [("w1", True, True), ("w3", True, False), ("w2", False, True)],
    ids=["w1_shared_quantizer", "w3_shared_quantizer", "w2_fused"],
)
def test_an_input_z_loss_on_a_packed_input_is_refused(name, shared_quantizer, fused):
    """The hook reads bf16, thus it cannot read a packed fp8 input. The error
    comes at the build, not at the first forward."""
    z_loss = {name: HiddenZLoss(coef=1e-2, on="input")}
    with pytest.raises(ValueError, match=f"{name} gets a packed fp8 input"):
        FeedForward(
            _fp8_ffn_config(shared_quantizer=shared_quantizer, fused=fused, **z_loss)
        )


def test_an_input_z_loss_at_zero_or_on_a_bf16_input_is_permitted():
    """A coefficient of zero attaches nothing. Without the packed operand, the
    input is bf16, thus the hook stays. The state dict does not change."""
    FeedForward(
        _fp8_ffn_config(
            shared_quantizer=True, fused=True, w1=HiddenZLoss(coef=0.0, on="input")
        )
    )
    ffn = FeedForward(
        _fp8_ffn_config(
            shared_quantizer=False, fused=True, w3=HiddenZLoss(coef=1e-2, on="input")
        )
    )
    assert len(ffn.w3._forward_pre_hooks) == 1
    assert set(ffn.state_dict()) == {"w1.weight", "w2.weight", "w3.weight"}


def test_a_frozen_projection_keeps_its_z_loss_route():
    """A frozen projection (a LoRA base) takes the same route as a plain one: on
    the fused path the kernel applies its z-loss, and its parameters stay frozen."""
    config = _fp8_ffn_config(
        shared_quantizer=False, fused=True, w1=HiddenZLoss(coef=1e-2)
    )
    config.w1 = FrozenConfig(inner=config.w1)
    ffn = FeedForward(config)
    assert ffn._hidden_z_loss_coefs == (1e-2, 0.0)
    assert not ffn.w1._forward_hooks
    assert ffn.w1._aux_loss_kinds == ("hidden_z_loss",)
    assert not ffn.w1.weight.requires_grad
    # The config of the caller keeps its spec: a rebuild gives the same module.
    assert config.w1.hidden_z_loss == HiddenZLoss(coef=1e-2)


# --- the aux-loss state of a fused kernel ---


def _record_state(scales, totals):
    """A stand-in for the opaque kernel state: it keeps its arguments."""
    return {"scales": scales, "totals": totals}


def test_wire_aux_loss_builds_the_two_states_of_a_fused_site():
    """Unwired, a site has no state. Wired, it has one state with the scale and
    the view of each total, and one state with no totals for
    ``suppress_aux_recording``. The two states are made one time, at the wire."""
    model = torch.nn.ModuleDict(
        {"w1": torch.nn.Linear(2, 2), "w3": torch.nn.Linear(2, 2)}
    )
    calls = []

    def make_state(scales, totals):
        calls.append(totals)
        return _record_state(scales, totals)

    splices = ((model["w1"], "hidden_z_loss"), (model["w3"], "hidden_z_loss"))
    declare_fused_aux_loss(model, splices, make_state)
    assert model["w1"]._aux_loss_kinds == ("hidden_z_loss",)
    assert fused_aux_loss_state(model) is None

    state = wire_aux_loss(model, device=torch.device("cpu"))
    recording = fused_aux_loss_state(model)
    assert recording["scales"] == (state.scale, state.scale)
    for total, name in zip(
        recording["totals"], ("hidden_z_loss/w1", "hidden_z_loss/w3")
    ):
        total.add_(1.0)
        assert state.totals[state.names.index(name)].item() == 1.0

    with suppress_aux_recording():
        suppressed = fused_aux_loss_state(model)
    assert suppressed["scales"] == recording["scales"]
    assert suppressed["totals"] == (None, None)
    assert fused_aux_loss_state(model) is recording
    assert len(calls) == 2


def test_a_fused_site_gives_no_total_for_a_splice_that_is_off():
    """One module splices two halves, and only the gate half is on: the up half
    has no scale and no total, as ``_apply_hidden_z_loss`` reads none for it."""
    experts = GroupedExperts.Config(
        dim=8, hidden_dim=4, num_experts=2, w1_hidden_z_loss_coef=1e-2
    ).build()
    # A bf16 expert path runs no fused kernel, thus it declares no fused site.
    assert getattr(experts, "_fused_aux_loss_site", None) is None
    experts._declare_fused_swiglu_site()
    site = experts._fused_aux_loss_site
    site.make_state = _record_state
    state = wire_aux_loss(experts, device=torch.device("cpu"))
    recording = fused_aux_loss_state(experts)
    assert recording["scales"] == (state.scale, None)
    assert recording["totals"][0] is not None and recording["totals"][1] is None


def test_the_pipeline_probe_suppresses_recording_in_its_forward(monkeypatch):
    """A fused kernel takes its totals at the forward, thus the forward probe
    must also run under ``suppress_aux_recording``."""
    from torchtitan.distributed import pipeline_parallel
    from torchtitan.models.common import regularization

    seen = {}

    def forward_probe(self, *args, **kwargs):
        seen["suppressed"] = regularization._SUPPRESS_RECORDING

    monkeypatch.setattr(
        pipeline_parallel.RecvPoolStage, "_forward_metadata_inference", forward_probe
    )
    stage = object.__new__(pipeline_parallel.MetadataSilentStage)
    stage.submod = torch.nn.Linear(2, 2)
    stage._forward_metadata_inference()
    assert seen == {"suppressed": True}
    assert regularization._SUPPRESS_RECORDING is False
