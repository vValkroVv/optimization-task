"""Router z-loss + hidden z-loss regularizers (CPU).

Covers the router z-loss formula, the grad-only splice (forward value unchanged,
gradient injected), the no-op-when-off invariant, and that the router surfaces
pre-activation logits.
"""

from dataclasses import dataclass

import pytest
import torch

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
    set_hidden_z_loss,
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
