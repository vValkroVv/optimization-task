"""ResidualScale: warmup schedule, module, and wiring into the block (CPU)."""

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch

from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.models.common.attention import MLA, MaskType, ScaledDotProductAttention
from torchtitan.models.common.decoder_blocks.transformer_block import (
    ResidualTransformerBlock,
)
from torchtitan.models.common.embeddings.rope import ComplexRoPE
from torchtitan.models.common.ffn import FeedForward
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.norms import RMSNorm
from torchtitan.models.common.residual_scale import (
    ResidualScale,
    ResidualScaleSchedule,
    residual_scale_alpha,
)

S = ResidualScaleSchedule


def _scale(
    *,
    warmup_steps: float = 100.0,
    start_value: float = 0.0,
    schedule: ResidualScaleSchedule = ResidualScaleSchedule.linear,
    power: float = 1.0,
) -> ResidualScale.Config:
    return ResidualScale.Config(
        warmup_steps=warmup_steps,
        start_value=start_value,
        schedule=schedule,
        power=power,
    )


# --- schedule --------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("schedule", "power", "f"),
    [
        (S.linear, 1.0, lambda p: p),
        (S.sqrt, 1.0, math.sqrt),
        (S.cosine, 1.0, lambda p: 1.0 - math.cos(math.pi * p / 2.0)),
        (S.power, 3.0, lambda p: p**3.0),
    ],
)
def test_alpha_matches_formula(
    schedule: ResidualScaleSchedule, power: float, f: Callable[[float], float]
) -> None:
    s0, warmup = 0.05, 40.0
    for step in (0, 1, 10, 20, 39, 40, 41, 1000):
        p = min(step / warmup, 1.0)
        expected = s0 + (1.0 - s0) * f(p)
        got = residual_scale_alpha(
            step, warmup_steps=warmup, start_value=s0, schedule=schedule, power=power
        )
        assert got == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("start_value", [0.0, 0.05, 0.2])
@pytest.mark.parametrize("schedule", list(S))
def test_alpha_endpoints_and_monotone(
    schedule: ResidualScaleSchedule, start_value: float
) -> None:
    alphas = [
        residual_scale_alpha(
            step,
            warmup_steps=10.0,
            start_value=start_value,
            schedule=schedule,
            power=2.0,
        )
        for step in range(-3, 15)
    ]
    assert alphas[0] == pytest.approx(start_value)  # negative step clamps to 0
    assert alphas[3] == pytest.approx(start_value)  # step 0
    assert alphas[-1] == 1.0  # past warmup: exactly 1, not ~1
    assert all(a <= b for a, b in zip(alphas, alphas[1:]))


def test_fractional_warmup() -> None:
    # T * (l + 1) / L is rarely whole.
    warmup = 100 * 1 / 3
    assert residual_scale_alpha(
        33, warmup_steps=warmup, start_value=0.0, schedule=S.linear
    ) == pytest.approx(33 / warmup)
    saturated = residual_scale_alpha(
        34, warmup_steps=warmup, start_value=0.0, schedule=S.linear
    )
    assert saturated == 1.0


# --- config validation ------------------------------------------------------ #


@pytest.mark.parametrize(
    "overrides",
    [
        dict(warmup_steps=0.0),
        dict(warmup_steps=-1.0),
        dict(start_value=-0.1),
        dict(start_value=1.5),
        dict(schedule=S.power, power=0.0),
        # power has no effect off schedule=power.
        dict(schedule=S.linear, power=2.0),
    ],
)
def test_config_rejects(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _scale(**overrides)


# --- module ----------------------------------------------------------------- #


def test_forward_scales_by_alpha() -> None:
    module = _scale(start_value=0.25).build()
    x = torch.randn(2, 3, 8)
    torch.testing.assert_close(module(x), x * 0.25)
    assert module.set_step(50) == pytest.approx(0.625)
    torch.testing.assert_close(module(x), x * 0.625)


def test_bf16_input_keeps_dtype() -> None:
    module = _scale(start_value=0.5).build()
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    assert module(x).dtype == torch.bfloat16


def test_alpha_change_does_not_recompile() -> None:
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    module = _scale(start_value=0.0).build()
    counter = CompileCounter()
    compiled = torch.compile(module, backend=counter, fullgraph=True)
    x = torch.randn(2, 8)
    for step in (0, 25, 50, 100, 200):  # through saturation at alpha == 1
        alpha = module.set_step(step)
        torch.testing.assert_close(compiled(x), x * alpha)
    assert counter.frame_count == 1


def test_eager_skips_multiply_once_saturated() -> None:
    module = _scale(start_value=0.5).build()
    x = torch.randn(2, 8, requires_grad=True)
    assert module(x) is not x
    module.set_step(100)
    assert module(x) is x


@pytest.mark.parametrize("meta_init", [False, True])
def test_int_start_value_keeps_a_float_alpha(meta_init: bool) -> None:
    cfg = ResidualScale.Config(warmup_steps=10, start_value=0)
    if meta_init:
        with torch.device("meta"):
            module = cfg.build()
        module.to_empty(device="cpu")
        module.init_states(buffer_device=torch.device("cpu"))
    else:
        module = cfg.build()
    assert module.alpha.dtype == torch.float32
    module.set_step(5)
    x = torch.randn(2, 8)
    torch.testing.assert_close(module(x), x * 0.5)


def test_alpha_is_not_in_state_dict() -> None:
    module = _scale().build()
    assert "alpha" in dict(module.named_buffers())
    assert module.state_dict() == {}


def test_set_step_writes_the_buffer() -> None:
    module = _scale(start_value=0.1).build()
    module.set_step(200)
    assert module.alpha.item() == 1.0
    module.set_step(0)
    assert module.alpha.item() == pytest.approx(0.1)


def test_meta_init_fills_start_value() -> None:
    with torch.device("meta"):
        module = _scale(start_value=0.3).build()
    module.to_empty(device="cpu")
    module.init_states(buffer_device=torch.device("cpu"))
    assert module.alpha.device.type == "cpu"
    assert module.alpha.item() == pytest.approx(0.3)
    # The cached value follows the re-initialized buffer.
    module.set_step(100)
    assert module.alpha.item() == 1.0


# --- block wiring ----------------------------------------------------------- #

_DIM = 64
_N_HEADS = 4
_QK_NOPE = 16
_QK_ROPE = 16
_V_HEAD = 16
_KV_LORA = 32
_SEQ_LEN = 8


def _block_config(**block_overrides: Any) -> ResidualTransformerBlock.Config:
    mla_cfg = MLA.Config(
        dim=_DIM,
        n_heads=_N_HEADS,
        q_lora_rank=0,
        kv_lora_rank=_KV_LORA,
        qk_nope_head_dim=_QK_NOPE,
        qk_rope_head_dim=_QK_ROPE,
        v_head_dim=_V_HEAD,
        wq=Linear.Config(
            in_features=_DIM, out_features=_N_HEADS * (_QK_NOPE + _QK_ROPE)
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
        mla_cfg.wkv_a,
        mla_cfg.wkv_b,
        mla_cfg.wo,
        mla_cfg.kv_norm,
    ):
        projection.param_init = {}
    return ResidualTransformerBlock.Config(
        attention=mla_cfg,
        attention_norm=RMSNorm.Config(normalized_shape=_DIM),
        ffn_norm=RMSNorm.Config(normalized_shape=_DIM),
        post_attn_norm=RMSNorm.Config(normalized_shape=_DIM),
        post_ffn_norm=RMSNorm.Config(normalized_shape=_DIM),
        feed_forward=FeedForward.Config(
            dim=_DIM,
            hidden_dim=128,
            w1=LinearConfig(param_init={}),
            w2=LinearConfig(param_init={}),
            w3=LinearConfig(param_init={}),
        ),
        **block_overrides,
    )


def _pair(
    **scales: Any,
) -> tuple[
    ResidualTransformerBlock, ResidualTransformerBlock, torch.Tensor, torch.Tensor
]:
    """A plain block and a scaled one with identical weights."""
    torch.manual_seed(0)
    plain = _block_config().build()
    scaled = _block_config(**scales).build()
    scaled.load_state_dict(plain.state_dict())
    rope = ComplexRoPE.Config(dim=_QK_ROPE, max_seq_len=_SEQ_LEN).build().cache
    x = torch.randn(2, _SEQ_LEN, _DIM)
    return plain, scaled, rope, x


def _ffn_branch(block: ResidualTransformerBlock, h: torch.Tensor) -> torch.Tensor:
    assert block.feed_forward is not None and block.post_ffn_norm is not None
    return block.post_ffn_norm(block.feed_forward(block.ffn_norm(h)))


def test_unset_scales_build_nothing() -> None:
    block = _block_config().build()
    assert block.attn_residual_scale is None
    assert block.ffn_residual_scale is None
    assert not any(isinstance(m, ResidualScale) for m in block.modules())


def test_scales_leave_state_dict_and_flops_unchanged() -> None:
    plain = _block_config()
    scaled = _block_config(attn_residual_scale=_scale(), ffn_residual_scale=_scale())
    assert plain.build().state_dict().keys() == scaled.build().state_dict().keys()
    assert plain.fwd_flops_per_token(_SEQ_LEN) == scaled.fwd_flops_per_token(_SEQ_LEN)


def test_saturated_scales_are_bit_identical() -> None:
    plain, scaled, rope, x = _pair(
        attn_residual_scale=_scale(), ffn_residual_scale=_scale()
    )
    for module in (scaled.attn_residual_scale, scaled.ffn_residual_scale):
        assert module is not None
        module.set_step(10_000)
    with torch.no_grad():
        assert torch.equal(plain(x, rope), scaled(x, rope))


def test_zero_attention_scale_drops_the_attention_branch() -> None:
    plain, scaled, rope, x = _pair(attn_residual_scale=_scale(start_value=0.0))
    with torch.no_grad():
        ffn_only = x + _ffn_branch(plain, x)
        torch.testing.assert_close(scaled(x, rope), ffn_only)


def test_each_branch_scales_independently() -> None:
    plain, scaled, rope, x = _pair(ffn_residual_scale=_scale(start_value=0.5))
    with torch.no_grad():
        assert plain.post_attn_norm is not None
        attn_in = plain.attention_norm(x)
        h = x + plain.post_attn_norm(plain.attention(attn_in, rope, None, None))
        expected = h + 0.5 * _ffn_branch(plain, h)
        torch.testing.assert_close(scaled(x, rope), expected)


def test_scale_without_post_norm() -> None:
    # Pre-norm blocks take it too.
    torch.manual_seed(0)
    cfg = _block_config(ffn_residual_scale=_scale(start_value=0.0))
    cfg.post_attn_norm = None
    cfg.post_ffn_norm = None
    block = cfg.build()
    rope = ComplexRoPE.Config(dim=_QK_ROPE, max_seq_len=_SEQ_LEN).build().cache
    x = torch.randn(2, _SEQ_LEN, _DIM)
    with torch.no_grad():
        expected = x + block.attention(block.attention_norm(x), rope, None, None)
        torch.testing.assert_close(block(x, rope), expected)


def test_grad_flows_through_scaled_branch() -> None:
    _, scaled, rope, x = _pair(attn_residual_scale=_scale(start_value=0.5))
    scaled(x, rope).sum().backward()
    assert scaled.attention.wo.weight.grad is not None
    assert scaled.attention.wo.weight.grad.abs().sum() > 0
