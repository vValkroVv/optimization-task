"""Unit tests for the sigmoid attention output gate (:class:`SigmoidGate`).

Covers the standalone gate module (fp32 numerics, shape/dtype, gradients) and its
integration into both softmax attention families (``GQAttention`` and ``MLA``):
with the gate projection zeroed, ``sigmoid(0) == 0.5`` must scale the gated
output to exactly half the ungated output (``wo`` is bias-free linear).
"""

import unittest

import torch

from torchtitan.config.schema.models.common.attention.qkv import QKVLinearConfig
from torchtitan.config.schema.models.common.attention.softmax.gate import (
    SigmoidGateConfig,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.protocols.module import Normal
from torchtitan.models.common.attention import (
    MLA,
    GQAttention,
    MaskType,
    ScaledDotProductAttention,
    SigmoidGate,
)
from torchtitan.models.common.attention.qkv import QKVLinear
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.norms import RMSNorm

# Tiny dims keep the test CPU-fast.
_DIM = 64
_N_HEADS = 4
_HEAD_DIM = _DIM // _N_HEADS
_QK_NOPE = 16
_QK_ROPE = 16
_V_HEAD = 16
_KV_LORA = 32
_SEQ_LEN = 8
_BSZ = 2


def _gate_config(out_features: int) -> "SigmoidGate.Config":
    return SigmoidGate.Config(
        gate_proj=Linear.Config(in_features=_DIM, out_features=out_features)
    )


def _make_gqa(*, gate: bool) -> "GQAttention":
    """Tiny GQAttention (no RoPE) with an optional sigmoid gate."""
    cfg = GQAttention.Config(
        n_heads=_N_HEADS,
        dim=_DIM,
        n_kv_heads=_N_HEADS,
        head_dim=_HEAD_DIM,
        qkv_linear=QKVLinear.Config(
            head_dim=_HEAD_DIM,
            wq=Linear.Config(in_features=_DIM, out_features=_N_HEADS * _HEAD_DIM),
            wkv=Linear.Config(in_features=_DIM, out_features=_N_HEADS * _HEAD_DIM),
        ),
        wo=Linear.Config(in_features=_N_HEADS * _HEAD_DIM, out_features=_DIM),
        inner_attention=ScaledDotProductAttention.Config(),
        use_rope=False,
        mask_type=MaskType.CAUSAL,
        gate=_gate_config(_N_HEADS * _HEAD_DIM) if gate else None,
    )
    return cfg.build()


def _make_mla(*, gate: bool) -> "MLA":
    """Tiny MLA (direct-Q path) with an optional sigmoid gate."""
    qk_head_dim = _QK_NOPE + _QK_ROPE
    cfg = MLA.Config(
        dim=_DIM,
        n_heads=_N_HEADS,
        q_lora_rank=0,
        kv_lora_rank=_KV_LORA,
        qk_nope_head_dim=_QK_NOPE,
        qk_rope_head_dim=_QK_ROPE,
        v_head_dim=_V_HEAD,
        wq=Linear.Config(in_features=_DIM, out_features=_N_HEADS * qk_head_dim),
        wq_a=None,
        wq_b=None,
        kv_norm=RMSNorm.Config(normalized_shape=_KV_LORA),
        wkv_a=Linear.Config(in_features=_DIM, out_features=_KV_LORA + _QK_ROPE),
        wkv_b=Linear.Config(
            in_features=_KV_LORA, out_features=_N_HEADS * (_QK_NOPE + _V_HEAD)
        ),
        wo=Linear.Config(in_features=_N_HEADS * _V_HEAD, out_features=_DIM),
        inner_attention=ScaledDotProductAttention.Config(),
        mask_type=MaskType.CAUSAL,
        gate=_gate_config(_N_HEADS * _V_HEAD) if gate else None,
    )
    return cfg.build()


def _mla_rope_config():
    from torchtitan.models.common.embeddings.rope import ComplexRoPE

    return ComplexRoPE.Config(dim=_QK_ROPE, max_seq_len=_SEQ_LEN)


def _mla_rope_cache() -> "torch.Tensor":
    from torchtitan.models.common.embeddings.rope import ComplexRoPE

    return ComplexRoPE.Config(dim=_QK_ROPE, max_seq_len=_SEQ_LEN).build().cache


class TestSigmoidGate(unittest.TestCase):
    """The standalone SigmoidGate module."""

    def test_numeric_matches_reference(self):
        torch.manual_seed(0)
        gate = _gate_config(_DIM).build()
        x = torch.randn(_BSZ, _SEQ_LEN, _DIM)
        attn = torch.randn(_BSZ, _SEQ_LEN, _DIM)
        ref = (
            attn.float()
            * torch.sigmoid(
                torch.nn.functional.linear(x, gate.gate_proj.weight).float()
            )
        ).to(attn.dtype)
        torch.testing.assert_close(gate(x, attn), ref)

    def test_gradients_flow(self):
        gate = _gate_config(_DIM).build()
        x = torch.randn(_BSZ, _SEQ_LEN, _DIM, requires_grad=True)
        attn = torch.randn(_BSZ, _SEQ_LEN, _DIM, requires_grad=True)
        gate(x, attn).pow(2).sum().backward()
        self.assertIsNotNone(gate.gate_proj.weight.grad)
        self.assertTrue(torch.isfinite(gate.gate_proj.weight.grad).all())
        self.assertIsNotNone(x.grad)


class TestGatedAttention(unittest.TestCase):
    """Gate integrated into GQAttention and MLA."""

    def _assert_zero_gate_halves(self, attn, fwd):
        """With gate_proj zeroed, sigmoid(0)=0.5 -> gated output is half of ungated.

        ``wo`` is a bias-free linear, so the 0.5 factor passes through it.
        """
        with torch.no_grad():
            attn.gate.gate_proj.weight.zero_()
        gated = fwd(attn)
        saved, attn.gate = attn.gate, None  # bypass the gate for the reference
        try:
            ungated = fwd(attn)
        finally:
            attn.gate = saved
        torch.testing.assert_close(gated, 0.5 * ungated)

    def test_gqa_zero_gate_halves_output(self):
        torch.manual_seed(0)
        attn = _make_gqa(gate=True)
        x = torch.randn(_BSZ, _SEQ_LEN, _DIM)
        self._assert_zero_gate_halves(
            attn, lambda m: m(x, rope_cache=None, attention_masks=None)
        )

    def test_mla_zero_gate_halves_output(self):
        torch.manual_seed(0)
        attn = _make_mla(gate=True)
        x = torch.randn(_BSZ, _SEQ_LEN, _DIM)
        rope = _mla_rope_cache()
        self._assert_zero_gate_halves(attn, lambda m: m(x, rope, attention_masks=None))

    def test_gate_changes_output(self):
        """A non-trivial (random) gate must change the MLA output."""
        torch.manual_seed(0)
        attn = _make_mla(gate=True)
        x = torch.randn(_BSZ, _SEQ_LEN, _DIM)
        rope = _mla_rope_cache()
        gated = attn(x, rope, attention_masks=None)
        saved, attn.gate = attn.gate, None
        try:
            ungated = attn(x, rope, attention_masks=None)
        finally:
            attn.gate = saved
        self.assertFalse(torch.allclose(gated, ungated))


class TestMakeGqaConfigGate(unittest.TestCase):
    """``make_gqa_config`` gate wiring + FLOP accounting."""

    def test_gate_param_init_builds_gate(self):
        init = {"weight": Normal(std=0.02)}
        cfg = GQAttention.Config(
            dim=_DIM,
            n_heads=_N_HEADS,
            qkv_linear=QKVLinearConfig(
                wq=LinearConfig(param_init=init), wkv=LinearConfig(param_init=init)
            ),
            wo=LinearConfig(param_init=init),
            inner_attention=ScaledDotProductAttention.Config(),
            gate=SigmoidGateConfig(gate_proj=LinearConfig(param_init=init)),
        )
        self.assertIsNotNone(cfg.gate)
        self.assertEqual(cfg.gate.gate_proj.in_features, _DIM)
        self.assertEqual(cfg.gate.gate_proj.out_features, _N_HEADS * _HEAD_DIM)

    def test_no_gate_param_init_no_gate(self):
        init = {"weight": Normal(std=0.02)}
        cfg = GQAttention.Config(
            dim=_DIM,
            n_heads=_N_HEADS,
            qkv_linear=QKVLinearConfig(
                wq=LinearConfig(param_init=init), wkv=LinearConfig(param_init=init)
            ),
            wo=LinearConfig(param_init=init),
            inner_attention=ScaledDotProductAttention.Config(),
        )
        self.assertIsNone(cfg.gate)

    def test_gate_adds_flops(self):
        init = {"weight": Normal(std=0.02)}
        common = dict(
            dim=_DIM,
            n_heads=_N_HEADS,
            qkv_linear=QKVLinearConfig(
                wq=LinearConfig(param_init=init), wkv=LinearConfig(param_init=init)
            ),
            wo=LinearConfig(param_init=init),
            inner_attention=ScaledDotProductAttention.Config(),
        )
        no_gate = GQAttention.Config(**common)
        with_gate = GQAttention.Config(
            **common, gate=SigmoidGateConfig(gate_proj=LinearConfig(param_init=init))
        )
        extra = with_gate.fwd_flops_per_token(_SEQ_LEN) - no_gate.fwd_flops_per_token(
            _SEQ_LEN
        )
        # gate_proj matmul: 2 * dim * (n_heads * head_dim).
        self.assertEqual(extra, 2 * _DIM * _N_HEADS * _HEAD_DIM)


if __name__ == "__main__":
    unittest.main()
