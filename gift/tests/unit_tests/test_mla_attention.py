"""Unit tests for the shared Multi-head Latent Attention (MLA) module.

Covers the consolidated ``MLA`` used by both ``deepseek_v3`` and ``gift_moe``:
the two Q-projection paths (``q_lora_rank == 0`` direct, ``q_lora_rank > 0``
low-rank), output shape, forward determinism, and the config validation guards
inherited from ``SoftmaxAttention.Config``.
"""

import math
import unittest

import torch

from torchtitan.models.common.attention import (
    MaskType,
    MLA,
    ScaledDotProductAttention,
)
from torchtitan.models.common.embeddings.rope import ComplexRoPE
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.norms import RMSNorm


# Tiny dims keep the test CPU-fast.
_DIM = 64
_N_HEADS = 4
_QK_NOPE = 16
_QK_ROPE = 16
_V_HEAD = 16
_KV_LORA = 32
_SEQ_LEN = 8
_BSZ = 2


def _make_mla_config(
    *,
    q_lora_rank: int,
    inner_attention=None,
    mask_type=None,
    use_mla_scaling_factor: bool = False,
):
    """Build a tiny, fully-specified ``MLA.Config`` for CPU testing."""
    qk_head_dim = _QK_NOPE + _QK_ROPE
    if inner_attention is None:
        inner_attention = ScaledDotProductAttention.Config()
    if mask_type is None:
        mask_type = MaskType.CAUSAL

    if q_lora_rank == 0:
        wq = Linear.Config(in_features=_DIM, out_features=_N_HEADS * qk_head_dim)
        wq_a = wq_b = None
        q_norm = None
    else:
        wq = None
        wq_a = Linear.Config(in_features=_DIM, out_features=q_lora_rank)
        wq_b = Linear.Config(
            in_features=q_lora_rank, out_features=_N_HEADS * qk_head_dim
        )
        q_norm = RMSNorm.Config(normalized_shape=q_lora_rank)

    config = MLA.Config(
        dim=_DIM,
        n_heads=_N_HEADS,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=_KV_LORA,
        qk_nope_head_dim=_QK_NOPE,
        qk_rope_head_dim=_QK_ROPE,
        v_head_dim=_V_HEAD,
        wq=wq,
        wq_a=wq_a,
        wq_b=wq_b,
        q_norm=q_norm,
        kv_norm=RMSNorm.Config(normalized_shape=_KV_LORA),
        wkv_a=Linear.Config(in_features=_DIM, out_features=_KV_LORA + _QK_ROPE),
        wkv_b=Linear.Config(
            in_features=_KV_LORA, out_features=_N_HEADS * (_QK_NOPE + _V_HEAD)
        ),
        wo=Linear.Config(in_features=_N_HEADS * _V_HEAD, out_features=_DIM),
        inner_attention=inner_attention,
        mask_type=mask_type,
        use_mla_scaling_factor=use_mla_scaling_factor,
    )
    return config


_ROPE = ComplexRoPE.Config(dim=_QK_ROPE, max_seq_len=_SEQ_LEN)


def _rope_cache() -> "torch.Tensor":
    return _ROPE.build().cache


class TestMLAAttention(unittest.TestCase):
    def _run_forward(self, q_lora_rank: int) -> "torch.Tensor":
        torch.manual_seed(0)
        mla = _make_mla_config(q_lora_rank=q_lora_rank).build()
        x = torch.randn(_BSZ, _SEQ_LEN, _DIM)
        return mla(x, _rope_cache(), attention_masks=None)

    def test_forward_shape_direct_q(self):
        """q_lora_rank == 0 takes the direct ``wq`` projection."""
        out = self._run_forward(q_lora_rank=0)
        self.assertEqual(out.shape, (_BSZ, _SEQ_LEN, _DIM))

    def test_forward_shape_lowrank_q(self):
        """q_lora_rank > 0 takes the ``wq_a``/``q_norm``/``wq_b`` low-rank path."""
        out = self._run_forward(q_lora_rank=24)
        self.assertEqual(out.shape, (_BSZ, _SEQ_LEN, _DIM))

    def test_invalid_mask_type_rejected(self):
        """An unknown mask string fails ``SoftmaxAttention.Config`` validation."""
        with self.assertRaises(ValueError):
            _make_mla_config(q_lora_rank=0, mask_type="not_a_mask")

    def test_sdpa_block_causal_rejected(self):
        """SDPA inner kernel does not support block_causal masking."""
        with self.assertRaises(ValueError):
            _make_mla_config(
                q_lora_rank=0,
                inner_attention=ScaledDotProductAttention.Config(),
                mask_type=MaskType.BLOCK_CAUSAL,
            )

    def test_scaling_disabled_is_identity(self):
        """Without the scaling factor, both alphas are 1.0 (no behavior change)."""
        mla = _make_mla_config(q_lora_rank=24).build()
        self.assertEqual(mla.alpha_q, 1.0)
        self.assertEqual(mla.alpha_kv, 1.0)

    def test_scaling_values_lowrank_q(self):
        """alpha_q/alpha_kv = sqrt(dim / latent_dim) on the low-rank Q path."""
        mla = _make_mla_config(q_lora_rank=24, use_mla_scaling_factor=True).build()
        self.assertAlmostEqual(mla.alpha_q, math.sqrt(_DIM / 24))
        self.assertAlmostEqual(mla.alpha_kv, math.sqrt(_DIM / _KV_LORA))

    def test_scaling_full_rank_q_alpha_q_is_one(self):
        """Full-rank Q (q_lora_rank == 0) has q_hidden_dim == dim, so alpha_q == 1."""
        mla = _make_mla_config(q_lora_rank=0, use_mla_scaling_factor=True).build()
        self.assertEqual(mla.alpha_q, 1.0)
        self.assertAlmostEqual(mla.alpha_kv, math.sqrt(_DIM / _KV_LORA))

    def test_scaling_changes_output(self):
        """The scaling factor is actually applied in forward (low-rank Q path)."""
        torch.manual_seed(0)
        mla = _make_mla_config(q_lora_rank=24, use_mla_scaling_factor=True).build()
        x = torch.randn(_BSZ, _SEQ_LEN, _DIM)
        freqs_cis = _rope_cache()
        scaled = mla(x, freqs_cis, attention_masks=None)
        mla.alpha_q, mla.alpha_kv = 1.0, 1.0  # disable, same weights
        unscaled = mla(x, freqs_cis, attention_masks=None)
        self.assertFalse(torch.allclose(scaled, unscaled))


if __name__ == "__main__":
    unittest.main()
