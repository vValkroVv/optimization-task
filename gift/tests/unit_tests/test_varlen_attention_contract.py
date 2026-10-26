"""Unit tests for the family-agnostic varlen attention contract.

Covers the shared varlen core (``attention/varlen.py``: pack/unpack helpers and
``create_varlen_metadata_for_document``) and the declarative ``attention_mask_kind``
seam that lets the trainer/decoder dispatch on the mask representation a family
needs without enumerating concrete inner-kernel types.
"""

import unittest

import torch

from torchtitan.config.schema.models.common.attention.qkv import (
    LinearAttnQKVConfig,
    QKVLinearConfig,
)
from torchtitan.config.schema.models.common.nn_modules import Conv1dConfig, LinearConfig
from torchtitan.config.schema.models.common.norms.fused_gated_rmsnorm import (
    FusedRMSNormGatedConfig,
)
from torchtitan.models.common.attention import (
    AttentionMaskKind,
    FlexAttention,
    GatedDeltaNet,
    GQAttention,
    KimiDeltaAttention,
    MaskType,
    ScaledDotProductAttention,
    VarlenAttention,
    VarlenMetadata,
    create_varlen_metadata_for_document,
    pack_varlen_qkv,
    unpack_varlen_out,
)

_NO_INIT = {"weight": lambda t: t}


def _gqa_kind(inner_attention) -> "AttentionMaskKind":
    """Build a GQAttention.Config with the given inner kernel and return its kind."""
    # SDPA does not support block_causal; use causal there, block_causal otherwise.
    mask_type = (
        MaskType.CAUSAL
        if isinstance(inner_attention, ScaledDotProductAttention.Config)
        else MaskType.BLOCK_CAUSAL
    )
    cfg = GQAttention.Config(
        dim=64,
        n_heads=4,
        qkv_linear=QKVLinearConfig(
            wq=LinearConfig(param_init=_NO_INIT), wkv=LinearConfig(param_init=_NO_INIT)
        ),
        wo=LinearConfig(param_init=_NO_INIT),
        inner_attention=inner_attention,
        mask_type=mask_type,
    )
    return cfg.attention_mask_kind


class TestVarlenCore(unittest.TestCase):
    """The shared, family-agnostic varlen primitives."""

    def test_pack_unpack_roundtrip_mha(self):
        q = torch.randn(2, 5, 4, 16)
        k = torch.randn(2, 5, 4, 16)
        v = torch.randn(2, 5, 4, 16)
        qp, kp, vp = pack_varlen_qkv(q, k, v)
        self.assertEqual(tuple(qp.shape), (10, 4, 16))
        self.assertTrue(torch.equal(unpack_varlen_out(qp, 2, 5), q))

    def test_pack_keeps_per_tensor_head_counts_gqa(self):
        """q and k/v may have different head counts under GQA."""
        q = torch.randn(2, 5, 8, 16)
        k = torch.randn(2, 5, 2, 16)
        v = torch.randn(2, 5, 2, 16)
        qp, kp, vp = pack_varlen_qkv(q, k, v)
        self.assertEqual(tuple(qp.shape), (10, 8, 16))
        self.assertEqual(tuple(kp.shape), (10, 2, 16))
        self.assertEqual(tuple(vp.shape), (10, 2, 16))

    def test_metadata_cu_seqlens_for_packed_documents(self):
        # batch0 docs of length 3,2 -> offsets 0,3 then base 5
        # batch1 docs of length 2,3 -> offsets 5,7 then total 10
        positions = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 0, 1, 2]])
        meta = create_varlen_metadata_for_document(positions)
        self.assertIsInstance(meta, VarlenMetadata)
        self.assertEqual(meta.cu_seq_q.tolist(), [0, 3, 5, 7, 10])
        # q and k share the same cumulative lengths for self-attention.
        self.assertEqual(meta.cu_seq_k.tolist(), [0, 3, 5, 7, 10])
        self.assertEqual(meta.max_q, 3)
        self.assertEqual(meta.max_k, 3)

    def test_metadata_single_document(self):
        positions = torch.tensor([[0, 1, 2, 3]])
        meta = create_varlen_metadata_for_document(positions)
        self.assertEqual(meta.cu_seq_q.tolist(), [0, 4])
        self.assertEqual(meta.max_q, 4)


class TestAttentionMaskKind(unittest.TestCase):
    """Each family's config reports the mask representation it needs."""

    def test_softmax_flex_is_block_mask(self):
        self.assertEqual(
            _gqa_kind(FlexAttention.Config()), AttentionMaskKind.BLOCK_MASK
        )

    def test_softmax_varlen_is_varlen(self):
        self.assertEqual(_gqa_kind(VarlenAttention.Config()), AttentionMaskKind.VARLEN)

    def test_softmax_sdpa_is_none(self):
        self.assertEqual(
            _gqa_kind(ScaledDotProductAttention.Config()), AttentionMaskKind.NONE
        )

    def _linear_kwargs(self, *, mask_type: MaskType) -> dict:
        return dict(
            dim=64,
            num_k_heads=2,
            num_v_heads=4,
            head_k_dim=16,
            head_v_dim=16,
            conv_size=4,
            qkv=LinearAttnQKVConfig(
                head_k_dim=16,
                head_v_dim=16,
                num_k_heads=2,
                num_v_heads=4,
                qkv_proj=LinearConfig(
                    param_init=_NO_INIT,
                    in_features=64,
                    out_features=(2 * 2 * 16 + 4 * 16),
                ),
                conv=Conv1dConfig(
                    param_init=_NO_INIT,
                    in_channels=(2 * 2 * 16 + 4 * 16),
                    out_channels=(2 * 2 * 16 + 4 * 16),
                    kernel_size=4,
                    groups=(2 * 2 * 16 + 4 * 16),
                    padding=4 - 1,
                    bias=False,
                ),
            ),
            o_proj=LinearConfig(param_init=_NO_INIT),
            o_norm=FusedRMSNormGatedConfig(param_init=_NO_INIT),
            param_init=_NO_INIT,
            mask_type=mask_type,
        )

    def test_linear_gdn_varlen_flag(self):
        on = GatedDeltaNet.Config(
            **self._linear_kwargs(mask_type=MaskType.BLOCK_CAUSAL)
        )
        off = GatedDeltaNet.Config(**self._linear_kwargs(mask_type=MaskType.CAUSAL))
        self.assertEqual(on.attention_mask_kind, AttentionMaskKind.VARLEN)
        self.assertEqual(off.attention_mask_kind, AttentionMaskKind.NONE)

    def test_linear_kda_varlen_flag(self):
        on = KimiDeltaAttention.Config(
            **self._linear_kwargs(mask_type=MaskType.BLOCK_CAUSAL)
        )
        off = KimiDeltaAttention.Config(
            **self._linear_kwargs(mask_type=MaskType.CAUSAL)
        )
        self.assertEqual(on.attention_mask_kind, AttentionMaskKind.VARLEN)
        self.assertEqual(off.attention_mask_kind, AttentionMaskKind.NONE)


if __name__ == "__main__":
    unittest.main()
