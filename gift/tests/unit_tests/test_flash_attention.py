"""Unit tests for the ``flash_attn``-backed inner kernels.

Two layers:

- **Config wiring** (CPU, no flash_attn needed): the mask metadata each kernel
  config declares, ``build()`` owner resolution, kernel knobs reaching the model
  config, and the FA4 scaffold raising on build.
- **Kernel numerics** (CUDA + an importable flash_attn build): dense output vs
  ``ScaledDotProductAttention``, ``return_lse`` shape, backward, and the varlen
  path vs SDPA. These self-skip when neither a flash backend nor a GPU is
  available (e.g. CPU CI, or the bundled flash_attn being ABI-incompatible).
"""

import unittest

import torch

from torchtitan.config.schema.models.common.attention.const import AttentionMaskKind
from torchtitan.config.schema.models.common.attention.qkv import QKVLinearConfig
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.protocols.module import Zeros
from torchtitan.models.common.attention import (
    GQAttention,
    MaskType,
    ScaledDotProductAttention,
)
from torchtitan.models.common.attention.softmax.kernels import (
    FlashAttention2,
    FlashAttention3,
    FlashAttention4,
    FlashVarlenAttention2,
    FlashVarlenAttention3,
    FlashVarlenAttention4,
)
from torchtitan.models.common.attention.varlen_meta import VarlenMetadata

# Tiny dims keep these fast; head_dim is a multiple of 8 (flash requirement).
_BSZ = 2
_SEQ = 64
_N_HEADS = 4
_N_KV_HEADS = 2
_HEAD_DIM = 64


def _available_flash_backends():
    """Return [(name, dense_cls, varlen_cls)] whose flash_attn build imports.

    Building a Config triggers the lazy import; a missing or ABI-broken package
    raises ``ValueError``, so we probe by attempting a build.
    """
    backends = []
    if not torch.cuda.is_available():
        return backends
    candidates = [
        ("fa2", FlashAttention2, FlashVarlenAttention2),
        ("fa3", FlashAttention3, FlashVarlenAttention3),
        ("fa4", FlashAttention4, FlashVarlenAttention4),
    ]
    for name, dense, varlen in candidates:
        try:
            dense.Config().build()
        except Exception:
            continue
        backends.append((name, dense, varlen))
    return backends


_FLASH_BACKENDS = _available_flash_backends()


class TestFlashConfigWiring(unittest.TestCase):
    """CPU-only: registration and config plumbing (no flash_attn needed)."""

    def test_dense_configs_declare_causal(self):
        for cls in [FlashAttention2, FlashAttention3, FlashAttention4]:
            self.assertEqual(cls.Config.default_mask_type(), MaskType.CAUSAL)
            self.assertEqual(cls.Config.mask_kind, AttentionMaskKind.NONE)

    def test_varlen_configs_declare_block_causal(self):
        for cls in [
            FlashVarlenAttention2,
            FlashVarlenAttention3,
            FlashVarlenAttention4,
        ]:
            self.assertEqual(cls.Config.default_mask_type(), MaskType.BLOCK_CAUSAL)
            self.assertEqual(cls.Config.mask_kind, AttentionMaskKind.VARLEN)

    def test_build_owner_resolution(self):
        # Each concrete Config must build its own class, not a base.
        for cls in [
            FlashAttention2,
            FlashAttention3,
            FlashAttention4,
            FlashVarlenAttention2,
            FlashVarlenAttention3,
            FlashVarlenAttention4,
        ]:
            self.assertIs(cls.Config._owner, cls)

    def test_attention_mask_kind(self):
        pinit = {"weight": Zeros()}

        def kind(inner):
            cfg = GQAttention.Config(
                dim=_HEAD_DIM,
                n_heads=_N_HEADS,
                qkv_linear=QKVLinearConfig(
                    wq=LinearConfig(param_init=pinit),
                    wkv=LinearConfig(param_init=pinit),
                ),
                wo=LinearConfig(param_init=pinit),
                inner_attention=inner,
            )
            return cfg.attention_mask_kind

        # The outer config reports what its kernel declares, and adopts that
        # kernel's default mask type without being told.
        self.assertEqual(kind(FlashAttention3.Config()), AttentionMaskKind.NONE)
        self.assertEqual(kind(FlashVarlenAttention3.Config()), AttentionMaskKind.VARLEN)

    def test_kernel_knobs_reach_the_model(self):
        # Kernel-specific knobs are set by naming the kernel config in the model
        # builder. Sliding-window attention is the case that matters: it was
        # implemented but unreachable while the kernel was picked by a string.
        from tests.unit_tests.config_helpers import load_model_spec

        cfg = load_model_spec(
            "llama3/debugmodel.yaml",
            {
                "config": {
                    "blocks": {
                        "dense": {
                            "attention": {
                                "inner_attention": {
                                    "type": "flash_varlen_fa3",
                                    "window_size": [128, 0],
                                }
                            }
                        }
                    }
                }
            },
        ).model
        attention = cfg.layers[0].attention
        self.assertEqual(attention.inner_attention.window_size, (128, 0))
        # ... and the mask type follows the kernel without being passed along.
        self.assertEqual(attention.mask_type, MaskType.BLOCK_CAUSAL)
        self.assertEqual(attention.attention_mask_kind, AttentionMaskKind.VARLEN)

    def test_missing_package_raises_clear_valueerror(self):
        # The guarded lazy import surfaces a helpful ValueError naming the
        # package, not a raw ImportError.
        orig = FlashAttention4._PACKAGE
        FlashAttention4._PACKAGE = "no_such_flash_pkg_zzz"
        try:
            with self.assertRaises(ValueError) as ctx:
                FlashAttention4.Config().build()
            self.assertIn("no_such_flash_pkg_zzz", str(ctx.exception))
        finally:
            FlashAttention4._PACKAGE = orig


@unittest.skipUnless(
    _FLASH_BACKENDS, "no importable flash_attn build / CUDA unavailable"
)
class TestFlashKernels(unittest.TestCase):
    """CUDA: numerics against the torch-native SDPA reference."""

    def setUp(self):
        torch.manual_seed(0)
        self.dev = torch.device("cuda")
        self.dt = torch.bfloat16
        mk = lambda h: torch.randn(  # noqa: E731
            _BSZ, _SEQ, h, _HEAD_DIM, device=self.dev, dtype=self.dt
        )
        self.q, self.k, self.v = mk(_N_HEADS), mk(_N_KV_HEADS), mk(_N_KV_HEADS)
        self.sdpa = ScaledDotProductAttention.Config().build()
        self.ref = self.sdpa(
            self.q, self.k, self.v, scale=None, is_causal=True, enable_gqa=True
        )

    def test_dense_matches_sdpa(self):
        for name, dense, _ in _FLASH_BACKENDS:
            with self.subTest(backend=name):
                kernel = dense.Config().build()
                out = kernel(
                    self.q, self.k, self.v, scale=None, is_causal=True, enable_gqa=True
                )
                self.assertEqual(out.shape, self.ref.shape)
                torch.testing.assert_close(
                    out.float(), self.ref.float(), atol=2.5e-2, rtol=0
                )

    def test_return_lse_shape(self):
        for name, dense, _ in _FLASH_BACKENDS:
            with self.subTest(backend=name):
                kernel = dense.Config().build()
                out, lse = kernel(
                    self.q, self.k, self.v, scale=None, is_causal=True, return_lse=True
                )
                self.assertEqual(out.shape, (_BSZ, _SEQ, _N_HEADS, _HEAD_DIM))
                # LSE follows the (bs, seq, heads) convention used by FlexAttention.
                self.assertEqual(lse.shape, (_BSZ, _SEQ, _N_HEADS))

    def test_backward(self):
        for name, dense, _ in _FLASH_BACKENDS:
            with self.subTest(backend=name):
                kernel = dense.Config().build()
                q = self.q.clone().requires_grad_(True)
                kernel(q, self.k, self.v, scale=None, is_causal=True).sum().backward()
                self.assertIsNotNone(q.grad)
                self.assertTrue(torch.isfinite(q.grad).all())

    def test_varlen_matches_sdpa(self):
        # One document per sequence == standard per-sequence causal attention.
        cu = torch.arange(
            0, (_BSZ + 1) * _SEQ, _SEQ, device=self.dev, dtype=torch.int32
        )
        meta = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=_SEQ, max_k=_SEQ)
        for name, _, varlen in _FLASH_BACKENDS:
            with self.subTest(backend=name):
                kernel = varlen.Config().build()
                out = kernel(self.q, self.k, self.v, attention_masks=meta, scale=None)
                self.assertEqual(out.shape, self.ref.shape)
                torch.testing.assert_close(
                    out.float(), self.ref.float(), atol=2.5e-2, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
