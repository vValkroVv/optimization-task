"""Unit tests for the linear-attention families (GDN / KDA) and the GIFT 3.5
hybrid model.

Config-level tests run without ``flash-linear-attention`` (``fla``); the module
build + forward tests need ``fla`` and a GPU and are skipped otherwise.
"""

import importlib.util
import unittest

import torch

from torchtitan.config.schema.models.common.attention.linear.param_init import (
    GatedDeltaALog,
    GatedDeltaDtBias,
)
from torchtitan.config.schema.models.common.attention.qkv import (
    LinearAttnQKVConfig,
)
from torchtitan.config.schema.models.common.decoder_blocks.decoder import (
    DecoderConfig,
)
from torchtitan.config.schema.models.common.nn_modules import Conv1dConfig, LinearConfig
from torchtitan.config.schema.models.common.norms.fused_gated_rmsnorm import (
    FusedRMSNormGatedConfig,
)
from torchtitan.config.schema.protocols.module import Normal, Ones
from torchtitan.models.common.attention import (
    AttentionMaskKind,
    GatedDeltaNet,
    KimiDeltaAttention,
    MaskType,
    create_varlen_metadata_for_document,
)

_GPU_OK = torch.cuda.is_available()
# fla forward tests need flash-linear-attention plus a GPU.
_FLA_OK = importlib.util.find_spec("fla") is not None

# Backward additionally needs a working gated delta-rule backward. On Hopper +
# Triton in [3.4.0, 3.7.1) the Triton kernel is numerically broken (fla #640) and
# fla routes through its tilelang backend instead; without tilelang (and without
# Triton >= 3.7.1) fla hard-raises. Reuse fla's own guard flags so the skip
# condition matches the kernel exactly.
_BWD_OK = _FLA_OK and _GPU_OK
if _BWD_OK:
    try:
        from fla.utils import IS_NVIDIA_HOPPER, TRITON_ABOVE_3_7_1

        if IS_NVIDIA_HOPPER and not TRITON_ABOVE_3_7_1:
            _BWD_OK = importlib.util.find_spec("tilelang") is not None
    except ImportError:
        # An fla build that does not export the flags: assume the kernel is fine
        # rather than skipping, so a real backward break still shows up.
        pass

# Real initializers for the forward tests (config-only tests ignore them):
# ``_INIT`` for the projections + short conv, ``_NORM_INIT`` for the gated norm.
_INIT = {"weight": Normal(std=0.02)}
_NORM_INIT = {"weight": Ones()}
# A_log / dt_bias init for the gated-delta families (real init so the kernels get
# sane decay values in the forward tests).
_DECAY_INIT = {"A_log": GatedDeltaALog(), "dt_bias": GatedDeltaDtBias()}

# Tiny dims keep config/CPU paths fast. num_v_heads is a multiple of num_k_heads.
_DIM = 64
_NUM_K_HEADS = 2
_NUM_V_HEADS = 4
_HEAD_K_DIM = 16
_HEAD_V_DIM = 16
_CONV_SIZE = 4
_SEQ_LEN = 16
_BSZ = 2


def _qkv_config():
    packed_width = 2 * _NUM_K_HEADS * _HEAD_K_DIM + _NUM_V_HEADS * _HEAD_V_DIM
    return LinearAttnQKVConfig(
        head_k_dim=_HEAD_K_DIM,
        head_v_dim=_HEAD_V_DIM,
        num_k_heads=_NUM_K_HEADS,
        num_v_heads=_NUM_V_HEADS,
        qkv_proj=LinearConfig(
            param_init=dict(_INIT),
            in_features=_DIM,
            out_features=packed_width,
        ),
        conv=Conv1dConfig(
            param_init=dict(_INIT),
            in_channels=packed_width,
            out_channels=packed_width,
            kernel_size=_CONV_SIZE,
            groups=packed_width,
            padding=_CONV_SIZE - 1,
            bias=False,
        ),
    )


def _gdn_cfg(*, mask_type: MaskType = MaskType.CAUSAL):
    return GatedDeltaNet.Config(
        dim=_DIM,
        num_k_heads=_NUM_K_HEADS,
        num_v_heads=_NUM_V_HEADS,
        head_k_dim=_HEAD_K_DIM,
        head_v_dim=_HEAD_V_DIM,
        conv_size=_CONV_SIZE,
        qkv=_qkv_config(),
        ba_proj=LinearConfig(param_init=_INIT),
        z_proj=LinearConfig(param_init=_INIT),
        o_proj=LinearConfig(param_init=_INIT),
        o_norm=FusedRMSNormGatedConfig(param_init=_NORM_INIT),
        param_init=_DECAY_INIT,
        mask_type=mask_type,
    )


def _kda_cfg(*, mask_type: MaskType = MaskType.CAUSAL):
    return KimiDeltaAttention.Config(
        dim=_DIM,
        num_k_heads=_NUM_K_HEADS,
        num_v_heads=_NUM_V_HEADS,
        head_k_dim=_HEAD_K_DIM,
        head_v_dim=_HEAD_V_DIM,
        conv_size=_CONV_SIZE,
        qkv=_qkv_config(),
        b_proj=LinearConfig(param_init=_INIT),
        f_proj_in=LinearConfig(param_init=_INIT),
        f_proj_out=LinearConfig(param_init=_INIT),
        g_proj_in=LinearConfig(param_init=_INIT),
        g_proj_out=LinearConfig(param_init=_INIT),
        o_proj=LinearConfig(param_init=_INIT),
        o_norm=FusedRMSNormGatedConfig(param_init=_NORM_INIT),
        param_init=_DECAY_INIT,
        mask_type=mask_type,
    )


class TestLinearAttentionConfig(unittest.TestCase):
    """Config-level checks that don't require fla."""

    def test_flops_linear_in_seqlen(self):
        # Linear-attention per-token cost is sequence-independent (no s^2 term).
        cfg = _gdn_cfg()
        self.assertEqual(cfg.fwd_flops_per_token(128), cfg.fwd_flops_per_token(4096))

    def test_projections_survive_a_config_replace(self):
        # ``build()`` goes through dataclasses.replace, which re-runs
        # __post_init__. A quantization converter has by then swapped a
        # projection for its quantized twin, so rebuilding would silently
        # undo the conversion.
        import dataclasses

        from torchtitan.models.common.nn_modules import Linear

        cfg = _gdn_cfg()
        substitute = Linear.Config(
            in_features=_DIM, out_features=2 * _NUM_V_HEADS, param_init=_INIT
        )
        cfg.ba_proj = substitute
        self.assertIs(dataclasses.replace(cfg).ba_proj, substitute)

    def test_float8_conversion_survives_the_rebuild(self):
        # The real converter path: it replaces every Linear.Config in the tree,
        # and those replacements must still be there after replace().
        import dataclasses

        from torchtitan.ops.float8.cuda_gemm import FP8Linear

        cfg = _gdn_cfg()
        cfg.qkv.qkv_proj = FP8Linear.Config(
            in_features=cfg.qkv.qkv_proj.in_features,
            out_features=cfg.qkv.qkv_proj.out_features,
            param_init=_INIT,
        )
        rebuilt = dataclasses.replace(cfg)
        self.assertIsInstance(rebuilt.qkv.qkv_proj, FP8Linear.Config)

    def test_projection_of_the_wrong_size_is_rejected(self):
        from torchtitan.models.common.nn_modules import Linear

        with self.assertRaises(ValueError):
            KimiDeltaAttention.Config(
                dim=_DIM,
                num_k_heads=_NUM_K_HEADS,
                num_v_heads=_NUM_V_HEADS,
                head_k_dim=_HEAD_K_DIM,
                head_v_dim=_HEAD_V_DIM,
                conv_size=_CONV_SIZE,
                qkv=_qkv_config(),
                b_proj=LinearConfig(param_init=_INIT),
                f_proj_in=LinearConfig(param_init=_INIT),
                f_proj_out=LinearConfig(param_init=_INIT),
                g_proj_in=LinearConfig(param_init=_INIT),
                g_proj_out=LinearConfig(param_init=_INIT),
                o_norm=FusedRMSNormGatedConfig(param_init=_NORM_INIT),
                param_init=_DECAY_INIT,
                o_proj=Linear.Config(
                    in_features=_DIM + 1, out_features=_DIM, param_init=_INIT
                ),
            )

    def test_block_causal_selects_varlen_metadata(self):
        # mask_type is the knob; the mask the decoder must build follows from it.
        for maker in (_gdn_cfg, _kda_cfg):
            packed = maker(mask_type=MaskType.BLOCK_CAUSAL)
            self.assertEqual(packed.attention_mask_kind, AttentionMaskKind.VARLEN)
            plain = maker()
            self.assertEqual(plain.mask_type, MaskType.CAUSAL)
            self.assertEqual(plain.attention_mask_kind, AttentionMaskKind.NONE)

    def test_unknown_mask_type_rejected(self):
        with self.assertRaises(ValueError):
            _gdn_cfg(mask_type="not_a_mask")


class TestGiftHybridConfig(unittest.TestCase):
    """GIFT 3.5 hybrid config wiring (no fla needed to build/introspect configs)."""

    def _preset(self):
        from tests.unit_tests.config_helpers import load_experiment_config

        return load_experiment_config("gift_3_5_debug.yaml")

    def test_hybrid_runs_the_base_runtime_preparation(self):
        """The hybrid overrides hooks, not the whole method: a guard added to
        the base's runtime preparation must reach gift_3_5 and gift_4 too."""
        from torchtitan.models.gift_moe._base.model import GiftMoEModel
        from torchtitan.models.gift_moe.gift_3_5.model import GiftHybridModel

        self.assertIs(
            GiftHybridModel.prepare_config.__func__,
            GiftMoEModel.prepare_config.__func__,
        )
        # Parallelism restrictions are module capabilities checked generically.
        self.assertIs(
            GiftHybridModel.Config.validate_parallelism,
            DecoderConfig.validate_parallelism,
        )
        self.assertIs(
            GiftMoEModel.Config.validate_parallelism,
            DecoderConfig.validate_parallelism,
        )
        self.assertIsNot(
            GiftHybridModel._set_sharding_config.__func__,
            GiftMoEModel._set_sharding_config.__func__,
        )

    def test_layers_are_interleaved(self):
        cfg = self._preset()
        assert cfg.model_spec is not None
        assert isinstance(cfg.model_spec.model, DecoderConfig)
        assert cfg.model_spec.model.materialized
        families = [
            type(layer.attention).__qualname__ for layer in cfg.model_spec.model.layers
        ]
        # 6 layers, the preset places MLA at indices 2 and 5, GDN elsewhere.
        self.assertEqual(families[2], "MLAConfig")
        self.assertEqual(families[5], "MLAConfig")
        self.assertEqual(families[0], "GatedDeltaNetConfig")

    def test_mixed_mask_kind_rejected(self):
        from torchtitan.config.schema.models.gift_moe.gift_3_5.model import (
            GiftHybridModelConfig,
        )

        cfg = self._preset()
        assert cfg.model_spec is not None
        assert isinstance(cfg.model_spec.model, GiftHybridModelConfig)
        # Force one linear layer to BLOCK_CAUSAL -> VARLEN, mixing with NONE layers.
        cfg.model_spec.model.layers[0].attention.mask_type = MaskType.BLOCK_CAUSAL
        with self.assertRaisesRegex(ValueError, "mixes attention masking"):
            _ = cfg.model_spec.model.attention

    def test_tensor_parallel_rejected(self):
        cfg = self._preset()
        assert cfg.model_spec is not None
        cfg.parallelism.tensor_parallel_degree = 2
        with self.assertRaisesRegex(ValueError, "tensor_parallel_degree"):
            cfg.model_spec.model.owner().prepare_config(
                cfg.model_spec.model, config=cfg
            )

    def test_context_parallel_rejected(self):
        cfg = self._preset()
        assert cfg.model_spec is not None
        cfg.parallelism.context_parallel_degree = 2
        with self.assertRaisesRegex(ValueError, "context_parallel_degree"):
            cfg.model_spec.model.owner().prepare_config(
                cfg.model_spec.model, config=cfg
            )

    def test_removed_varlen_field_is_rejected(self):
        from tests.unit_tests.config_helpers import load_model_spec
        from torchtitan.config import ConfigSchemaError

        for preset in ("gift_moe/debugmodel.yaml", "gift_moe/gift_3_5/debugmodel.yaml"):
            with self.assertRaises(ConfigSchemaError):
                load_model_spec(preset, {"config": {"varlen": True}})
        spec = load_model_spec(
            "gift_moe/debugmodel.yaml",
            {
                "config": {
                    "blocks": {
                        name: {"attention": {"inner_attention": {"type": "varlen"}}}
                        for name in ("mla_dense", "mla_moe")
                    }
                }
            },
        )
        kinds = {layer.attention.attention_mask_kind for layer in spec.model.layers}
        self.assertEqual(kinds, {AttentionMaskKind.VARLEN})

    def test_varlen_preset_is_all_varlen(self):
        from tests.unit_tests.config_helpers import load_experiment_config

        cfg = load_experiment_config("gift_3_5_debug_varlen.yaml")
        assert cfg.model_spec is not None
        assert isinstance(cfg.model_spec.model, DecoderConfig)
        assert cfg.model_spec.model.materialized
        kinds = {
            layer.attention.attention_mask_kind for layer in cfg.model_spec.model.layers
        }
        self.assertEqual(kinds, {AttentionMaskKind.VARLEN})
        # Runtime preparation accepts the uniform all-VARLEN hybrid.
        cfg.model_spec.model.owner().prepare_config(cfg.model_spec.model, config=cfg)

    def test_kernel_drives_gdn_mask_and_rejects_flex(self):
        # The full-attention kernel determines packing for every hybrid layer.
        from tests.unit_tests.config_helpers import load_model_spec
        from torchtitan.config import ConfigSchemaError

        def layers(*, kernel):
            return load_model_spec(
                "gift_moe/gift_3_5/debugmodel.yaml",
                {
                    "config": {
                        "blocks": {
                            name: {"attention": {"inner_attention": {"type": kernel}}}
                            for name in ("mla_moe", "mtp")
                        }
                    }
                },
            ).model.layers

        def gdn_masks(layers):
            return [
                layer.attention.mask_type
                for layer in layers
                if isinstance(layer.attention, GatedDeltaNet.Config)
            ]

        none_layers = layers(kernel="sdpa")
        varlen_layers = layers(kernel="varlen")
        self.assertTrue(gdn_masks(none_layers))
        self.assertTrue(all(m is MaskType.CAUSAL for m in gdn_masks(none_layers)))
        self.assertTrue(
            all(m is MaskType.BLOCK_CAUSAL for m in gdn_masks(varlen_layers))
        )
        with self.assertRaises(ConfigSchemaError):
            layers(kernel="flex")


@unittest.skipUnless(
    _FLA_OK and _GPU_OK,
    "linear-attention forward requires flash-linear-attention and a CUDA device",
)
class TestLinearAttentionForward(unittest.TestCase):
    """Build + forward the linear families (requires fla + GPU kernels)."""

    def _build(self, cfg):
        module = cfg.build()
        module.init_states()
        return module.to("cuda").to(torch.bfloat16)

    def _run(self, cfg_cls, cfg):
        torch.manual_seed(0)
        module = self._build(cfg)
        x = torch.randn(_BSZ, _SEQ_LEN, _DIM, device="cuda", dtype=torch.bfloat16)
        out = module(x, rope_cache=None, attention_masks=None)
        self.assertIsInstance(module, cfg_cls)
        self.assertEqual(tuple(out.shape), (_BSZ, _SEQ_LEN, _DIM))
        # No hidden randomness.
        out2 = module(x, rope_cache=None, attention_masks=None)
        self.assertTrue(torch.equal(out, out2))

    def test_gdn_forward(self):
        self._run(GatedDeltaNet, _gdn_cfg())

    def test_kda_forward(self):
        self._run(KimiDeltaAttention, _kda_cfg())


@unittest.skipUnless(
    _BWD_OK,
    "linear-attention backward requires flash-linear-attention and a CUDA device; "
    "on Hopper + Triton in [3.4.0, 3.7.1) it also needs the fla tilelang backend "
    "(pip install tilelang) or Triton >= 3.7.1 (fla #640)",
)
class TestLinearAttentionBackward(unittest.TestCase):
    """Forward+backward for the linear families with GVA heads (num_v > num_k).

    Regression guard for fla #640: the *gated* delta-rule backward on Hopper +
    Triton in [3.4.0, 3.7.1) is numerically broken in the Triton kernel and must
    run through fla's tilelang backend. ``_gdn_cfg``/``_kda_cfg`` are GVA
    (num_k=2, num_v=4). Mirrors the forward test but differentiates and asserts
    finite grads on the input and every trainable parameter.
    """

    def _build(self, cfg):
        module = cfg.build()
        module.init_states()
        return module.to("cuda").to(torch.bfloat16)

    def _run(self, cfg):
        torch.manual_seed(0)
        module = self._build(cfg)
        x = torch.randn(
            _BSZ,
            _SEQ_LEN,
            _DIM,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        out = module(x, rope_cache=None, attention_masks=None)
        out.float().sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        trained = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
        missing = [n for n, p in trained if p.grad is None]
        self.assertEqual(missing, [], f"params without grad: {missing}")
        nonfinite = [n for n, p in trained if not torch.isfinite(p.grad).all()]
        self.assertEqual(nonfinite, [], f"params with non-finite grad: {nonfinite}")

    def test_gdn_backward(self):
        self._run(_gdn_cfg())

    def test_kda_backward(self):
        self._run(_kda_cfg())


@unittest.skipUnless(
    _FLA_OK and _GPU_OK,
    "linear-attention varlen forward requires flash-linear-attention and a CUDA device",
)
class TestLinearAttentionVarlen(unittest.TestCase):
    """Document-packed (varlen) correctness for the linear families."""

    _B, _S = 2, 32

    def _build(self, cfg):
        torch.manual_seed(0)
        module = cfg.build()
        module.init_states()
        return module.to("cuda").to(torch.bfloat16)

    def _meta(self, doc_lens_per_row):
        # positions reset to 0 at each document start within every row.
        row = []
        for n in doc_lens_per_row:
            row += list(range(n))
        positions = torch.tensor(row, device="cuda").unsqueeze(0).repeat(self._B, 1)
        return create_varlen_metadata_for_document(positions)

    def _equivalence(self, cfg_fn):
        # varlen with one document per row must equal the non-varlen batched path
        # (each row is then an independent sequence).
        module = self._build(cfg_fn())
        x = torch.randn(self._B, self._S, _DIM, device="cuda", dtype=torch.bfloat16)
        out_ref = module(x, None, None)
        module.varlen = True  # same weights, exercise the [1, B*S] varlen path
        out_var = module(x, None, self._meta([self._S]))
        self.assertEqual(tuple(out_var.shape), (self._B, self._S, _DIM))
        torch.testing.assert_close(out_ref, out_var, rtol=0.0, atol=1e-2)

    def _boundary(self, cfg_fn):
        # Splitting each row into two documents must change the output (the conv
        # and scan reset at the boundary).
        module = self._build(cfg_fn(mask_type=MaskType.BLOCK_CAUSAL))
        x = torch.randn(self._B, self._S, _DIM, device="cuda", dtype=torch.bfloat16)
        one_doc = module(x, None, self._meta([self._S]))
        two_docs = module(x, None, self._meta([self._S // 2, self._S // 2]))
        self.assertGreater((one_doc - two_docs).abs().max().item(), 1e-4)

    def test_gdn_varlen_matches_batched(self):
        self._equivalence(_gdn_cfg)

    def test_kda_varlen_matches_batched(self):
        self._equivalence(_kda_cfg)

    def test_gdn_varlen_boundary_resets(self):
        self._boundary(_gdn_cfg)

    def test_kda_varlen_boundary_resets(self):
        self._boundary(_kda_cfg)


if __name__ == "__main__":
    unittest.main()
