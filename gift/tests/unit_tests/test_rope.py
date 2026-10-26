# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import dataclasses
import math
import unittest

import torch

from torchtitan.config.schema.models.common.attention.qkv import QKVLinearConfig
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.models.common.embeddings.rope import (
    ComplexRoPE,
    CosSinRoPE,
    LlamaScaling,
    RoPE,
    YarnScaling,
    _maybe_check_max_pos,
    apply_rotary_emb_complex,
    apply_rotary_emb_cos_sin,
)


class TestApplyRotaryEmbCosSin(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.bsz = 2
        self.seqlen = 16
        self.n_heads = 4
        self.head_dim = 64
        self.xq = torch.randn(
            self.bsz, self.seqlen, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.xk = torch.randn(
            self.bsz, self.seqlen, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.rope_cache = torch.randn(
            self.seqlen, self.head_dim * 2, dtype=torch.float32
        )

    def test_output_dtype_matches_input(self):
        xq_out, xk_out = apply_rotary_emb_cos_sin(self.xq, self.xk, self.rope_cache)
        self.assertEqual(xq_out.dtype, self.xq.dtype)
        self.assertEqual(xk_out.dtype, self.xk.dtype)

    def test_output_shape_matches_input(self):
        xq_out, xk_out = apply_rotary_emb_cos_sin(self.xq, self.xk, self.rope_cache)
        self.assertEqual(xq_out.shape, self.xq.shape)
        self.assertEqual(xk_out.shape, self.xk.shape)

    def test_computes_in_fp32(self):
        """Output must match a reference computed entirely in float32.

        Ensures inductor cannot fuse away the fp32 upcast when compiling
        adjacent ops (e.g. q_norm/k_norm) with the RoPE computation.
        """
        xq_out, xk_out = apply_rotary_emb_cos_sin(self.xq, self.xk, self.rope_cache)

        cos = self.rope_cache[..., : self.head_dim].unsqueeze(0).unsqueeze(2)
        sin = self.rope_cache[..., self.head_dim :].unsqueeze(0).unsqueeze(2)

        def rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

        xq_ref = (
            (self.xq.float() * cos) + (rotate_half(self.xq.float()) * sin)
        ).bfloat16()
        xk_ref = (
            (self.xk.float() * cos) + (rotate_half(self.xk.float()) * sin)
        ).bfloat16()

        self.assertEqual((xq_out - xq_ref).abs().max().item(), 0.0)
        self.assertEqual((xk_out - xk_ref).abs().max().item(), 0.0)


class TestMaybeCheckMaxPos(unittest.TestCase):
    """Tests for the _maybe_check_max_pos bounds check."""

    def test_positions_within_bounds(self):
        positions = torch.tensor([[0, 1, 2, 3]])
        _maybe_check_max_pos(positions, max_valid_pos=3)

    def test_positions_at_boundary(self):
        positions = torch.tensor([[0, 5, 10, 15]])
        _maybe_check_max_pos(positions, max_valid_pos=15)

    def test_positions_out_of_bounds_raises(self):
        positions = torch.tensor([[0, 1, 2, 16]])
        with self.assertRaises(RuntimeError):
            _maybe_check_max_pos(positions, max_valid_pos=15)
            torch.cuda.synchronize() if torch.cuda.is_available() else None


class TestRoPEPositionBoundsComplex(unittest.TestCase):
    """RoPE complex-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_seq_len = 32
        rope_cfg = ComplexRoPE.Config(dim=self.head_dim, max_seq_len=self.max_seq_len)
        rope = rope_cfg.build()
        self.freqs_cis = rope.cache

    def test_valid_positions(self):
        bsz, seqlen = 2, 8
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)
        apply_rotary_emb_complex(xq, xk, freqs_cis=self.freqs_cis, positions=positions)

    def test_out_of_range_positions_raises(self):
        bsz, seqlen = 1, 4
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.tensor([[0, 1, self.max_seq_len, self.max_seq_len + 1]])
        with self.assertRaises(RuntimeError):
            apply_rotary_emb_complex(
                xq, xk, freqs_cis=self.freqs_cis, positions=positions
            )


class TestRoPEPositionBoundsCosSin(unittest.TestCase):
    """RoPE cos/sin-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_seq_len = 32
        rope_cfg = CosSinRoPE.Config(dim=self.head_dim, max_seq_len=self.max_seq_len)
        rope = rope_cfg.build()
        self.rope_cache = rope.cache

    def test_valid_positions(self):
        bsz, seqlen = 2, 8
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)
        apply_rotary_emb_cos_sin(xq, xk, self.rope_cache, positions)

    def test_out_of_range_positions_raises(self):
        bsz, seqlen = 1, 4
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.tensor([[0, 1, self.max_seq_len, self.max_seq_len + 1]])
        with self.assertRaises(RuntimeError):
            apply_rotary_emb_cos_sin(xq, xk, self.rope_cache, positions)


class TestPrepareConfigSeqLenValidation(unittest.TestCase):
    """Runtime preparation rejects seq_len > rope.max_seq_len."""

    def _make_trainer_config(self, seq_len):
        from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
        from torchtitan.config.schema.components.loss import ChunkedCELossConfig
        from torchtitan.config.schema.datasets.hf.text_datasets import (
            HuggingFaceTextDataLoaderConfig,
        )
        from torchtitan.config.schema.trainer import TrainerConfig

        return TrainerConfig(
            dataloader=HuggingFaceTextDataLoaderConfig(),
            loss=ChunkedCELossConfig(),
            training=dataclasses.replace(TrainingConfig(), seq_len=seq_len),
            parallelism=ParallelismConfig(),
            debug=DebugConfig(),
        )

    def _make_config(self):
        """Build a minimal Llama3 debug config."""
        from tests.unit_tests.config_helpers import load_model_spec

        return load_model_spec("llama3/debugmodel.yaml").model

    def test_rejects_oversized_seq_len(self):
        cfg = self._make_config()
        rope_max = cfg.rope.max_seq_len
        with self.assertRaises(ValueError):
            cfg.owner().prepare_config(
                cfg, config=self._make_trainer_config(rope_max + 1)
            )

    def test_accepts_valid_seq_len(self):
        cfg = self._make_config()
        rope_max = cfg.rope.max_seq_len
        cfg.owner().prepare_config(cfg, config=self._make_trainer_config(rope_max))
        self.assertEqual(cfg.rope.max_seq_len, rope_max)

    def test_vllm_max_model_len_as_seq_len(self):
        """vLLM wrapper translates max_model_len to TrainingConfig.seq_len.

        When seq_len equals rope.max_seq_len, the RoPE cache stays at
        the model's intrinsic maximum.
        """
        cfg = self._make_config()
        original_max = cfg.rope.max_seq_len
        cfg.owner().prepare_config(cfg, config=self._make_trainer_config(original_max))
        self.assertEqual(cfg.rope.max_seq_len, original_max)


class TestRopeCacheInvariants(unittest.TestCase):
    """Cache properties no hash is needed to state. The bitwise golden table for
    every shipped combination lands in its own MR."""

    def test_cache_len_only_shortens_the_cache(self):
        """Allocation only: entry ``i`` is the same number."""
        full = ComplexRoPE.Config(dim=64, max_seq_len=4096, theta=10000.0)
        short = dataclasses.replace(full, cache_len=512)
        self.assertEqual(short.num_cached_positions, 512)
        torch.testing.assert_close(
            short.build().cache, full.build().cache[:512], rtol=0, atol=0
        )

    def test_yarn_scales_even_below_original_seq_len(self):
        """The old gate compared ``max_seq_len`` (already lowered to the run's
        seq_len) against ``original_seq_len`` and did nothing below it. Gone, so a
        YaRN config scales wherever it appears."""
        common = dict(dim=64, max_seq_len=4096, theta=10000.0)
        scaled = ComplexRoPE.Config(
            **common,
            scaling=YarnScaling(rope_factor=40.0, original_seq_len=4096),
        )
        unscaled = ComplexRoPE.Config(**common)
        self.assertFalse(
            torch.equal(scaled.build().cache, unscaled.build().cache),
            "YaRN at max_seq_len == original_seq_len must still scale; a no-op "
            "here would mean the seq-len gate came back.",
        )


class TestRopeConfigValidation(unittest.TestCase):
    """The pairings a config can no longer hold silently."""

    def test_cos_sin_rejects_llama_scaling(self):
        with self.assertRaisesRegex(ValueError, "LlamaScaling"):
            CosSinRoPE.Config(dim=64, max_seq_len=64, scaling=LlamaScaling())

    def test_yarn_rejects_a_factor_that_does_not_extend(self):
        """The old config accepted it and did nothing."""
        with self.assertRaisesRegex(ValueError, "rope_factor"):
            YarnScaling(rope_factor=1.0, original_seq_len=4096)

    def test_yarn_rejects_inverted_betas(self):
        with self.assertRaisesRegex(ValueError, "beta_fast > beta_slow"):
            YarnScaling(
                rope_factor=8.0, original_seq_len=4096, beta_fast=1.0, beta_slow=32.0
            )

    def test_cos_sin_yarn_window_must_fit_the_head_dim(self):
        """Was a bare assert inside the cache build."""
        # dim=4 puts the window's high edge past dim/2 - 1 (low 0.506, high 1.087,
        # cap 1.0), which the reference math cannot represent.
        with self.assertRaisesRegex(ValueError, "correction window"):
            CosSinRoPE.Config(
                dim=4,
                max_seq_len=64,
                theta=150000.0,
                scaling=YarnScaling(rope_factor=32.0, original_seq_len=4096),
            )

    def test_cache_len_cannot_exceed_max_seq_len(self):
        with self.assertRaisesRegex(ValueError, "cache_len"):
            ComplexRoPE.Config(dim=64, max_seq_len=128, cache_len=256)

    def test_the_abstract_base_cannot_be_constructed(self):
        """The first error a port hits; it must name the replacements."""
        with self.assertRaisesRegex(ValueError, "abstract"):
            RoPE.Config(dim=64, max_seq_len=64)

    def test_cache_format_is_derived_from_the_config_type(self):
        """And agrees with the built cache, which is what ``forward`` reads."""
        for config, expected in (
            (ComplexRoPE.Config(dim=64, max_seq_len=64), True),
            (CosSinRoPE.Config(dim=64, max_seq_len=64), False),
        ):
            with self.subTest(type(config).__qualname__):
                self.assertIs(config.cache_is_complex, expected)
                self.assertIs(config.build().cache.is_complex(), expected)


class TestRopeContract(unittest.TestCase):
    """``accept_rope``: what the decoder pushes down, and what it refuses. The
    claim is that no attention config carries a second copy of the rope."""

    @staticmethod
    def _rope(**kw):
        return ComplexRoPE.Config(dim=16, max_seq_len=64, **kw)

    def _mla_config(self, **kw):
        from torchtitan.models.common.attention import MLA, MaskType
        from torchtitan.models.common.nn_modules import Linear
        from torchtitan.models.common.norms import RMSNorm

        dim, n_heads, nope, rope_dim, v_head, kv_lora = 64, 4, 16, 16, 16, 32
        return MLA.Config(
            dim=dim,
            n_heads=n_heads,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=v_head,
            wq=Linear.Config(in_features=dim, out_features=n_heads * (nope + rope_dim)),
            kv_norm=RMSNorm.Config(normalized_shape=kv_lora),
            wkv_a=Linear.Config(in_features=dim, out_features=kv_lora + rope_dim),
            wkv_b=Linear.Config(
                in_features=kv_lora, out_features=n_heads * (nope + v_head)
            ),
            wo=Linear.Config(in_features=n_heads * v_head, out_features=dim),
            mask_type=MaskType.CAUSAL,
            **kw,
        )

    def _gqa_config(self, **kw):  # noqa: D401
        from torchtitan.models.common.attention import (
            GQAttention,
            MaskType,
            ScaledDotProductAttention,
        )

        return GQAttention.Config(
            dim=64,
            n_heads=4,
            mask_type=MaskType.CAUSAL,
            inner_attention=ScaledDotProductAttention.Config(),
            # Sizing its own projections needs an init dict; irrelevant here, but
            # __post_init__ builds them before accept_rope ever runs.
            qkv_linear=QKVLinearConfig(
                wq=LinearConfig(param_init={"weight": lambda t: t}),
                wkv=LinearConfig(param_init={"weight": lambda t: t}),
            ),
            wo=LinearConfig(param_init={"weight": lambda t: t}),
            **kw,
        )

    # --- what a family takes ------------------------------------------------

    def test_mla_takes_the_factor_the_rope_scales_by(self):
        """Filled from the rope when unset, checked when stated -- so ``build()``
        needs no parent."""
        scaling_rope = self._rope(
            scaling=YarnScaling(rope_factor=8.0, original_seq_len=32)
        )
        config = self._mla_config()
        config.accept_rope(scaling_rope)
        self.assertEqual(config.yarn_factor, 8.0)
        with self.assertRaisesRegex(ValueError, "set yarn_factor=8.0"):
            self._mla_config(yarn_factor=4.0).accept_rope(scaling_rope)
        self._mla_config(yarn_factor=8.0).accept_rope(scaling_rope)  # agrees

    def test_mla_accepts_no_factor_when_the_rope_does_not_scale(self):
        self._mla_config().accept_rope(self._rope())

    def test_gqa_takes_nothing_and_stays_backend_agnostic(self):
        """A field reappearing here would be the second copy this seam removed."""
        names = {f.name for f in dataclasses.fields(self._gqa_config())}
        self.assertEqual(names & {"rope_backend", "yarn_factor"}, set())

    # --- what a family refuses ---------------------------------------------

    def test_mla_refuses_a_cos_sin_rope(self):
        config = self._mla_config()
        with self.assertRaisesRegex(ValueError, "real cache"):
            config.accept_rope(CosSinRoPE.Config(dim=16, max_seq_len=64))

    def test_mla_refuses_a_width_it_does_not_rotate(self):
        config = self._mla_config()
        with self.assertRaisesRegex(ValueError, "qk_rope_head_dim"):
            config.accept_rope(ComplexRoPE.Config(dim=32, max_seq_len=64))

    def test_gqa_refuses_a_width_it_does_not_rotate(self):
        config = self._gqa_config()
        with self.assertRaisesRegex(ValueError, "rotates"):
            config.accept_rope(ComplexRoPE.Config(dim=32, max_seq_len=64))

    def test_mla_refuses_a_factor_the_rope_contradicts(self):
        """The compensation is invisible once folded, so it cannot pass quietly."""
        with self.assertRaisesRegex(ValueError, "set yarn_factor=None"):
            self._mla_config(yarn_factor=40.0).accept_rope(self._rope())

    def test_mla_check_is_idempotent(self):
        config = self._mla_config()
        config.accept_rope(self._rope())
        config.accept_rope(self._rope())

    # --- what the runtime actually does -------------------------------------

    def test_gqa_forward_picks_the_rotation_the_cache_belongs_to(self):
        """Swapping the ``is_complex()`` branches changes every model's embedding
        and no config-level test would notice."""
        torch.manual_seed(0)
        head_dim, bsz, seqlen = 16, 2, 8
        config = self._gqa_config(head_dim=head_dim)
        attention = config.build()
        x = torch.randn(bsz, seqlen, 64)
        for rope_config, apply in (
            (
                ComplexRoPE.Config(dim=head_dim, max_seq_len=seqlen),
                lambda q, k, c: apply_rotary_emb_complex(q, k, freqs_cis=c),
            ),
            (
                CosSinRoPE.Config(dim=head_dim, max_seq_len=seqlen),
                lambda q, k, c: apply_rotary_emb_cos_sin(q, k, c),
            ),
        ):
            with self.subTest(type(rope_config).__qualname__):
                cache = rope_config.build().cache
                got = attention(x, cache, None)
                q, k, v = attention.qkv_linear(x)
                q, k = apply(q, k, cache)
                want = attention.wo(
                    attention.inner_attention(q, k, v, attention_masks=None)
                    .contiguous()
                    .view(bsz, seqlen, -1)
                )
                torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_mla_folds_the_compensation_squared(self):
        """``mscale**2``, not ``mscale``: it stands in for both q and k."""
        config = self._mla_config(mscale=0.7, yarn_factor=40.0)
        config.accept_rope(
            self._rope(scaling=YarnScaling(rope_factor=40.0, original_seq_len=32))
        )
        mscale = 0.1 * 0.7 * math.log(40.0) + 1.0
        qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.assertAlmostEqual(
            config.build().softmax_scale,
            qk_head_dim**-0.5 * mscale * mscale,
            places=12,
        )

    def test_the_decoder_pushes_the_contract_down(self):
        """Every other test calls ``accept_rope`` directly, so a broken walk would
        leave them all green."""
        from tests.unit_tests.config_helpers import load_experiment_config

        trainer_config = load_experiment_config("llama3_debug.yaml")
        model = trainer_config.model_spec.model
        model.rope = dataclasses.replace(model.rope, dim=model.rope.dim * 2)
        with self.assertRaisesRegex(ValueError, "rotates"):
            model.owner().prepare_config(model, config=trainer_config)

    # --- who must answer ----------------------------------------------------

    def test_every_rotating_family_answers(self):
        """``__subclasses__`` sees only imported classes, so the model packages
        are imported first."""
        import importlib
        import inspect

        from torchtitan.models.common.attention.base import BaseAttention

        for package in (
            "llama3",
            "llama4",
            "qwen3",
            "qwen3_vl",
            "deepseek_v3",
            "gpt_oss",
            "gift_moe",
            "flux",
        ):
            importlib.import_module(f"torchtitan.models.{package}")

        def walk(cls):
            yield cls
            for sub in cls.__subclasses__():
                yield from walk(sub)

        unanswered = []
        for cls in walk(BaseAttention):
            try:
                rotates = "apply_rotary_emb" in inspect.getsource(cls)
            except OSError:  # pragma: no cover - source-less class
                continue
            if rotates and "accept_rope" not in cls.Config.__dict__:
                unanswered.append(cls.__qualname__)
        self.assertEqual(unanswered, [])
        # The walk finding nothing would pass just as quietly.
        self.assertGreaterEqual(len(list(walk(BaseAttention))), 8)

    def test_the_base_refuses_to_stay_silent(self):
        from torchtitan.models.common.attention.base import BaseAttention

        with self.assertRaisesRegex(NotImplementedError, "does not declare"):
            BaseAttention.Config().accept_rope(self._rope())


if __name__ == "__main__":
    unittest.main()
