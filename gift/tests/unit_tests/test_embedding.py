# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import ClassVar

import torch
import torch.nn as nn

from torchtitan.models.common.embeddings import Embedding, Unembedding
from torchtitan.models.common.nn_modules import Linear
from torchtitan.config.schema.protocols.module import Normal, TruncNormal
from torchtitan.config.configurable import config


class TestEmbedding(unittest.TestCase):
    """Tests for the Embedding class used in the codebase."""

    def test_config_build(self):
        """Embedding.Config.build() creates a working embedding."""
        config = Embedding.Config(num_embeddings=100, embedding_dim=32)
        emb = config.build()
        self.assertIsInstance(emb, Embedding)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_build_without_fields_raises(self):
        """Embedding.Config() leaves the widths for the owning model to fill;
        building it without them raises."""
        with self.assertRaises((TypeError, ValueError)):
            Embedding.Config().build()

    def test_init_states(self):
        """init_states re-initializes the weight tensor."""
        config = Embedding.Config(
            num_embeddings=50,
            embedding_dim=16,
            param_init={"weight": TruncNormal(std=0.02)},
        )
        emb = config.build()

        nn.init.zeros_(emb.weight)
        self.assertTrue(torch.all(emb.weight == 0))
        emb.init_states()
        self.assertFalse(torch.all(emb.weight == 0))

    def test_custom_init_std(self):
        """Embedding respects custom mean and std."""
        config = Embedding.Config(
            num_embeddings=1000,
            embedding_dim=160,
            param_init={"weight": Normal(std=0.02, mean=0.1)},
        )
        emb = config.build()

        torch.manual_seed(42)
        emb.init_states()
        # With large amount of samples (160 * 1000) the sample statistics should
        # be close to the requested values. places=3 checks within 0.0005, which
        # is well within statistical tolerance for this sample size.
        self.assertAlmostEqual(emb.weight.mean().item(), 0.1, places=3)
        self.assertAlmostEqual(emb.weight.std().item(), 0.02, places=3)

    def test_config_pre_specified_build(self):
        """Embedding.Config with both fields pre-specified builds with no kwargs."""
        config = Embedding.Config(num_embeddings=100, embedding_dim=32)
        emb = config.build()
        self.assertIsInstance(emb, Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_partial_pre_specified(self):
        """Embedding.Config with fields specified at construction builds correctly."""
        config = Embedding.Config(num_embeddings=100, embedding_dim=32)
        emb = config.build()
        self.assertIsInstance(emb, Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_inheritance_preset(self):
        """Inheriting Embedding.Config can put fields back in __init__."""

        @config
        class PresetConfig(Embedding.Config):
            _owner_spec: ClassVar[str] = Embedding.Config._owner_spec
            num_embeddings: int = 100
            embedding_dim: int = 32

        preset_config = PresetConfig()
        emb = preset_config.build()
        self.assertIsInstance(emb, Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))


class TestUnembedding(unittest.TestCase):
    """Tests for the LM head and its optional logit soft-cap."""

    _DIM = 8
    _VOCAB = 16
    _CAP = 30.0

    def _heads(self, logit_softcap: float) -> tuple[Linear, Unembedding]:
        """A plain Linear and an Unembedding sharing the same weight."""
        plain = Linear.Config(in_features=self._DIM, out_features=self._VOCAB).build()
        head = Unembedding.Config(
            in_features=self._DIM, out_features=self._VOCAB, logit_softcap=logit_softcap
        ).build()
        head.load_state_dict(plain.state_dict())
        return plain, head

    def test_softcap_defaults_off(self):
        """The cap is opt-in; the default is a plain projection."""
        config = Unembedding.Config(in_features=self._DIM, out_features=self._VOCAB)
        self.assertEqual(config.logit_softcap, 0.0)
        self.assertIsInstance(config.build(), Linear)

    def test_softcap_off_is_bitwise_identical(self):
        """logit_softcap=0.0 takes the identity early-return: bit-for-bit a Linear.

        This is what keeps models that do not opt in numerically unchanged, so
        assert exact equality rather than a tolerance.
        """
        torch.manual_seed(0)
        x = torch.randn(2, 4, self._DIM)
        plain, head = self._heads(0.0)
        self.assertTrue(torch.equal(plain(x), head(x)))

    def test_softcap_matches_reference_formula(self):
        """cap * tanh(logits / cap), computed in float32 and cast back."""
        torch.manual_seed(0)
        # Scaled up so the raw logits comfortably exceed the cap.
        x = torch.randn(2, 4, self._DIM) * 100
        plain, head = self._heads(self._CAP)

        raw = plain(x)
        expected = (self._CAP * torch.tanh(raw.float() / self._CAP)).to(raw.dtype)
        self.assertTrue(torch.equal(head(x), expected))
        # tanh saturates to exactly 1.0 in float32, so the bound is inclusive.
        self.assertLessEqual(head(x).abs().max().item(), self._CAP)
        self.assertGreater(raw.abs().max().item(), self._CAP)

    def test_softcap_preserves_dtype(self):
        """The float32 round-trip does not leak into the output dtype."""
        torch.manual_seed(0)
        _, head = self._heads(self._CAP)
        head = head.to(torch.bfloat16)
        out = head(torch.randn(2, 4, self._DIM, dtype=torch.bfloat16))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_softcap_is_differentiable(self):
        """Gradients flow through the cap to the weight."""
        torch.manual_seed(0)
        _, head = self._heads(self._CAP)
        head(torch.randn(2, 4, self._DIM)).sum().backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertFalse(torch.all(head.weight.grad == 0))

    def test_negative_softcap_raises(self):
        """A negative cap is a config error, not a silent sign flip."""
        with self.assertRaises(ValueError):
            Unembedding.Config(
                in_features=self._DIM, out_features=self._VOCAB, logit_softcap=-1.0
            ).build()

    def test_state_dict_matches_plain_linear(self):
        """The cap adds no parameters, so checkpoints stay interchangeable."""
        plain, head = self._heads(self._CAP)
        self.assertEqual(list(head.state_dict()), list(plain.state_dict()))

    def test_flops_inherited_from_linear(self):
        """The cap is elementwise; the reported matmul FLOPs are unchanged."""
        capped = Unembedding.Config(
            in_features=self._DIM, out_features=self._VOCAB, logit_softcap=self._CAP
        )
        plain = Linear.Config(in_features=self._DIM, out_features=self._VOCAB)
        self.assertEqual(capped.fwd_flops_per_token(0), plain.fwd_flops_per_token(0))


if __name__ == "__main__":
    unittest.main()
