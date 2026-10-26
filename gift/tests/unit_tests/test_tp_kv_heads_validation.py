# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for TP-degree / n_kv_heads divisibility validation in model configs.

Covers Issue #2574: models with GQA (n_kv_heads < n_heads) would crash deep in
the forward pass when tensor_parallel_degree > n_kv_heads. The fix adds an
early ValueError in prepare_config for llama3, llama4, qwen3, and gpt_oss.
"""

import unittest
from types import SimpleNamespace

from tests.unit_tests.config_helpers import init_policy
from torchtitan.config import ParallelismConfig
from torchtitan.models.common import ComplexRoPE, LlamaScaling
from torchtitan.models.llama3 import Llama3Model

_DIM = 256
_N_LAYERS = 2
_VOCAB_SIZE = 2048


def _make_trainer_config(tp: int, seq_len: int = 2048):
    """Minimal config stub with just the fields prepare_config reads."""
    training = SimpleNamespace(seq_len=seq_len)
    parallelism = ParallelismConfig(
        tensor_parallel_degree=tp,
        disable_loss_parallel=True,
    )
    return SimpleNamespace(training=training, parallelism=parallelism)


def _make_llama3_config(n_heads: int, n_kv_heads: int | None) -> "Llama3Model.Config":
    """Build a minimal materialized Llama3Model.Config with the given head counts."""
    from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
        LayerGroupConfig,
    )
    from torchtitan.config.schema.models.common.decoder_blocks.transformer_block import (
        ResidualTransformerBlockConfig,
    )
    from torchtitan.config.schema.models.common.ffn import LlamaFeedForwardConfig
    from torchtitan.config.schema.models.common.norms.rmsnorm import RMSNormConfig
    from torchtitan.models.common.attention import (
        GQAttention,
        ScaledDotProductAttention,
    )

    return Llama3Model.Config(
        dim=_DIM,
        vocab_size=_VOCAB_SIZE,
        norm=RMSNormConfig(),
        blocks={
            "dense": ResidualTransformerBlockConfig(
                attention=GQAttention.Config(
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    inner_attention=ScaledDotProductAttention.Config(),
                ),
                feed_forward=LlamaFeedForwardConfig(multiple_of=256),
                attention_norm=RMSNormConfig(),
                ffn_norm=RMSNormConfig(),
            )
        },
        layers=[LayerGroupConfig(block="dense", repeat=_N_LAYERS)],
        rope=ComplexRoPE.Config(
            dim=_DIM // n_heads,
            max_seq_len=4096,
            theta=500000,
            scaling=LlamaScaling(),
        ),
        init=init_policy("depth_scaled"),
    ).materialize()


class TestTPKVHeadsValidation(unittest.TestCase):
    """Validate that prepare_config rejects configs where n_heads or
    n_kv_heads are not divisible by tensor_parallel_degree."""

    # ------------------------------------------------------------------
    # n_kv_heads not divisible by TP  →  should raise
    # ------------------------------------------------------------------

    def test_llama3_kv_heads_not_divisible_raises(self):
        """n_kv_heads=2, tp=4 → fractional KV heads per rank → ValueError."""
        cfg = _make_llama3_config(n_heads=8, n_kv_heads=2)
        with self.assertRaises(ValueError):
            cfg.owner().prepare_config(cfg, config=_make_trainer_config(tp=4))

    # ------------------------------------------------------------------
    # n_heads not divisible by TP  →  should raise
    # ------------------------------------------------------------------

    def test_llama3_n_heads_not_divisible_raises(self):
        """n_heads=8 divides dim=256, but tp=3 gives fractional Q heads per rank."""
        cfg = _make_llama3_config(n_heads=8, n_kv_heads=8)
        with self.assertRaises(ValueError):
            cfg.owner().prepare_config(cfg, config=_make_trainer_config(tp=3))

    # ------------------------------------------------------------------
    # Valid configs  →  should not raise
    # ------------------------------------------------------------------

    def test_llama3_valid_gqa_does_not_raise(self):
        """n_kv_heads=8, n_heads=16, tp=4 → both divisible → no error."""
        cfg = _make_llama3_config(n_heads=16, n_kv_heads=8)
        cfg.owner().prepare_config(cfg, config=_make_trainer_config(tp=4))

    def test_llama3_mha_none_kv_heads_does_not_raise(self):
        """n_kv_heads=None (MHA, falls back to n_heads=16), tp=4 → no error."""
        cfg = _make_llama3_config(n_heads=16, n_kv_heads=None)
        cfg.owner().prepare_config(cfg, config=_make_trainer_config(tp=4))

    def test_llama3_tp1_skips_check(self):
        """tp=1 → check skipped even with indivisible n_kv_heads."""
        cfg = _make_llama3_config(n_heads=8, n_kv_heads=4)
        cfg.owner().prepare_config(cfg, config=_make_trainer_config(tp=1))


if __name__ == "__main__":
    unittest.main()
