# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

from tests.unit_tests.config_helpers import load_model_spec, PRESET_ROOT
from torchtitan.config import deep_merge, load_raw_document, ParallelismConfig
from torchtitan.config.configurable import to_plain
from torchtitan.config.decode import decode
from torchtitan.config.schema.protocols.module import Normal, Skip
from torchtitan.config.sections import ModelSection
from torchtitan.config.schema.models.qwen3.model import (
    Qwen3ModelConfig,
)
from torchtitan.tools.logging import logger

from torchtitan.models.llama3.model import Llama3Model


def _make_config(enable_weight_tying: bool = False) -> Llama3Model.Config:
    # Start from the standard debugmodel config and adjust weight tying; the
    # config skips the tok_embeddings init when tied (it shares lm_head's weight).
    model_config = load_model_spec(
        "llama3/debugmodel.yaml",
        {
            "config": {
                "blocks": {
                    "dense": {"attention": {"inner_attention": {"type": "sdpa"}}}
                },
                "enable_weight_tying": enable_weight_tying,
            }
        },
    ).model
    assert isinstance(model_config, Llama3Model.Config)
    return model_config


class TestLlama3WeightTying(unittest.TestCase):
    def test_weights_are_shared_when_tying_enabled(self):
        """tok_embeddings.weight and output.weight should share the same storage."""
        model = Llama3Model(_make_config(enable_weight_tying=True))
        self.assertIs(
            model.tok_embeddings.weight,
            model.lm_head.weight,
            "tok_embeddings.weight and output.weight must be the same tensor object",
        )

    def test_weights_are_independent_when_tying_disabled(self):
        """Without weight tying, tok_embeddings and output have separate weights."""
        model = Llama3Model(_make_config(enable_weight_tying=False))
        self.assertIsNot(
            model.tok_embeddings.weight,
            model.lm_head.weight,
            "tok_embeddings.weight and output.weight must be distinct tensor objects",
        )

    def test_weights_remain_tied_after_init_states(self):
        """Weights must still be shared after calling init_states."""
        config = _make_config(enable_weight_tying=True)
        model = Llama3Model(config)
        model.init_states()
        self.assertIs(
            model.tok_embeddings.weight,
            model.lm_head.weight,
            "tok_embeddings.weight and output.weight must remain tied after init_states",
        )

    def test_tied_parameter_count_matches_unique_parameters(self):
        """Tied embeddings should be counted once, not subtracted away."""
        config = _make_config(enable_weight_tying=True)
        model = Llama3Model(config)
        unique_param_count = sum(
            p.numel() for p in {id(p): p for p in model.parameters()}.values()
        )
        reported_param_count, _ = config.owner().get_nparams_and_flops(
            config, model, seq_len=512
        )

        self.assertEqual(reported_param_count, unique_param_count)

    def test_pp_guard_raises_when_weight_tying_and_pp_enabled(self):
        """Runtime preparation rejects weight tying with pipeline parallelism."""
        config = _make_config(enable_weight_tying=True)

        runtime_config = SimpleNamespace(
            parallelism=ParallelismConfig(pipeline_parallel_degree=2)
        )

        with self.assertRaisesRegex(ValueError, "Weight tying.*Pipeline Parallel"):
            config.owner().prepare_config(config, config=runtime_config)

    def test_pp_guard_does_not_raise_without_weight_tying(self):
        """Runtime preparation accepts pipeline parallelism without weight tying."""
        config = _make_config(enable_weight_tying=False)

        runtime_config = SimpleNamespace(
            parallelism=ParallelismConfig(pipeline_parallel_degree=2)
        )

        # Should not raise
        config.owner().prepare_config(config, config=runtime_config)


def _qwen3_section(**config) -> ModelSection:
    """The compact qwen3 debugmodel section with *config* layered on."""
    model = deep_merge(
        load_raw_document(str(PRESET_ROOT / "qwen3/debugmodel.yaml")).data["model"],
        {"config": config},
    )
    return decode(ModelSection, model, path="model")


class TestWeightTyingInit(unittest.TestCase):
    """The tied embedding's Skip is derived at materialize, never written into
    the ``init`` policy that resolved_config.yaml stores."""

    def test_tied_embedding_skips_init(self):
        section = _qwen3_section(enable_weight_tying=True)
        assert isinstance(section.config, Qwen3ModelConfig)
        model = section.config.materialize()
        assert model.tok_embeddings is not None
        self.assertEqual(model.tok_embeddings.param_init, {"weight": Skip()})
        self.assertEqual(section.config.init.embedding, {"weight": Normal(std=1.0)})

    def test_untying_a_resolved_config_restores_embedding_init(self):
        # What resolved_config.yaml stores, then rerun with tying turned off.
        data = to_plain(_qwen3_section(enable_weight_tying=True))
        data["config"]["enable_weight_tying"] = False
        model = decode(ModelSection, data, path="model").config.materialize()
        self.assertEqual(model.tok_embeddings.param_init, {"weight": Normal(std=1.0)})

    def test_embedding_init_with_tying_is_skipped_silently(self):
        # The tie skips the embedding's init (the LM head initializes the
        # shared weight) without a warning.
        section = _qwen3_section(
            enable_weight_tying=True,
            init={"embedding": {"weight": {"type": "normal", "std": 0.5}}},
        )
        with self.assertNoLogs(logger, "WARNING"):
            model = section.config.materialize()
        assert isinstance(model, Qwen3ModelConfig)
        assert model.tok_embeddings is not None
        self.assertEqual(model.tok_embeddings.param_init, {"weight": Skip()})

    def test_default_embedding_init_with_tying_is_silent(self):
        section = _qwen3_section(enable_weight_tying=True)
        with self.assertNoLogs(logger, "WARNING"):
            section.config.materialize()


if __name__ == "__main__":
    unittest.main()
