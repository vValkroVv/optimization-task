import unittest
from types import SimpleNamespace

import pytest

from sglang.srt.configs.model_config import (
    _get_attention_architectures,
    get_dsa_index_head_dim,
    is_deepseek_dsa,
)
from sglang.srt.utils.hf_transformers_utils import get_hf_text_config

pytestmark = pytest.mark.unit


class DeepseekV3Config(SimpleNamespace):
    pass


class TestGigaVisionConfig(unittest.TestCase):
    def test_attention_architecture_comes_from_gigavision_language_model(self):
        config = SimpleNamespace(architectures=["GigaVisionForCausalLM"])

        for architecture in ["LlamaForCausalLM", "DeepseekV3ForCausalLM"]:
            with self.subTest(architecture=architecture):
                text_config = SimpleNamespace(architectures=[architecture])
                self.assertEqual(
                    _get_attention_architectures(config, text_config),
                    [architecture],
                )

    def test_attention_architecture_stays_outer_for_other_models(self):
        config = SimpleNamespace(architectures=["DotsVLMForCausalLM"])
        text_config = SimpleNamespace(architectures=["LlamaForCausalLM"])

        self.assertEqual(
            _get_attention_architectures(config, text_config),
            ["DotsVLMForCausalLM"],
        )

    def test_inner_deepseek_config_is_marked_as_nsa_before_model_construction(self):
        llm_config = DeepseekV3Config(
            architectures=None,
            index_head_dim=128,
            index_n_heads=16,
            index_topk=2048,
        )
        config = SimpleNamespace(
            architectures=["GigaVisionForCausalLM"],
            llm_config=SimpleNamespace(config=llm_config),
        )

        text_config = get_hf_text_config(config)

        self.assertIs(text_config, llm_config)
        self.assertEqual(text_config.architectures, ["DeepseekV3ForCausalLM"])
        self.assertTrue(is_deepseek_dsa(text_config))
        self.assertTrue(is_deepseek_dsa(config))
        self.assertEqual(get_dsa_index_head_dim(config), 128)


if __name__ == "__main__":
    unittest.main()
