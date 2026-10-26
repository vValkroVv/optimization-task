import unittest
from types import SimpleNamespace

import pytest

from sglang.srt.configs.model_config import (
    _get_attention_architectures,
    is_deepseek_dsa,
    multimodal_model_archs,
)
from sglang.srt.utils.hf_transformers_utils import get_hf_text_config

pytestmark = pytest.mark.unit


class TestGigaOmniConfig(unittest.TestCase):
    def test_attention_architecture_comes_from_language_model(self):
        config = SimpleNamespace(architectures=["GigaOmniForCausalLM"])

        for architecture in ["LlamaForCausalLM", "DeepseekV3ForCausalLM"]:
            with self.subTest(architecture=architecture):
                text_config = SimpleNamespace(architectures=[architecture])
                self.assertEqual(
                    _get_attention_architectures(config, text_config),
                    [architecture],
                )

    def test_text_config_is_built_from_language_model_dict(self):
        config = SimpleNamespace(
            architectures=["GigaOmniForCausalLM"],
            language_model={
                "architectures": ["DeepseekV3ForCausalLM"],
                "num_attention_heads": 16,
                "hidden_size": 1024,
            },
        )

        text_config = get_hf_text_config(config)

        self.assertEqual(text_config.num_attention_heads, 16)
        self.assertEqual(text_config.hidden_size, 1024)
        self.assertEqual(text_config.architectures, ["DeepseekV3ForCausalLM"])

    def test_gigaomni_is_registered_as_multimodal(self):
        self.assertIn("GigaOmniForCausalLM", multimodal_model_archs)

    def test_nsa_detection_looks_inside_language_model(self):
        nsa_config = SimpleNamespace(
            architectures=["GigaOmniForCausalLM"],
            language_model={
                "architectures": ["DeepseekV3ForCausalLM"],
                "index_head_dim": 128,
                "index_n_heads": 16,
                "index_topk": 2048,
            },
        )
        self.assertTrue(is_deepseek_dsa(nsa_config))

    def test_nsa_detection_is_negative_without_index_fields(self):
        plain_config = SimpleNamespace(
            architectures=["GigaOmniForCausalLM"],
            language_model={
                "architectures": ["DeepseekV3ForCausalLM"],
                "num_attention_heads": 16,
            },
        )
        self.assertFalse(is_deepseek_dsa(plain_config))

    def test_nsa_detection_is_negative_for_non_nsa_language_model(self):
        llama_config = SimpleNamespace(
            architectures=["GigaOmniForCausalLM"],
            language_model={"architectures": ["LlamaForCausalLM"]},
        )
        self.assertFalse(is_deepseek_dsa(llama_config))

    def test_nsa_detection_is_negative_without_language_model(self):
        bare_config = SimpleNamespace(architectures=["GigaOmniForCausalLM"])
        self.assertFalse(is_deepseek_dsa(bare_config))


if __name__ == "__main__":
    unittest.main()
