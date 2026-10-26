from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs import model_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.models import qwen3_5_mean_var
from sglang.srt.server_args import ServerArgs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("location", ["hf_config", "hf_text_config", None])
def test_mean_var_detection_and_immutable_weights(location):
    config = object.__new__(ModelConfig)
    config.hf_config = SimpleNamespace()
    config.hf_text_config = SimpleNamespace()
    if location:
        getattr(config, location).mean_var_schema = "gemma-mean-var-v1"
    assert config.is_mean_var == (location is not None)
    error = TokenizerControlMixin._weight_update_error(
        SimpleNamespace(model_config=config)
    )
    assert bool(error) == (location is not None)


@pytest.mark.parametrize(
    "architecture,expected",
    [
        ("Gemma4ForCausalLM", "Gemma4ForMeanVar"),
        ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForMeanVar"),
    ],
)
@pytest.mark.parametrize("schema_location", [None, "root", "text"])
def test_mean_var_dispatch_is_opt_in(
    monkeypatch, architecture, expected, schema_location
):
    text = PretrainedConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=32,
        max_position_embeddings=128,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
    )
    config = PretrainedConfig(architectures=[architecture], text_config=text)
    if schema_location:
        target = text if schema_location == "text" else config
        target.mean_var_schema = "gemma-mean-var-v1"
    monkeypatch.setattr(model_config, "get_config", Mock(return_value=config))
    monkeypatch.setattr(model_config, "get_generation_config", Mock(return_value=None))
    resolved = ModelConfig(model_path="dummy", is_embedding=True)
    assert resolved.hf_config.architectures == [
        expected if schema_location else architecture
    ]
    assert config.architectures == [architecture]
    if schema_location:
        assert not resolved.is_multimodal
    if architecture == "Gemma4ForCausalLM":
        assert resolved.is_hybrid_swa and resolved.is_hybrid_swa_compress
        assert resolved.swa_attention_layer_ids == [0]
        assert resolved.full_attention_layer_ids == [1]
    else:
        assert ServerArgs._support_mamba_cache_extra_buffer(
            SimpleNamespace(linear_attn_backend="triton"),
            resolved.hf_config.architectures[0],
        )


@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
def test_qwen_rejects_unvalidated_topologies(monkeypatch, tp, pp):
    monkeypatch.setattr(
        qwen3_5_mean_var, "get_pp_group", lambda: SimpleNamespace(world_size=pp)
    )
    monkeypatch.setattr(
        qwen3_5_mean_var, "get_tensor_model_parallel_world_size", lambda: tp
    )
    with pytest.raises(ValueError, match="TP=PP=1"):
        qwen3_5_mean_var.Qwen3_5ForMeanVar(SimpleNamespace())


def test_qwen_last_marker_head_preserves_mean_var_math(monkeypatch):
    monkeypatch.setattr(
        qwen3_5_mean_var, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(
        qwen3_5_mean_var, "get_tensor_model_parallel_world_size", lambda: 1
    )
    backbone = Mock(
        return_value=torch.tensor(
            [[99.0, 99.0], [2.0, -3.0], [4.0, -5.0]], dtype=torch.bfloat16
        )
    )
    monkeypatch.setattr(
        qwen3_5_mean_var, "Qwen3_5ForCausalLM", Mock(return_value=backbone)
    )
    config = SimpleNamespace(
        model_type="qwen3_5",
        mean_var_schema="gemma-mean-var-v1",
        text_config=SimpleNamespace(hidden_size=2),
    )
    model = qwen3_5_mean_var.Qwen3_5ForMeanVar(config)
    with torch.no_grad():
        model.mean_var_head.weight.copy_(torch.eye(2))
        model.mean_var_head.bias.zero_()
    batch = SimpleNamespace(extend_seq_lens=torch.tensor([2, 1]), dimensions=None)
    ids, positions = torch.tensor([1, 2, 3]), torch.tensor([0, 1, 0])
    output = model(ids, positions, batch)
    backbone.assert_called_once_with(ids, positions, batch)
    torch.testing.assert_close(
        output.embeddings, torch.tensor([[2.0, 3.1], [4.0, 5.1]])
    )
    assert output.embeddings.dtype == torch.float32
