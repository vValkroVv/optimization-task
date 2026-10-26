from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import AttentionArch, ModelConfig

_NUM_LAYERS = 8


def _dense_config(**overrides):
    fields = dict(
        model_type="gigachat3_5_dense",
        architectures=["Gigachat35DenseForCausalLM"],
        num_hidden_layers=_NUM_LAYERS,
        layer_types=["linear_attention"] * (_NUM_LAYERS - 1) + ["full_attention"],
        full_attention_layers=[_NUM_LAYERS - 1],
        hidden_size=2048,
        intermediate_size=8192,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        rope_theta=100000,
        rope_parameters={"rope_theta": 100000, "rope_type": "default"},
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        attention_bias=False,
        gated_attention=True,
        attention_type="LlamaAttention",
        linear_attention_type="Qwen3NextGatedDeltaNet",
        num_nextn_predict_layers=1,
        nextn_is_sparse=False,
        first_k_dense_replace=_NUM_LAYERS,
        n_routed_experts=0,
        moe_layer_freq=1,
        q_lora_rank=None,
        kv_lora_rank=1,
        qk_nope_head_dim=128,
        qk_rope_head_dim=128,
        v_head_dim=128,
        vocab_size=1024,
        dtype="bfloat16",
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _make_model_config(hf_config, **kwargs):
    with patch(
        "sglang.srt.configs.model_config.get_config", return_value=hf_config
    ), patch(
        "sglang.srt.configs.model_config.get_generation_config", return_value=None
    ):
        return ModelConfig(model_path="dummy", **kwargs)


@pytest.mark.unit
class TestGigachat35DenseModelConfig:
    def test_dense_arch_is_normalized_to_gigachat35(self):
        model_config = _make_model_config(_dense_config())

        assert model_config.hf_config.architectures == ["Gigachat35ForCausalLM"]
        assert model_config.attention_arch == AttentionArch.MHA
        assert model_config.head_dim == 128
        assert model_config.get_num_kv_heads(1) == 8
        assert model_config.context_len == 4096

    def test_draft_model_remaps_to_nextn(self):
        model_config = _make_model_config(_dense_config(), is_draft_model=True)

        assert model_config.hf_config.architectures == ["Gigachat35ForCausalLMNextN"]
        assert model_config.hf_config.nextn_is_dsa is False
        assert model_config.attention_arch == AttentionArch.MHA
        assert model_config.head_dim == 128

    def test_draft_model_with_a_linear_mtp_block_fails_before_loading(self):
        with pytest.raises(NotImplementedError, match="linear-attention/GDN"):
            _make_model_config(_dense_config(nextn_is_linear=True), is_draft_model=True)

    def test_draft_model_with_a_legacy_linear_anchor_fails_before_loading(self):
        layer_types = ["full_attention"] + ["linear_attention"] * (_NUM_LAYERS - 1)
        with pytest.raises(NotImplementedError, match="linear-attention/GDN"):
            _make_model_config(
                _dense_config(layer_types=layer_types, full_attention_layers=[0]),
                is_draft_model=True,
            )

    def test_target_model_with_a_linear_mtp_block_is_accepted(self):
        model_config = _make_model_config(_dense_config(nextn_is_linear=True))

        assert model_config.hf_config.architectures == ["Gigachat35ForCausalLM"]

    def test_architecture_override_is_noop(self):
        model_config = _make_model_config(
            _dense_config(architectures=["Gigachat35ForCausalLM"]),
        )

        assert model_config.hf_config.architectures == ["Gigachat35ForCausalLM"]
        assert model_config.attention_arch == AttentionArch.MHA

    def test_mla_gigachat35_keeps_mla_shapes(self):
        hf_config = _dense_config(
            architectures=["Gigachat35ForCausalLM"],
            attention_type="LlamaLatentSparseAttention",
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )

        model_config = _make_model_config(hf_config)

        assert model_config.attention_arch == AttentionArch.MLA
        assert model_config.head_dim == 256
        assert model_config.kv_lora_rank == 512
        assert model_config.qk_rope_head_dim == 64
