import argparse
from unittest.mock import patch

import pytest
import torch


def _training_cfg(**overrides):
    """Embedded HF config of a small dense GQA + GDN hybrid checkpoint, training field names included."""
    cfg = {
        "model_type": "giga_mix",
        "attention_type": "LlamaAttention",
        "hidden_size": 16,
        "attention_hidden_size": 16,
        "intermediate_size": 32,
        "dense_intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "first_k_dense_replace": 4,
        "num_routed_experts": 192,
        "vocab_size": 64,
        "gated_attention": True,
        "norm_type": "GatedNorm",
        "layernorm_type": "pre_post",
        "linear_attention_type": "Qwen3NextGatedDeltaNet",
        "full_attention_layers": [1, 3],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_gating_type": "gated_rmsnorm_sigmoid",
        "linear_sigmoid_gate_scale": 2.0,
        "use_mtp": True,
        "mtp_recurrent": True,
        "mtp_block_type": "dense",
        "lm_head_logit_softcapping": 30.0,
        "rope_theta": 100000.0,
        "_name_or_path": "some/tokenizer",
    }
    cfg.update(overrides)
    return cfg


class TestGigachat35DenseConfig:
    def _config(self, **overrides):
        from models.gigachat3_5_dense.configuration_gigachat3_5_dense import Gigachat35DenseConfig

        return Gigachat35DenseConfig(**overrides)

    def test_derives_layer_types_for_hybrid_layout(self):
        cfg = self._config(
            num_hidden_layers=4,
            linear_attention_type="Qwen3NextGatedDeltaNet",
            full_attention_layers=[1, 3],
        )
        assert cfg.layer_types == ["linear_attention", "full_attention"] * 2

    def test_pure_gqa_layout_uses_full_attention_everywhere(self):
        cfg = self._config(num_hidden_layers=3, linear_attention_type=None)
        assert cfg.full_attention_layers == [0, 1, 2]
        assert cfg.layer_types == ["full_attention"] * 3

    def test_defaults_head_dim_and_kv_heads_from_attention_heads(self):
        cfg = self._config(hidden_size=32, num_attention_heads=4)
        assert cfg.head_dim == 8
        assert cfg.num_key_value_heads == 4

    def test_stale_layer_types_from_saved_config_are_recomputed(self):
        cfg = self._config(num_hidden_layers=4, linear_attention_type="Qwen3NextGatedDeltaNet", full_attention_layers=[3])
        saved = cfg.to_dict()
        saved["num_hidden_layers"] = 2
        saved["full_attention_layers"] = [1]
        reloaded = self._config(**saved)
        assert reloaded.layer_types == ["linear_attention", "full_attention"]

    def test_rejects_non_gqa_attention(self):
        with pytest.raises(ValueError, match="LlamaAttention"):
            self._config(attention_type="LlamaLatentAttention")

    def test_rejects_unknown_layernorm_and_gating_types(self):
        with pytest.raises(ValueError, match="layernorm_type"):
            self._config(layernorm_type="weird")
        with pytest.raises(ValueError, match="linear_gating_type"):
            self._config(linear_gating_type="weird")

    def test_layernorm_flags(self):
        cfg = self._config(layernorm_type="pre_post")
        assert cfg._use_pre_layernorm and cfg._use_post_layernorm
        cfg = self._config(layernorm_type="post")
        assert not cfg._use_pre_layernorm and cfg._use_post_layernorm


class TestGigachat35DenseMapper:
    def _mapper(self, **overrides):
        from base_mappers import Gigachat35DenseMapper

        mapper = Gigachat35DenseMapper()
        mapper.ensure_config(_training_cfg(**overrides), argparse.Namespace(torch_dtype="bfloat16"))
        return mapper

    def test_build_config_returns_dense_config_with_training_fields_translated(self):
        from models.gigachat3_5_dense.configuration_gigachat3_5_dense import Gigachat35DenseConfig

        cfg = self._mapper().cfg

        assert isinstance(cfg, Gigachat35DenseConfig)
        assert cfg.model_type == "gigachat3_5_dense"
        assert cfg.intermediate_size == 32
        assert cfg.head_dim == 4
        assert cfg.num_key_value_heads == 2
        assert cfg.num_nextn_predict_layers == 1
        assert cfg.final_logit_softcapping == 30.0
        assert cfg.layer_types == ["linear_attention", "full_attention"] * 2
        assert cfg.rope_theta == 100000.0
        assert cfg.dtype == torch.bfloat16
        assert cfg._auto_map_for_save == {
            "AutoConfig": "configuration_gigachat3_5_dense.Gigachat35DenseConfig",
            "AutoModel": "modeling_gigachat3_5_dense.Gigachat35DenseModel",
            "AutoModelForCausalLM": "modeling_gigachat3_5_dense.Gigachat35DenseForCausalLM",
        }

    def test_build_config_drops_training_only_fields(self):
        saved = self._mapper().cfg.to_dict()
        for key in ("num_routed_experts", "dense_intermediate_size", "attention_hidden_size",
                    "use_mtp", "mtp_recurrent", "mtp_block_type", "lm_head_logit_softcapping"):
            assert key not in saved, key
        assert not saved.get("_name_or_path")
        assert saved["architectures"] == ["Gigachat35DenseForCausalLM"]

    def test_build_config_exports_sglang_mla_moe_fields_for_the_gqa_layout(self):
        saved = self._mapper().cfg.to_dict()
        assert saved["first_k_dense_replace"] == 4
        assert saved["n_routed_experts"] == 0
        assert saved["moe_layer_freq"] == 1
        assert saved["nextn_is_sparse"] is False
        assert saved["q_lora_rank"] is None
        assert saved["kv_lora_rank"] == 1
        assert (saved["qk_nope_head_dim"], saved["qk_rope_head_dim"], saved["v_head_dim"]) == (4, 4, 4)

    def test_build_config_derives_head_dim_from_attention_hidden_size(self):
        cfg = self._mapper(hidden_size=32, attention_hidden_size=32).cfg
        assert cfg.head_dim == 8

    def test_build_config_prefers_explicit_head_dim(self):
        cfg = self._mapper(head_dim=4, attention_hidden_size=None).cfg
        assert cfg.head_dim == 4

    def test_build_config_defaults_kv_heads_to_all_heads(self):
        cfg = self._mapper(num_key_value_heads=None).cfg
        assert cfg.num_key_value_heads == 4

    def test_build_config_is_a_fixed_point_of_its_own_output(self):
        mapper = self._mapper()
        reloaded = mapper.build_config(mapper.cfg.to_dict(), argparse.Namespace(torch_dtype="bfloat16"))
        assert reloaded.to_dict() == mapper.cfg.to_dict()

    def test_build_config_without_mtp(self):
        cfg = self._mapper(use_mtp=False).cfg
        assert cfg.num_nextn_predict_layers == 0

    def test_build_config_rejects_mla_checkpoints(self):
        with pytest.raises(ValueError, match="DeepseekV3Mapper"):
            self._mapper(attention_type="LlamaLatentAttention")

    def test_build_config_rejects_moe_layers(self):
        with pytest.raises(ValueError, match="MoE layers"):
            self._mapper(first_k_dense_replace=1)

    def test_build_config_rejects_moe_mtp_block(self):
        with pytest.raises(ValueError, match="mtp_block_type"):
            self._mapper(mtp_block_type="moe")

    def test_build_config_rejects_qk_norm(self):
        with pytest.raises(ValueError, match="apply_qk_norm"):
            self._mapper(apply_qk_norm=True)

    def test_build_config_rejects_attention_bias(self):
        with pytest.raises(ValueError, match="attention_bias"):
            self._mapper(attention_bias=True)

    def test_build_config_rejects_partial_rotary(self):
        with pytest.raises(ValueError, match="partial rotary"):
            self._mapper(attention_hidden_size=32)

    def test_build_config_rejects_all_sparse_layers(self):
        with pytest.raises(ValueError, match="MoE layers"):
            self._mapper(first_k_dense_replace=None)

    def test_build_config_requires_explicit_dense_mtp_block(self):
        with pytest.raises(ValueError, match="mtp_block_type"):
            self._mapper(mtp_block_type=None)

    def test_build_config_rejects_linear_mtp_anchor_layer(self):
        with pytest.raises(ValueError, match="Dense MTP requires full attention in the MTP block"):
            self._mapper(full_attention_layers=[0, 2])

    def test_build_config_rejects_hybrid_mtp_without_full_attention_layers(self):
        with pytest.raises(ValueError, match="Dense MTP requires full attention in the MTP block"):
            self._mapper(full_attention_layers=None)

    def test_build_config_accepts_linear_last_layer_without_mtp(self):
        cfg = self._mapper(full_attention_layers=[0, 2], use_mtp=False).cfg
        assert cfg.num_nextn_predict_layers == 0

    def test_build_config_accepts_pure_gqa_layout_with_mtp(self):
        cfg = self._mapper(linear_attention_type=None, full_attention_layers=None).cfg
        assert cfg.num_nextn_predict_layers == 1

    def test_build_config_reads_dense_fields_given_only_in_llmfoundry_yaml(self, tmp_path):
        import yaml
        from base_mappers import Gigachat35DenseMapper
        from utils.llmfoundry_config_utils import enrich_hf_config_from_llmfoundry_yaml

        yaml_only = {
            "head_dim": 8,
            "attention_hidden_size": 32,
            "apply_qk_norm": False,
            "attention_bias": False,
            "mtp_block_type": "dense",
            "mtp_recurrent": True,
        }
        embedded = {k: v for k, v in _training_cfg(mtp_predictor_num=3).items() if k not in yaml_only}
        yaml_path = tmp_path / "llm_foundry_config.yaml"
        yaml_path.write_text(yaml.safe_dump({"model": {"params": yaml_only}}))
        cfg = Gigachat35DenseMapper().build_config(
            enrich_hf_config_from_llmfoundry_yaml(embedded, yaml_path=yaml_path),
            argparse.Namespace(torch_dtype="bfloat16"),
        )

        assert cfg.head_dim == 8
        assert cfg.num_nextn_predict_layers == 1

    def test_map_splits_fused_qkv_for_full_attention_layers(self):
        mapper = self._mapper()
        # heads=4, kv_heads=2, head_dim=4 → q 16 rows, k 8, v 8
        qkv = torch.randn(32, 16)
        o_w = torch.randn(16, 16)
        gate_w = torch.randn(16, 16)

        out = mapper.map({
            "model.layers.1.self_attn._attn.qkv_proj.weight": qkv,
            "model.layers.1.self_attn._attn.o_proj.weight": o_w,
            "model.layers.1.self_attn._attn.gate_proj.weight": gate_w,
        })

        assert set(out) == {
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.self_attn.gate_proj.weight",
        }
        torch.testing.assert_close(out["model.layers.1.self_attn.q_proj.weight"], qkv[:16].to(torch.bfloat16))
        torch.testing.assert_close(out["model.layers.1.self_attn.k_proj.weight"], qkv[16:24].to(torch.bfloat16))
        torch.testing.assert_close(out["model.layers.1.self_attn.v_proj.weight"], qkv[24:].to(torch.bfloat16))
        torch.testing.assert_close(out["model.layers.1.self_attn.o_proj.weight"], o_w.to(torch.bfloat16))

    def test_map_keeps_linear_attention_qkv_on_the_gdn_path(self):
        mapper = self._mapper()
        qkv = torch.randn(12, 16)
        z = torch.randn(4, 16)

        first = mapper.map({"model.layers.0.self_attn._attn.qkv_proj.weight": qkv})
        second = mapper.map({"model.layers.0.self_attn._attn.z_proj.weight": z})

        assert first == {}
        assert set(second) == {"model.layers.0.self_attn.in_proj_qkvz.weight"}
        assert mapper.flush() == {}

    def test_map_splits_fused_qkv_of_dense_mtp_block(self):
        mapper = self._mapper()
        qkv = torch.randn(32, 16)

        out = mapper.map({
            "model.mtp_block.mtp_layers.0.decoder_layer.self_attn.base_attn.qkv_proj.weight": qkv,
            "model.mtp_block.mtp_layers.0.decoder_layer.self_attn.base_attn.o_proj.weight": torch.randn(16, 16),
        })

        assert set(out) == {
            "model.layers.4.self_attn.q_proj.weight",
            "model.layers.4.self_attn.k_proj.weight",
            "model.layers.4.self_attn.v_proj.weight",
            "model.layers.4.self_attn.o_proj.weight",
        }

    def test_map_rejects_fused_qkv_with_unexpected_rows(self):
        mapper = self._mapper()
        with pytest.raises(ValueError, match="expected fused qkv"):
            mapper.map({"model.layers.1.self_attn._attn.qkv_proj.weight": torch.randn(30, 16)})

    def test_map_passes_dense_mlp_and_norms_through(self):
        mapper = self._mapper()
        out = mapper.map({
            "model.layers.1.self_attn.mlp.gate_proj.weight": torch.randn(32, 16),
            "model.layers.1.self_attn.input_layernorm.weight": torch.randn(16),
            "model.layers.1.self_attn.post_feedforward_layernorm.gate_up_projection.weight": torch.randn(4, 16),
        })
        assert set(out) == {
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "model.layers.1.post_feedforward_layernorm.gate_up_projection.weight",
        }

    @patch("base_mappers.write_hf_assembly_manifest")
    def test_export_manifest_records_dense_family(self, mock_write):
        mapper = self._mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.write_export_manifest(_training_cfg(), args, "out")

        mock_write.assert_called_once_with("out", args, model_family="gigachat3_5_dense")

    @patch("base_mappers._save_config_and_tokenizer", return_value=(None, None, None))
    def test_save_meta_vendors_dense_runtime_files(self, mock_save, tmp_path):
        mapper = self._mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.save_meta(_training_cfg(), args, str(tmp_path))

        mock_save.assert_called_once()
        assert (tmp_path / "configuration_gigachat3_5_dense.py").is_file()
        assert (tmp_path / "modeling_gigachat3_5_dense.py").is_file()
        assert not (tmp_path / "modeling_gigachat3_5.py").exists()



class TestGigachat35DenseModel:
    def _model(self, **overrides):
        from models.gigachat3_5_dense.configuration_gigachat3_5_dense import Gigachat35DenseConfig
        from models.gigachat3_5_dense.modeling_gigachat3_5_dense import Gigachat35DenseForCausalLM

        config = Gigachat35DenseConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, **overrides,
        )
        return Gigachat35DenseForCausalLM(config).eval()

    def test_pure_gqa_model_generates(self):
        model = self._model()
        out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, do_sample=False)
        assert out.shape == (1, 5)

    def test_hybrid_model_generates(self, monkeypatch):
        from models.gigachat3_5_dense import modeling_gigachat3_5_dense as modeling

        # Force the torch fallbacks: the fla/causal-conv1d kernels need a GPU
        for kernel in ("FusedRMSNormGated", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
                       "causal_conv1d_fn", "causal_conv1d_update"):
            monkeypatch.setattr(modeling, kernel, None)
        model = self._model(
            linear_attention_type="Qwen3NextGatedDeltaNet", full_attention_layers=[1],
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=4, linear_value_head_dim=4,
        )
        out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, do_sample=False)
        assert out.shape == (1, 5)
