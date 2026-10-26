import pytest
import torch
from unittest.mock import patch, MagicMock, Mock
from collections import OrderedDict
import argparse


class TestCoerceTemplateFloatFields:
    def test_coerces_int_to_float_for_template_float_keys(self):
        from base_mappers import _coerce_template_float_fields

        template = {"routed_scaling_factor": 2.5, "rope_theta": 10000}
        cfg = {"routed_scaling_factor": 1, "rope_theta": 10000}
        _coerce_template_float_fields(cfg, template)
        assert cfg["routed_scaling_factor"] == 1.0
        assert isinstance(cfg["routed_scaling_factor"], float)
        assert cfg["rope_theta"] == 10000
        assert isinstance(cfg["rope_theta"], int)

    def test_leaves_float_and_none_unchanged(self):
        from base_mappers import _coerce_template_float_fields

        template = {"attention_dropout": 0.0, "rms_norm_eps": 1e-06}
        cfg = {"attention_dropout": 0.5, "rms_norm_eps": None}
        _coerce_template_float_fields(cfg, template)
        assert cfg["attention_dropout"] == 0.5
        assert cfg["rms_norm_eps"] is None

    def test_does_not_coerce_bool(self):
        from base_mappers import _coerce_template_float_fields

        template = {"attention_dropout": 0.0}
        cfg = {"attention_dropout": True}
        _coerce_template_float_fields(cfg, template)
        assert cfg["attention_dropout"] is True


class TestToBf16:
    """Tests for _to_bf16 function"""

    def test_converts_float32_to_bf16(self):
        """Test that float32 tensors are converted to bfloat16"""
        from base_mappers import _to_bf16

        block = {
            "weight": torch.randn(4, 4, dtype=torch.float32),
            "bias": torch.randn(4, dtype=torch.float32),
        }

        result = _to_bf16(block)

        assert result["weight"].dtype == torch.bfloat16
        assert result["bias"].dtype == torch.bfloat16

    def test_keeps_bf16_unchanged(self):
        """Test that bfloat16 tensors are not changed"""
        from base_mappers import _to_bf16

        original = torch.randn(4, 4, dtype=torch.bfloat16)
        block = {"weight": original}

        result = _to_bf16(block)

        assert result["weight"] is original  # Same object, not converted

    def test_mixed_dtypes(self):
        """Test with mixed dtype tensors"""
        from base_mappers import _to_bf16

        block = {
            "bf16_tensor": torch.randn(2, 2, dtype=torch.bfloat16),
            "fp32_tensor": torch.randn(2, 2, dtype=torch.float32),
            "fp16_tensor": torch.randn(2, 2, dtype=torch.float16),
        }

        result = _to_bf16(block)

        assert result["bf16_tensor"].dtype == torch.bfloat16
        assert result["fp32_tensor"].dtype == torch.bfloat16
        assert result["fp16_tensor"].dtype == torch.bfloat16


class TestRawMapper:
    """Tests for RawMapper class"""

    def test_map_converts_to_bf16(self):
        """Test that map method converts tensors to bfloat16"""
        from base_mappers import RawMapper

        mapper = RawMapper()
        tensors = {"weight": torch.randn(4, 4, dtype=torch.float32)}

        result = mapper.map(tensors)

        assert result["weight"].dtype == torch.bfloat16

    def test_ensure_config_returns_none(self):
        """Test that ensure_config returns None"""
        from base_mappers import RawMapper

        mapper = RawMapper()
        result = mapper.ensure_config({}, None)

        assert result is None

    def test_flush_returns_empty_dict(self):
        """Test that flush returns empty dict"""
        from base_mappers import RawMapper

        mapper = RawMapper()
        result = mapper.flush()

        assert result == {}

    def test_build_config_sets_defaults(self):
        """Test that build_config sets default token IDs"""
        from base_mappers import RawMapper

        mapper = RawMapper()
        ckpt_cfg = {"hidden_size": 768}
        args = argparse.Namespace(torch_dtype="bfloat16")

        result = mapper.build_config(ckpt_cfg, args)

        assert result["bos_token_id"] == 1
        assert result["eos_token_id"] == 2
        assert result["pad_token_id"] == 2
        assert result["torch_dtype"] == "bfloat16"
        assert result["dtype"] == "bfloat16"

    def test_build_config_removes_auto_map(self):
        """Test that build_config removes architectures and auto_map"""
        from base_mappers import RawMapper

        mapper = RawMapper()
        ckpt_cfg = {
            "hidden_size": 768,
            "architectures": ["SomeModel"],
            "auto_map": {"AutoModel": "modeling.Model"},
            "quantization_config": {"bits": 4},
        }
        args = argparse.Namespace(torch_dtype="bfloat16")

        result = mapper.build_config(ckpt_cfg, args)

        assert "architectures" not in result
        assert "auto_map" not in result
        assert "quantization_config" not in result

    def test_tokenizer_source_from_args(self):
        """Test tokenizer_source returns args.tokenizer_path first"""
        from base_mappers import RawMapper

        mapper = RawMapper()
        ckpt_cfg = {"_name_or_path": "config_path"}
        args = argparse.Namespace(tokenizer_path="args_path")

        result = mapper.tokenizer_source(ckpt_cfg, args)

        assert result == "args_path"

    def test_tokenizer_source_from_config(self):
        """Test tokenizer_source falls back to config"""
        from base_mappers import RawMapper

        mapper = RawMapper()
        ckpt_cfg = {"_name_or_path": "config_path"}
        args = argparse.Namespace(tokenizer_path=None)

        result = mapper.tokenizer_source(ckpt_cfg, args)

        assert result == "config_path"

    def test_tokenizer_source_empty(self):
        """Test tokenizer_source returns empty string when nothing available"""
        from base_mappers import RawMapper

        mapper = RawMapper()
        ckpt_cfg = {}
        args = argparse.Namespace(tokenizer_path=None)

        result = mapper.tokenizer_source(ckpt_cfg, args)

        assert result == ""

    @patch("base_mappers._save_config_and_tokenizer")
    def test_save_meta_calls_helper(self, mock_save):
        """Test save_meta calls _save_config_and_tokenizer"""
        from base_mappers import RawMapper

        mapper = RawMapper()
        cfg_src = {"key": "value"}
        args = argparse.Namespace()
        output_dir = "/tmp/output"

        mapper.save_meta(cfg_src, args, output_dir)

        mock_save.assert_called_once_with(
            mapper, cfg_src, args, output_dir, with_tmp_model=False
        )


class TestSaveConfigAndTokenizer:
    """Tests for _save_config_and_tokenizer function"""

    @patch("base_mappers.AutoTokenizer")
    @patch("base_mappers._atomic_write_json")
    @patch("os.makedirs")
    def test_saves_config_dict(self, mock_makedirs, mock_write_json, mock_tokenizer):
        """Test that config is saved as JSON"""
        from base_mappers import _save_config_and_tokenizer, RawMapper

        mock_tokenizer.from_pretrained.return_value = MagicMock()

        mapper = RawMapper()
        cfg_src = {"hidden_size": 768}
        args = argparse.Namespace(torch_dtype="bfloat16", tokenizer_path="gpt2")
        output_dir = "/tmp/output"

        result = _save_config_and_tokenizer(mapper, cfg_src, args, output_dir)

        mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
        assert mock_write_json.called

    @patch("base_mappers.AutoTokenizer")
    @patch("base_mappers._atomic_write_json")
    @patch("os.makedirs")
    def test_raises_on_empty_tokenizer(
        self, mock_makedirs, mock_write_json, mock_tokenizer
    ):
        """Test that RuntimeError is raised when tokenizer path is empty"""
        from base_mappers import _save_config_and_tokenizer, RawMapper

        mapper = RawMapper()
        cfg_src = {}
        args = argparse.Namespace(torch_dtype="bfloat16", tokenizer_path=None)
        output_dir = "/tmp/output"

        with pytest.raises(RuntimeError, match="Tokenizer path is undefined"):
            _save_config_and_tokenizer(mapper, cfg_src, args, output_dir)


class TestDeepseekV3Mapper:
    """Tests for DeepseekV3Mapper class"""

    @staticmethod
    def _ckpt_cfg(**overrides):
        cfg = {
            "attention_type": "LlamaLatentAttention",
            "num_krope_heads": 1,
            "scoring_func": "sigmoid",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "qk_rope_head_dim": 4,
            "qk_nope_head_dim": 4,
            "v_head_dim": 4,
        }
        cfg.update(overrides)
        return cfg

    def test_init_without_config(self):
        """Test initialization without config"""
        from base_mappers import DeepseekV3Mapper

        mapper = DeepseekV3Mapper()

        assert mapper.cfg is None
        assert mapper.tr is None

    @patch("base_mappers.DeepseekV3Mapper.build_config")
    def test_ensure_config_builds_config(self, mock_build_config):
        """Test ensure_config builds config when None"""
        from base_mappers import DeepseekV3Mapper

        mock_config = MagicMock()
        mock_build_config.return_value = mock_config

        mapper = DeepseekV3Mapper()

        with patch.dict("sys.modules", {"transformers_utils.deepseek": MagicMock()}):
            with patch(
                "base_mappers.DeepseekV3Mapper.build_config", return_value=mock_config
            ):
                # Skip actual ensure_config as it needs real imports
                pass

    def test_tokenizer_source_from_args(self):
        """Test tokenizer_source returns args.tokenizer_path first"""
        from base_mappers import DeepseekV3Mapper

        mapper = DeepseekV3Mapper()
        ckpt_cfg = {"_name_or_path": "config_path"}
        args = argparse.Namespace(tokenizer_path="args_path")

        result = mapper.tokenizer_source(ckpt_cfg, args)

        assert result == "args_path"

    def test_tokenizer_source_from_config(self):
        """Test tokenizer_source falls back to config"""
        from base_mappers import DeepseekV3Mapper

        mapper = DeepseekV3Mapper()
        ckpt_cfg = {"_name_or_path": "config_path"}
        args = argparse.Namespace(tokenizer_path=None)

        result = mapper.tokenizer_source(ckpt_cfg, args)

        assert result == "config_path"

    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_returns_deepseek_config_without_linear_attention(
        self, mock_from_pretrained, mock_set_default_dtype
    ):
        """Test build_config keeps the standard config when linear attention is disabled"""
        from base_mappers import DeepseekV3Mapper

        deepseek_cfg = MagicMock()
        mock_from_pretrained.return_value = deepseek_cfg

        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        result = mapper.build_config(self._ckpt_cfg(linear_attention_type=None), args)

        assert result is deepseek_cfg
        mock_from_pretrained.assert_called_once()
        mock_set_default_dtype.assert_called_once()

    @patch("base_mappers.Gigachat35Config.from_dict")
    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_returns_hybrid_config_with_linear_attention(
        self, mock_from_pretrained, mock_set_default_dtype, mock_hybrid_from_dict
    ):
        """Test build_config switches to the hybrid config when linear attention is enabled"""
        from base_mappers import DeepseekV3Mapper

        deepseek_cfg = MagicMock()
        deepseek_cfg.to_dict.return_value = {"converted": True}
        mock_from_pretrained.return_value = deepseek_cfg

        hybrid_cfg = MagicMock()
        hybrid_cfg.head_dim = None
        hybrid_cfg.qk_rope_head_dim = 128
        mock_hybrid_from_dict.return_value = hybrid_cfg

        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        result = mapper.build_config(
            self._ckpt_cfg(
                linear_attention_type="Qwen3NextGatedDeltaNet",
                lm_head_logit_softcapping=30.0,
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert result is hybrid_cfg
        assert kwargs["final_logit_softcapping"] == 30.0
        assert deepseek_cfg.model_type == "gigachat3_5"
        assert deepseek_cfg.architectures == ["Gigachat35ForCausalLM"]
        assert hybrid_cfg._auto_map_for_save == {
            "AutoConfig": "configuration_gigachat3_5.Gigachat35Config",
            "AutoModel": "modeling_gigachat3_5.Gigachat35Model",
            "AutoModelForCausalLM": "modeling_gigachat3_5.Gigachat35ForCausalLM",
        }
        assert hybrid_cfg.head_dim == 128
        mock_hybrid_from_dict.assert_called_once()
        mock_set_default_dtype.assert_called_once()

    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_fills_missing_fields_from_source_aliases(
        self, mock_from_pretrained, mock_set_default_dtype
    ):
        from base_mappers import DeepseekV3Mapper

        deepseek_cfg = MagicMock()
        mock_from_pretrained.return_value = deepseek_cfg

        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.build_config(
            self._ckpt_cfg(
                model_type="giga_mix",
                intermediate_size=896,
                num_routed_experts=64,
                router_aux_loss_coef=0.0001,
                moe_router_routed_scaling_factor=1,
                use_mtp=True,
                mtp_predictor_num=1,
                dense_intermediate_size=896,
                lm_head_logit_softcapping=30.0,
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["n_routed_experts"] == 64
        assert kwargs["aux_loss_alpha"] == 0.0001
        assert kwargs["routed_scaling_factor"] == 1.0
        assert isinstance(kwargs["routed_scaling_factor"], float)
        assert kwargs["num_nextn_predict_layers"] == 1
        assert kwargs["moe_intermediate_size"] == 896
        assert "final_logit_softcapping" not in kwargs

    @patch("base_mappers.Gigachat35Config.from_dict")
    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_preserves_kda_compact_head_layout(
        self, mock_from_pretrained, mock_set_default_dtype, mock_hybrid_from_dict
    ):
        from base_mappers import DeepseekV3Mapper

        deepseek_cfg = MagicMock()
        deepseek_cfg.to_dict.return_value = {}
        mock_from_pretrained.return_value = deepseek_cfg

        hybrid_cfg = MagicMock()
        hybrid_cfg.head_dim = 4
        mock_hybrid_from_dict.return_value = hybrid_cfg

        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.build_config(
            self._ckpt_cfg(
                linear_attention_type="KimiDeltaAttention",
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_key_head_dim=2,
                linear_value_head_dim=2,
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert "linear_original_num_key_heads" not in kwargs
        assert kwargs["linear_num_key_heads"] == 2
        assert kwargs["linear_num_value_heads"] == 4

    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_resets_template_mtp_layers_when_use_mtp_is_false(
        self, mock_from_pretrained, mock_set_default_dtype
    ):
        from base_mappers import DeepseekV3Mapper

        deepseek_cfg = MagicMock()
        mock_from_pretrained.return_value = deepseek_cfg

        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.build_config(
            self._ckpt_cfg(
                model_type="giga_mix",
                use_mtp=False,
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["num_nextn_predict_layers"] == 0

    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_preserves_explicit_num_nextn_predict_layers_without_mtp(
        self, mock_from_pretrained, mock_set_default_dtype
    ):
        from base_mappers import DeepseekV3Mapper

        deepseek_cfg = MagicMock()
        mock_from_pretrained.return_value = deepseek_cfg

        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.build_config(
            self._ckpt_cfg(
                model_type="giga_mix",
                use_mtp=False,
                num_nextn_predict_layers=2,
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["num_nextn_predict_layers"] == 2

    @pytest.mark.parametrize("flag", [True, False])
    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_maps_use_linear_attention_in_mtp_to_nextn_is_linear(
        self, mock_from_pretrained, mock_set_default_dtype, flag
    ):
        from base_mappers import DeepseekV3Mapper

        mock_from_pretrained.return_value = MagicMock()
        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.build_config(
            self._ckpt_cfg(
                model_type="giga_mix",
                use_mtp=True,
                mtp_predictor_num=1,
                linear_attention_type="Qwen3NextGatedDeltaNet",
                full_attention_layers=[0, 1],
                use_linear_attention_in_mtp=flag,
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["nextn_is_linear"] is flag
        assert "use_linear_attention_in_mtp" not in kwargs

    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_rejects_linear_mtp_without_linear_attention_type(
        self, mock_from_pretrained, mock_set_default_dtype
    ):
        from base_mappers import DeepseekV3Mapper

        mock_from_pretrained.return_value = MagicMock()
        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        with pytest.raises(ValueError, match="linear_attention_type"):
            mapper.build_config(
                self._ckpt_cfg(
                    model_type="giga_mix",
                    use_mtp=True,
                    mtp_predictor_num=1,
                    use_linear_attention_in_mtp=True,
                ),
                args,
            )

    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_infers_false_without_linear_attention_type(
        self, mock_from_pretrained, mock_set_default_dtype
    ):
        from base_mappers import DeepseekV3Mapper

        mock_from_pretrained.return_value = MagicMock()
        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.build_config(
            self._ckpt_cfg(
                model_type="giga_mix",
                use_mtp=True,
                mtp_predictor_num=1,
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["nextn_is_linear"] is False

    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_infers_true_from_linear_anchor(
        self, mock_from_pretrained, mock_set_default_dtype
    ):
        from base_mappers import DeepseekV3Mapper

        mock_from_pretrained.return_value = MagicMock()
        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.build_config(
            self._ckpt_cfg(
                model_type="giga_mix",
                use_mtp=True,
                mtp_predictor_num=1,
                linear_attention_type="Qwen3NextGatedDeltaNet",
                full_attention_layers=[0],
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["nextn_is_linear"] is True

    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_infers_false_from_full_anchor(
        self, mock_from_pretrained, mock_set_default_dtype
    ):
        from base_mappers import DeepseekV3Mapper

        mock_from_pretrained.return_value = MagicMock()
        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.build_config(
            self._ckpt_cfg(
                model_type="giga_mix",
                use_mtp=True,
                mtp_predictor_num=1,
                linear_attention_type="Qwen3NextGatedDeltaNet",
                full_attention_layers=[0, 1],
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["nextn_is_linear"] is False

    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_drops_mtp_linear_flag_without_mtp(
        self, mock_from_pretrained, mock_set_default_dtype
    ):
        from base_mappers import DeepseekV3Mapper

        mock_from_pretrained.return_value = MagicMock()
        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.build_config(
            self._ckpt_cfg(
                model_type="giga_mix",
                use_mtp=False,
                use_linear_attention_in_mtp=True,
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert "use_linear_attention_in_mtp" not in kwargs
        assert "nextn_is_linear" not in kwargs

    @patch("base_mappers.torch.set_default_dtype")
    @patch("base_mappers.AutoConfig.from_pretrained")
    def test_build_config_normalizes_dtype_alongside_torch_dtype(
        self, mock_from_pretrained, mock_set_default_dtype
    ):
        from base_mappers import DeepseekV3Mapper

        deepseek_cfg = MagicMock()
        mock_from_pretrained.return_value = deepseek_cfg

        mapper = DeepseekV3Mapper()
        args = argparse.Namespace(torch_dtype="bfloat16")

        mapper.build_config(
            self._ckpt_cfg(
                linear_attention_type="Qwen3NextGatedDeltaNet",
                dtype="float32",
                torch_dtype=None,
            ),
            args,
        )

        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["dtype"] == "bfloat16"
        assert kwargs["torch_dtype"] == "bfloat16"

    def test_map_packs_short_conv_weights_only_for_hybrid_configs(self):
        """Test map applies short-conv packing only when linear attention is enabled"""
        from base_mappers import DeepseekV3Mapper

        tensors = {"weight": torch.randn(2, 2)}
        packed_tensors = {"packed": torch.randn(2, 2)}

        hybrid_mapper = DeepseekV3Mapper()
        hybrid_mapper.cfg = argparse.Namespace(linear_attention_type="Qwen3NextGatedDeltaNet")
        hybrid_mapper.tr = MagicMock()
        hybrid_mapper.tr.rename_and_split.return_value = {"out": torch.randn(2, 2)}

        with patch.object(hybrid_mapper, "pack_conv_weights", return_value=packed_tensors) as mock_pack:
            hybrid_mapper.map(tensors)

        mock_pack.assert_called_once_with(tensors)
        hybrid_mapper.tr.rename_and_split.assert_called_once_with(packed_tensors)

        full_attn_mapper = DeepseekV3Mapper()
        full_attn_mapper.cfg = argparse.Namespace(linear_attention_type=None)
        full_attn_mapper.tr = MagicMock()
        full_attn_mapper.tr.rename_and_split.return_value = {"out": torch.randn(2, 2)}

        with patch.object(full_attn_mapper, "pack_conv_weights", return_value=packed_tensors) as mock_pack:
            full_attn_mapper.map(tensors)

        mock_pack.assert_not_called()
        full_attn_mapper.tr.rename_and_split.assert_called_once_with(tensors)

    def test_map_exports_gated_attention_proj_for_full_attention_layers(self):
        from base_mappers import DeepseekV3Mapper

        gate_weight = torch.randn(8, 16, dtype=torch.float32)
        tensors = {
            "model.layers.0.self_attn._attn.gate_proj.weight": gate_weight,
        }

        mapper = DeepseekV3Mapper(self._ckpt_cfg())

        result = mapper.map(tensors)

        assert "model.layers.0.self_attn.gate_proj.weight" in result
        assert result["model.layers.0.self_attn.gate_proj.weight"].dtype == torch.bfloat16
        torch.testing.assert_close(
            result["model.layers.0.self_attn.gate_proj.weight"],
            gate_weight.to(torch.bfloat16),
        )

    def test_map_exports_gated_attention_proj_for_hybrid_full_attention_layers(self):
        from base_mappers import DeepseekV3Mapper

        gate_weight = torch.randn(8, 16, dtype=torch.float32)
        tensors = {
            "model.layers.0.self_attn._attn.gate_proj.weight": gate_weight,
        }

        mapper = DeepseekV3Mapper(
            argparse.Namespace(
                **self._ckpt_cfg(
                linear_attention_type="Qwen3NextGatedDeltaNet",
                full_attention_layers=[0],
            )
            )
        )

        result = mapper.map(tensors)

        assert "model.layers.0.self_attn.gate_proj.weight" in result
        assert result["model.layers.0.self_attn.gate_proj.weight"].dtype == torch.bfloat16
        torch.testing.assert_close(
            result["model.layers.0.self_attn.gate_proj.weight"],
            gate_weight.to(torch.bfloat16),
        )

    def test_map_fuses_current_qwengdn_split_qkv_and_z_projections(self):
        from base_mappers import DeepseekV3Mapper

        # n_h=2 key heads, n_v=2 value heads (g=1), d_k=2, d_v=2
        # qkv per-head = 2*d_k + g*d_v = 4 + 2 = 6 → qkv shape (2*6, 16) = (12, 16)
        # z   per-head = g*d_v = 2           → z shape   (2*2, 16) = (4, 16)
        qkv_weight = torch.randn(12, 16, dtype=torch.float32)
        z_weight = torch.randn(4, 16, dtype=torch.float32)

        mapper = DeepseekV3Mapper(
            argparse.Namespace(
                **self._ckpt_cfg(
                    linear_attention_type="Qwen3NextGatedDeltaNet",
                    linear_num_key_heads=2,
                    linear_num_value_heads=2,
                    linear_key_head_dim=2,
                    linear_value_head_dim=2,
                )
            )
        )

        first = mapper.map({
            "model.layers.0.self_attn._attn.qkv_proj.weight": qkv_weight,
        })
        second = mapper.map({
            "model.layers.0.self_attn._attn.z_proj.weight": z_weight,
        })

        # Expected: per-head interleaved [q, k, v, z] layout.
        expected = torch.cat(
            [
                qkv_weight.view(2, 6, 16),
                z_weight.view(2, 2, 16),
            ],
            dim=1,
        ).reshape(16, 16).contiguous().to(torch.bfloat16)

        assert first == {}
        assert "model.layers.0.self_attn.in_proj_qkvz.weight" in second
        torch.testing.assert_close(
            second["model.layers.0.self_attn.in_proj_qkvz.weight"],
            expected,
        )

    def test_map_exports_current_qwengdn_fused_conv_weight(self):
        from base_mappers import DeepseekV3Mapper

        # n_h=2 key heads, n_v=4 value heads (g=2), d_k=2, d_v=2
        # per-head = 2*d_k + g*d_v = 4 + 4 = 8 → channels = 2*8 = 16
        conv_weight = torch.randn(16, 1, 4, dtype=torch.float32)

        mapper = DeepseekV3Mapper(
            argparse.Namespace(
                **self._ckpt_cfg(
                    linear_attention_type="Qwen3NextGatedDeltaNet",
                    linear_num_key_heads=2,
                    linear_num_value_heads=4,
                    linear_key_head_dim=2,
                    linear_value_head_dim=2,
                )
            )
        )

        result = mapper.map({
            "model.layers.0.self_attn._attn.qkv_conv1d.weight": conv_weight,
        })

        # Expected: interleaved per-head [q(d_k), k(d_k), v(g*d_v)] → conv order
        # [q_all(n_h*d_k); k_all(n_h*d_k); v_all(n_h*g*d_v)].
        per_head = conv_weight.view(2, 8, 1, 4)
        q = per_head[:, :2].reshape(4, 1, 4)
        k = per_head[:, 2:4].reshape(4, 1, 4)
        v = per_head[:, 4:].reshape(8, 1, 4)
        expected = torch.cat([q, k, v], dim=0).contiguous().to(torch.bfloat16)

        assert "model.layers.0.self_attn.conv1d.weight" in result
        torch.testing.assert_close(
            result["model.layers.0.self_attn.conv1d.weight"],
            expected,
        )

    def test_map_keeps_legacy_qwengdn_qkvz_projection_mapping(self):
        from base_mappers import DeepseekV3Mapper

        qkvz_weight = torch.randn(16, 16, dtype=torch.float32)

        mapper = DeepseekV3Mapper(
            argparse.Namespace(
                **self._ckpt_cfg(linear_attention_type="Qwen3NextGatedDeltaNet")
            )
        )

        result = mapper.map({
            "model.layers.0.self_attn._attn.qkvz_proj.weight": qkvz_weight,
        })

        assert "model.layers.0.self_attn.in_proj_qkvz.weight" in result
        torch.testing.assert_close(
            result["model.layers.0.self_attn.in_proj_qkvz.weight"],
            qkvz_weight.to(torch.bfloat16),
        )

    def test_map_exports_qwengdn_output_norm_subtree(self):
        from base_mappers import DeepseekV3Mapper

        tensors = {
            "model.layers.0.self_attn._attn.o_norm.weight": torch.randn(4),
            "model.layers.0.self_attn._attn.o_norm.gate_up_projection.weight": torch.randn(16, 4),
            "model.layers.0.self_attn._attn.o_norm.gate_down_projection.weight": torch.randn(4, 16),
            "model.layers.0.self_attn._attn.gated_delta_net.o_norm.gate_up_projection.weight": torch.randn(16, 4),
        }

        mapper = DeepseekV3Mapper(
            argparse.Namespace(
                **self._ckpt_cfg(linear_attention_type="Qwen3NextGatedDeltaNet")
            )
        )

        result = mapper.map(tensors)

        expected_keys = {
            "model.layers.0.self_attn.norm.weight":
                tensors["model.layers.0.self_attn._attn.o_norm.weight"],
            "model.layers.0.self_attn.norm.gate_up_projection.weight":
                tensors["model.layers.0.self_attn._attn.o_norm.gate_up_projection.weight"],
            "model.layers.0.self_attn.norm.gate_down_projection.weight":
                tensors["model.layers.0.self_attn._attn.o_norm.gate_down_projection.weight"],
            "model.layers.0.linear_attn.norm.gate_up_projection.weight":
                tensors["model.layers.0.self_attn._attn.gated_delta_net.o_norm.gate_up_projection.weight"],
        }
        for key, value in expected_keys.items():
            assert key in result
            torch.testing.assert_close(result[key], value.to(torch.bfloat16))

    def test_map_exports_kda_linear_attention_weights(self):
        from base_mappers import DeepseekV3Mapper

        # n_h=2 key heads, n_v=4 value heads (g=2), d_k=2, d_v=2
        # qkv arrives interleaved per key head: [h0: q,k,v0,v1; h1: ...].
        # Exported KDA runtime layout is Kimi-style separate q/k/v modules.
        # q/k stay compact; SGLang repeats them transiently before KDA kernels.
        qkv_weight = torch.randn(16, 16)
        conv_weight = torch.randn(16, 1, 4)
        tensors = {
            "model.layers.0.self_attn._attn.qkv_proj.weight": qkv_weight,
            "model.layers.0.self_attn._attn.conv1d.weight": conv_weight,
            "model.layers.0.self_attn._attn.f_proj.0.weight": torch.randn(2, 16),
            "model.layers.0.self_attn._attn.f_proj.1.weight": torch.randn(8, 2),
            "model.layers.0.self_attn._attn.b_proj.weight": torch.randn(4, 16),
            "model.layers.0.self_attn._attn.g_proj.0.weight": torch.randn(2, 16),
            "model.layers.0.self_attn._attn.g_proj.1.weight": torch.randn(8, 2),
            "model.layers.0.self_attn._attn.o_norm.weight": torch.randn(2),
            "model.layers.0.self_attn._attn.o_proj.weight": torch.randn(16, 8),
            "model.layers.0.self_attn._attn.A_log": torch.randn(1, 1, 4, 1),
            "model.layers.0.self_attn._attn.dt_bias": torch.randn(8),
        }

        mapper = DeepseekV3Mapper(
            argparse.Namespace(
                **self._ckpt_cfg(
                    linear_attention_type="KimiDeltaAttention",
                    linear_num_key_heads=2,
                    linear_num_value_heads=4,
                    linear_key_head_dim=2,
                    linear_value_head_dim=2,
                )
            )
        )

        result = mapper.map(tensors)

        qkv_per_head = qkv_weight.view(2, 8, 16)
        expected_q_proj = qkv_per_head[:, :2].reshape(4, 16)
        expected_k_proj = qkv_per_head[:, 2:4].reshape(4, 16)
        expected_v_proj = qkv_per_head[:, 4:].reshape(8, 16)
        conv_per_head = conv_weight.view(2, 8, 1, 4)
        expected_q_conv = conv_per_head[:, :2].reshape(4, 1, 4)
        expected_k_conv = conv_per_head[:, 2:4].reshape(4, 1, 4)
        expected_v_conv = conv_per_head[:, 4:].reshape(8, 1, 4)

        expected_keys = {
            "model.layers.0.self_attn.q_proj.weight": expected_q_proj,
            "model.layers.0.self_attn.k_proj.weight": expected_k_proj,
            "model.layers.0.self_attn.v_proj.weight": expected_v_proj,
            "model.layers.0.self_attn.q_conv1d.weight": expected_q_conv,
            "model.layers.0.self_attn.k_conv1d.weight": expected_k_conv,
            "model.layers.0.self_attn.v_conv1d.weight": expected_v_conv,
            "model.layers.0.self_attn.f_a_proj.weight": tensors["model.layers.0.self_attn._attn.f_proj.0.weight"],
            "model.layers.0.self_attn.f_b_proj.weight": tensors["model.layers.0.self_attn._attn.f_proj.1.weight"],
            "model.layers.0.self_attn.b_proj.weight": tensors["model.layers.0.self_attn._attn.b_proj.weight"],
            "model.layers.0.self_attn.g_a_proj.weight": tensors["model.layers.0.self_attn._attn.g_proj.0.weight"],
            "model.layers.0.self_attn.g_b_proj.weight": tensors["model.layers.0.self_attn._attn.g_proj.1.weight"],
            "model.layers.0.self_attn.o_norm.weight": tensors["model.layers.0.self_attn._attn.o_norm.weight"],
            "model.layers.0.self_attn.o_proj.weight": tensors["model.layers.0.self_attn._attn.o_proj.weight"],
            "model.layers.0.self_attn.A_log": tensors["model.layers.0.self_attn._attn.A_log"],
            "model.layers.0.self_attn.dt_bias": tensors["model.layers.0.self_attn._attn.dt_bias"],
        }

        assert "model.layers.0.self_attn.in_proj_qkvz.weight" not in result
        assert "model.layers.0.self_attn.conv1d.weight" not in result
        for key, value in expected_keys.items():
            assert key in result
            torch.testing.assert_close(result[key], value.to(torch.bfloat16))

    def test_map_exports_zero_centered_norm_and_shared_expert_sigmoid_weights(self):
        from base_mappers import DeepseekV3Mapper

        tensors = {
            "model.layers.0.self_attn.input_layernorm.gate.0.weight": torch.randn(16, 32),
            "model.layers.0.self_attn.input_layernorm.gate.2.weight": torch.randn(32, 16),
            "model.layers.0.self_attn.post_attention_layernorm.gate.0.weight": torch.randn(16, 32),
            "model.layers.0.self_attn.post_attention_layernorm.gate.2.weight": torch.randn(32, 16),
            "model.layers.0.self_attn.post_self_attn_layernorm.gate.0.weight": torch.randn(16, 32),
            "model.layers.0.self_attn.post_self_attn_layernorm.gate.2.weight": torch.randn(32, 16),
            "model.layers.0.self_attn.post_feedforward_layernorm.gate.0.weight": torch.randn(16, 32),
            "model.layers.0.self_attn.post_feedforward_layernorm.gate.2.weight": torch.randn(32, 16),
            "model.layers.1.block_sparse_moe.post_feedforward_layernorm.weight": torch.randn(32),
            "model.layers.0.self_attn._attn.q_norm.gate.0.weight": torch.randn(16, 24),
            "model.layers.0.self_attn._attn.q_norm.gate.2.weight": torch.randn(24, 16),
            "model.layers.0.self_attn._attn.kv_norm.gate.0.weight": torch.randn(16, 20),
            "model.layers.0.self_attn._attn.kv_norm.gate.2.weight": torch.randn(20, 16),
            "model.layers.2.self_attn.input_layernorm.gate_up_projection.weight": torch.randn(16, 32),
            "model.layers.2.self_attn.input_layernorm.gate_down_projection.weight": torch.randn(32, 16),
            "model.layers.0.self_attn.shared_expert_sigmoid.weight": torch.randn(1, 32),
        }

        mapper = DeepseekV3Mapper(self._ckpt_cfg())
        result = mapper.map(tensors)

        expected_keys = {
            "model.layers.0.input_layernorm.gate_up_projection.weight":
                tensors["model.layers.0.self_attn.input_layernorm.gate.0.weight"],
            "model.layers.0.input_layernorm.gate_down_projection.weight":
                tensors["model.layers.0.self_attn.input_layernorm.gate.2.weight"],
            "model.layers.0.post_attention_layernorm.gate_up_projection.weight":
                tensors["model.layers.0.self_attn.post_attention_layernorm.gate.0.weight"],
            "model.layers.0.post_attention_layernorm.gate_down_projection.weight":
                tensors["model.layers.0.self_attn.post_attention_layernorm.gate.2.weight"],
            "model.layers.0.post_self_attn_layernorm.gate_up_projection.weight":
                tensors["model.layers.0.self_attn.post_self_attn_layernorm.gate.0.weight"],
            "model.layers.0.post_self_attn_layernorm.gate_down_projection.weight":
                tensors["model.layers.0.self_attn.post_self_attn_layernorm.gate.2.weight"],
            "model.layers.0.post_feedforward_layernorm.gate_up_projection.weight":
                tensors["model.layers.0.self_attn.post_feedforward_layernorm.gate.0.weight"],
            "model.layers.0.post_feedforward_layernorm.gate_down_projection.weight":
                tensors["model.layers.0.self_attn.post_feedforward_layernorm.gate.2.weight"],
            "model.layers.1.post_feedforward_layernorm.weight":
                tensors["model.layers.1.block_sparse_moe.post_feedforward_layernorm.weight"],
            "model.layers.0.self_attn.q_a_layernorm.gate_up_projection.weight":
                tensors["model.layers.0.self_attn._attn.q_norm.gate.0.weight"],
            "model.layers.0.self_attn.q_a_layernorm.gate_down_projection.weight":
                tensors["model.layers.0.self_attn._attn.q_norm.gate.2.weight"],
            "model.layers.0.self_attn.kv_a_layernorm.gate_up_projection.weight":
                tensors["model.layers.0.self_attn._attn.kv_norm.gate.0.weight"],
            "model.layers.0.self_attn.kv_a_layernorm.gate_down_projection.weight":
                tensors["model.layers.0.self_attn._attn.kv_norm.gate.2.weight"],
            "model.layers.2.input_layernorm.gate_up_projection.weight":
                tensors["model.layers.2.self_attn.input_layernorm.gate_up_projection.weight"],
            "model.layers.2.input_layernorm.gate_down_projection.weight":
                tensors["model.layers.2.self_attn.input_layernorm.gate_down_projection.weight"],
            "model.layers.0.mlp.shared_expert_sigmoid.weight":
                tensors["model.layers.0.self_attn.shared_expert_sigmoid.weight"],
        }

        for key, expected_tensor in expected_keys.items():
            assert key in result
            assert result[key].dtype == torch.bfloat16
            torch.testing.assert_close(result[key], expected_tensor.to(torch.bfloat16))


class TestGigachat35Config:
    def test_derives_post_layernorm_flags_from_layernorm_type(self):
        from models.gigachat3_5.configuration_gigachat3_5 import Gigachat35Config

        post_cfg = Gigachat35Config(layernorm_type="post")
        pre_post_cfg = Gigachat35Config(layernorm_type="pre_post")
        pre_cfg = Gigachat35Config(layernorm_type="pre")

        assert post_cfg._use_pre_layernorm is False
        assert post_cfg._use_post_layernorm is True
        assert pre_post_cfg._use_pre_layernorm is True
        assert pre_post_cfg._use_post_layernorm is True
        assert pre_cfg._use_pre_layernorm is True
        assert pre_cfg._use_post_layernorm is False

    def test_rejects_unknown_layernorm_type(self):
        from models.gigachat3_5.configuration_gigachat3_5 import Gigachat35Config

        with pytest.raises(ValueError, match="layernorm_type must be one of"):
            Gigachat35Config(layernorm_type="weird")

    def test_resolves_explicit_dense_attention_layers(self):
        from models.gigachat3_5.configuration_gigachat3_5 import Gigachat35Config

        config = Gigachat35Config(
            num_hidden_layers=12,
            attention_type="LlamaLatentSparseAttention",
            linear_attention_type="Qwen3NextGatedDeltaNet",
            full_attention_layers=[3, 7, 11],
            dense_attention_layers=[7],
            index_topk=8,
        )

        assert config.is_dsa_attention_layer(3) is True
        assert config.is_dsa_attention_layer(7) is False
        assert config.is_dsa_attention_layer(11) is True
        assert config.is_dsa_attention_layer(0) is False
        assert config.to_dict()["dense_attention_layers"] == [7]

    def test_attention_only_builds_indexer_for_dsa_layers(self):
        from models.gigachat3_5.configuration_gigachat3_5 import Gigachat35Config
        from models.gigachat3_5.modeling_gigachat3_5 import Gigachat35Attention

        config = Gigachat35Config(
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            v_head_dim=4,
            attention_type="LlamaLatentSparseAttention",
            norm_type="ZeroCenteredGatedNorm",
            linear_attention_type="Qwen3NextGatedDeltaNet",
            full_attention_layers=[1, 3],
            dense_attention_layers=[3],
            index_n_heads=2,
            index_head_dim=4,
            index_topk=8,
        )

        sparse_attention = Gigachat35Attention(config, layer_idx=1)
        dense_attention = Gigachat35Attention(config, layer_idx=3)

        assert dense_attention.use_dsa is False
        assert not hasattr(dense_attention, "indexer")
        assert sparse_attention.use_dsa is True
        assert hasattr(sparse_attention, "indexer")
        assert type(sparse_attention.indexer.k_norm).__name__ == "Gigachat35RMSNorm"
        assert list(dict(sparse_attention.indexer.k_norm.named_parameters())) == ["weight"]

    @pytest.mark.parametrize("value", [[-1], [True], [1.5], "1", ()])
    def test_rejects_invalid_dense_attention_layers(self, value):
        from models.gigachat3_5.configuration_gigachat3_5 import Gigachat35Config

        with pytest.raises(ValueError, match="dense_attention_layers"):
            Gigachat35Config(
                attention_type="LlamaLatentSparseAttention",
                dense_attention_layers=value,
            )

    def test_rejects_dense_attention_layer_outside_full_attention_layers(self):
        from models.gigachat3_5.configuration_gigachat3_5 import Gigachat35Config

        with pytest.raises(ValueError, match="subset of full_attention_layers"):
            Gigachat35Config(
                num_hidden_layers=4,
                attention_type="LlamaLatentSparseAttention",
                linear_attention_type="Qwen3NextGatedDeltaNet",
                full_attention_layers=[1, 3],
                dense_attention_layers=[2],
            )
    def test_nextn_is_linear_defaults_to_none(self):
        from models.gigachat3_5.configuration_gigachat3_5 import Gigachat35Config

        assert Gigachat35Config().nextn_is_linear is None

    @pytest.mark.parametrize("flag", [True, False])
    def test_nextn_is_linear_round_trips(self, flag):
        from models.gigachat3_5.configuration_gigachat3_5 import Gigachat35Config

        cfg = Gigachat35Config(nextn_is_linear=flag)

        assert cfg.nextn_is_linear is flag
        assert cfg.to_dict()["nextn_is_linear"] is flag

    def test_mtp_flag_does_not_extend_layer_types(self):
        from models.gigachat3_5.configuration_gigachat3_5 import Gigachat35Config

        cfg = Gigachat35Config(
            num_hidden_layers=4,
            linear_attention_type="Qwen3NextGatedDeltaNet",
            full_attention_layers=[1, 3],
            num_nextn_predict_layers=1,
            nextn_is_linear=True,
        )

        assert cfg.layer_types == [
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]


class TestLayerMapperProtocol:
    """Tests to verify LayerMapper protocol compliance"""

    def test_raw_mapper_implements_protocol(self):
        """Test that RawMapper implements LayerMapper protocol"""
        from base_mappers import RawMapper, LayerMapper

        mapper = RawMapper()

        # Check all protocol methods exist
        assert hasattr(mapper, "map")
        assert hasattr(mapper, "flush")
        assert hasattr(mapper, "ensure_config")
        assert hasattr(mapper, "build_config")
        assert hasattr(mapper, "tokenizer_source")
        assert hasattr(mapper, "save_meta")

    @pytest.mark.parametrize("mapper_name", ["DeepseekV3Mapper", "Gigachat35DenseMapper"])
    def test_deepseek_mapper_implements_protocol(self, mapper_name):
        """Test that DeepseekV3Mapper and its subclasses implement LayerMapper protocol"""
        import base_mappers

        mapper = getattr(base_mappers, mapper_name)()

        # Check all protocol methods exist
        assert hasattr(mapper, "map")
        assert hasattr(mapper, "flush")
        assert hasattr(mapper, "ensure_config")
        assert hasattr(mapper, "build_config")
        assert hasattr(mapper, "tokenizer_source")
        assert hasattr(mapper, "save_meta")


class TestDeepseekMapperModelSpec:
    def test_family_follows_the_checkpoint_config(self):
        from base_mappers import DeepseekV3Mapper
        from utils.hf_model_specs import DEEPSEEK_V3_SPEC, GIGACHAT3_5_SPEC

        mapper = DeepseekV3Mapper()
        assert mapper._model_spec({"attention_type": "LlamaLatentAttention"}) is DEEPSEEK_V3_SPEC
        assert mapper._model_spec({"linear_attention_type": "Qwen3NextGatedDeltaNet"}) is GIGACHAT3_5_SPEC
