import pytest
import torch
import math
from pathlib import Path
from mappers import map_merge_lora, map_fsdp_lora_to_hf, map_fsdp_lora_atapter_to_hf
from unittest.mock import Mock, patch


@pytest.mark.unit
class TestMergeLora:
    """Тесты для map_merge_lora"""

    def test_merge_simple_lora_layer(self, mock_args):
        """Тест базового слияния LoRA адаптера с base layer"""
        checkpoint = self._create_lora_checkpoint()
        mock_args.skip_optimizer = True

        result = map_merge_lora(checkpoint, args=mock_args)

        model = result["state"]["model"]

        # После слияния должны остаться только merged слои
        assert "model.layers.0.self_attn.q_proj.weight" in model
        assert "model.layers.0.self_attn.q_proj.base_layer.weight" not in model
        assert "model.layers.0.self_attn.q_proj.lora_A.default.weight" not in model
        assert "model.layers.0.self_attn.q_proj.lora_B.default.weight" not in model

    def test_lora_scaling_applied(self, mock_args):
        """Тест применения scaling factor при слиянии"""
        checkpoint = self._create_lora_checkpoint(
            lora_alpha=16, lora_r=8, use_rslora=False
        )
        mock_args.skip_optimizer = True

        # Запоминаем оригинальные веса
        base_weight = checkpoint["state"]["model"][
            "model.layers.0.self_attn.q_proj.base_layer.weight"
        ].clone()
        lora_A = checkpoint["state"]["model"][
            "model.layers.0.self_attn.q_proj.lora_A.default.weight"
        ].clone()
        lora_B = checkpoint["state"]["model"][
            "model.layers.0.self_attn.q_proj.lora_B.default.weight"
        ].clone()

        result = map_merge_lora(checkpoint, args=mock_args)

        merged = result["state"]["model"]["model.layers.0.self_attn.q_proj.weight"]

        # scaling = alpha / r = 16 / 8 = 2
        expected_scaling = 16.0 / 8.0
        expected = (
            base_weight
            + expected_scaling * (lora_B.float() @ lora_A.float()).bfloat16()
        )

        assert torch.allclose(merged, expected, atol=1e-5)

    def test_rslora_scaling_applied(self, mock_args):
        """Тест применения rsLoRA scaling"""
        checkpoint = self._create_lora_checkpoint(
            lora_alpha=16, lora_r=8, use_rslora=True
        )
        mock_args.skip_optimizer = True

        base_weight = checkpoint["state"]["model"][
            "model.layers.0.self_attn.q_proj.base_layer.weight"
        ].clone()
        lora_A = checkpoint["state"]["model"][
            "model.layers.0.self_attn.q_proj.lora_A.default.weight"
        ].clone()
        lora_B = checkpoint["state"]["model"][
            "model.layers.0.self_attn.q_proj.lora_B.default.weight"
        ].clone()

        result = map_merge_lora(checkpoint, args=mock_args)

        merged = result["state"]["model"]["model.layers.0.self_attn.q_proj.weight"]

        # rsLoRA: scaling = alpha / sqrt(r) = 16 / sqrt(8)
        expected_scaling = 16.0 / math.sqrt(8)
        expected = (
            base_weight
            + expected_scaling * (lora_B.float() @ lora_A.float()).bfloat16()
        )

        assert torch.allclose(merged, expected, atol=1e-5)

    def test_merge_qkv_packed_lora(self, mock_args):
        """Тест слияния LoRA для packed qkv_proj"""
        checkpoint = self._create_lora_checkpoint_with_qkv()
        mock_args.skip_optimizer = True

        result = map_merge_lora(checkpoint, args=mock_args)

        model = result["state"]["model"]

        # qkv_proj должен быть слит
        assert "model.layers.0.self_attn.qkv_proj.weight" in model

        # Все компоненты LoRA должны быть удалены
        assert "model.layers.0.self_attn.qkv_proj.base_layer.weight" not in model
        # NOTE: For packed qkv, the LoRA keys are under qkv_proj.q_proj.lora_A etc
        assert (
            "model.layers.0.self_attn.qkv_proj.q_proj.lora_A.default.weight"
            not in model
        )
        assert (
            "model.layers.0.self_attn.qkv_proj.k_proj.lora_A.default.weight"
            not in model
        )
        assert (
            "model.layers.0.self_attn.qkv_proj.v_proj.lora_A.default.weight"
            not in model
        )

    def test_merge_multiple_layers(self, mock_args):
        """Тест слияния LoRA в нескольких слоях"""
        checkpoint = self._create_lora_checkpoint_multilayer()
        mock_args.skip_optimizer = True

        result = map_merge_lora(checkpoint, args=mock_args)

        model = result["state"]["model"]

        # Проверяем все слои
        for layer_idx in range(2):
            assert f"model.layers.{layer_idx}.self_attn.q_proj.weight" in model
            assert f"model.layers.{layer_idx}.self_attn.k_proj.weight" in model
            assert f"model.layers.{layer_idx}.mlp.gate_proj.weight" in model

    def test_non_lora_layers_unchanged(self, mock_args):
        """Тест что слои без LoRA остаются неизменными"""
        checkpoint = self._create_lora_checkpoint()
        mock_args.skip_optimizer = True

        # Добавляем слой без LoRA
        checkpoint["state"]["model"]["model.norm.weight"] = torch.randn(512)
        original_norm = checkpoint["state"]["model"]["model.norm.weight"].clone()

        result = map_merge_lora(checkpoint, args=mock_args)

        # norm должен остаться неизменным
        assert torch.equal(result["state"]["model"]["model.norm.weight"], original_norm)

    def test_lora_config_in_checkpoint(self, mock_args):
        """Тест что lora_config присутствует в чекпоинте"""
        checkpoint = self._create_lora_checkpoint()
        mock_args.skip_optimizer = True

        config = checkpoint["state"]["integrations"]["huggingface"]["model"]["config"][
            "content"
        ]
        assert "lora_config" in config
        assert config["lora_config"]["r"] == 8
        assert config["lora_config"]["lora_alpha"] == 16

    def _create_lora_checkpoint(self, lora_alpha=16, lora_r=8, use_rslora=False):
        """Создает чекпоинт с LoRA адаптерами"""
        hidden_size = 512

        checkpoint = {
            "state": {
                "model": {
                    # Base layer
                    "model.layers.0.self_attn.q_proj.base_layer.weight": torch.randn(
                        hidden_size, hidden_size
                    ),
                    # LoRA adapters
                    "model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(
                        lora_r, hidden_size
                    ),
                    "model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(
                        hidden_size, lora_r
                    ),
                    # Другие слои
                    "model.layers.0.self_attn.k_proj.base_layer.weight": torch.randn(
                        hidden_size, hidden_size
                    ),
                    "model.layers.0.self_attn.k_proj.lora_A.default.weight": torch.randn(
                        lora_r, hidden_size
                    ),
                    "model.layers.0.self_attn.k_proj.lora_B.default.weight": torch.randn(
                        hidden_size, lora_r
                    ),
                },
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "model_type": "gigar",
                                    "hidden_size": hidden_size,
                                    "lora_config": {
                                        "r": lora_r,
                                        "lora_alpha": lora_alpha,
                                        "use_rslora": use_rslora,
                                        "target_modules": [
                                            "q_proj",
                                            "k_proj",
                                            "v_proj",
                                        ],
                                    },
                                }
                            }
                        }
                    }
                },
            }
        }
        return checkpoint

    def _create_lora_checkpoint_with_qkv(self):
        """Создает чекпоинт с packed qkv и LoRA

        IMPORTANT: For packed QKV with LoRA, the keys must be:
        - base_layer: model.layers.0.self_attn.qkv_proj.base_layer.weight
        - LoRA A/B: model.layers.0.self_attn.qkv_proj.{q,k,v}_proj.lora_{A,B}.default.weight

        This is because map_merge_lora does:
        q_A_key = module.replace('base_layer', 'q_proj.lora_A.default')
        """
        hidden_size = 512
        lora_r = 8

        checkpoint = {
            "state": {
                "model": {
                    # Packed base layer
                    "model.layers.0.self_attn.qkv_proj.base_layer.weight": torch.randn(
                        hidden_size * 3, hidden_size
                    ),
                    # LoRA adapters for each component - NOTE the qkv_proj prefix!
                    "model.layers.0.self_attn.qkv_proj.q_proj.lora_A.default.weight": torch.randn(
                        lora_r, hidden_size
                    ),
                    "model.layers.0.self_attn.qkv_proj.q_proj.lora_B.default.weight": torch.randn(
                        hidden_size, lora_r
                    ),
                    "model.layers.0.self_attn.qkv_proj.k_proj.lora_A.default.weight": torch.randn(
                        lora_r, hidden_size
                    ),
                    "model.layers.0.self_attn.qkv_proj.k_proj.lora_B.default.weight": torch.randn(
                        hidden_size, lora_r
                    ),
                    "model.layers.0.self_attn.qkv_proj.v_proj.lora_A.default.weight": torch.randn(
                        lora_r, hidden_size
                    ),
                    "model.layers.0.self_attn.qkv_proj.v_proj.lora_B.default.weight": torch.randn(
                        hidden_size, lora_r
                    ),
                },
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "model_type": "gigar",
                                    "hidden_size": hidden_size,
                                    "lora_config": {
                                        "r": lora_r,
                                        "lora_alpha": 16,
                                        "use_rslora": False,
                                        "target_modules": ["qkv_proj"],
                                    },
                                }
                            }
                        }
                    }
                },
            }
        }
        return checkpoint

    def _create_lora_checkpoint_multilayer(self):
        """Создает чекпоинт с LoRA в нескольких слоях"""
        hidden_size = 512
        lora_r = 8
        num_layers = 2

        model_dict = {}

        for layer_idx in range(num_layers):
            prefix = f"model.layers.{layer_idx}"

            # Attention LoRA
            for proj in ["q_proj", "k_proj", "v_proj"]:
                model_dict[f"{prefix}.self_attn.{proj}.base_layer.weight"] = (
                    torch.randn(hidden_size, hidden_size)
                )
                model_dict[f"{prefix}.self_attn.{proj}.lora_A.default.weight"] = (
                    torch.randn(lora_r, hidden_size)
                )
                model_dict[f"{prefix}.self_attn.{proj}.lora_B.default.weight"] = (
                    torch.randn(hidden_size, lora_r)
                )

            # MLP LoRA
            model_dict[f"{prefix}.mlp.gate_proj.base_layer.weight"] = torch.randn(
                hidden_size * 4, hidden_size
            )
            model_dict[f"{prefix}.mlp.gate_proj.lora_A.default.weight"] = torch.randn(
                lora_r, hidden_size
            )
            model_dict[f"{prefix}.mlp.gate_proj.lora_B.default.weight"] = torch.randn(
                hidden_size * 4, lora_r
            )

        checkpoint = {
            "state": {
                "model": model_dict,
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "model_type": "gigar",
                                    "hidden_size": hidden_size,
                                    "num_hidden_layers": num_layers,
                                    "lora_config": {
                                        "r": lora_r,
                                        "lora_alpha": 16,
                                        "use_rslora": False,
                                        "target_modules": [
                                            "q_proj",
                                            "k_proj",
                                            "v_proj",
                                            "gate_proj",
                                        ],
                                    },
                                }
                            }
                        }
                    }
                },
            }
        }
        return checkpoint


@pytest.mark.unit
class TestFsdpLoraToHf:
    """Tests for map_fsdp_lora_to_hf"""

    def test_basic_lora_merge_and_convert(self, temp_dir, mock_args):
        """Test basic LoRA merge and HF conversion"""
        checkpoint = self._create_lora_checkpoint()
        mock_args.output_checkpoint_path = str(temp_dir / "output")
        mock_args.tokenizer_path = None
        mock_args.input_tp_size = 1

        with patch("mappers.map_fsdp_to_hf") as mock_convert:
            map_fsdp_lora_to_hf(checkpoint, args=mock_args)
            mock_convert.assert_called_once()

    def test_lora_merged_before_hf_conversion(self, temp_dir, mock_args):
        """Test that LoRA is merged before HF conversion"""
        checkpoint = self._create_lora_checkpoint()
        mock_args.output_checkpoint_path = str(temp_dir / "output")
        mock_args.input_tp_size = 1

        with patch("mappers.map_fsdp_to_hf") as mock_convert:
            map_fsdp_lora_to_hf(checkpoint, args=mock_args)

            # Check that merged checkpoint was passed
            call_args = mock_convert.call_args[0][0]
            model = call_args["state"]["model"]

            # After merge, should have regular weights not base_layer
            assert any(
                "q_proj.weight" in k and "base_layer" not in k for k in model.keys()
            )

    def test_tp_size_handled(self, temp_dir, mock_args):
        """Test handling of TP>1 checkpoints"""
        checkpoint = self._create_lora_checkpoint()
        mock_args.output_checkpoint_path = str(temp_dir / "output")
        mock_args.input_tp_size = 2

        with patch("mappers.map_fsdp_to_hf") as mock_convert:
            # Pass list of checkpoints for TP=2
            map_fsdp_lora_to_hf([checkpoint, checkpoint], args=mock_args)
            mock_convert.assert_called_once()

    def _create_lora_checkpoint(self):
        hidden_size = 512
        return {
            "state": {
                "model": {
                    "model.embed_tokens.weight": torch.randn(1000, hidden_size),
                    "model.layers.0.self_attn.q_proj.base_layer.weight": torch.randn(
                        hidden_size, hidden_size
                    ),
                    "model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(
                        8, hidden_size
                    ),
                    "model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(
                        hidden_size, 8
                    ),
                    "model.lm_head.weight": torch.randn(1000, hidden_size),
                },
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "model_type": "gigar",
                                    "hidden_size": hidden_size,
                                    "num_attention_heads": 8,
                                    "num_key_value_heads": 8,
                                    "tp_size": 1,
                                    "lora_config": {
                                        "r": 8,
                                        "lora_alpha": 16,
                                        "target_modules": ["q_proj"],
                                    },
                                    "_name_or_path": "test_model",
                                }
                            }
                        }
                    }
                },
            }
        }


@pytest.mark.unit
class TestFsdpLoraAdapterToHf:
    """Тесты для map_fsdp_lora_atapter_to_hf - сохранение только адаптера"""

    def test_save_adapter_only(self, temp_dir, mock_args):
        """Тест сохранения только LoRA адаптера без base модели"""
        checkpoint = self._create_lora_checkpoint()
        mock_args.output_checkpoint_path = str(temp_dir / "adapter_output")
        mock_args.tokenizer_path = None
        mock_args.save_full_model = False
        mock_args.input_tp_size = 1

        with patch("mappers.get_peft_model") as mock_peft, patch(
            "mappers.AutoModelForCausalLM"
        ), patch("mappers.AutoTokenizer"), patch("mappers.AutoConfig"):
            mock_model = Mock()
            mock_model.save_pretrained = Mock()
            mock_peft.return_value = mock_model

            map_fsdp_lora_atapter_to_hf(checkpoint, args=mock_args)

            # Проверяем что save_pretrained был вызван
            mock_model.save_pretrained.assert_called_once()

    def test_save_full_merged_model(self, temp_dir, mock_args):
        """Тест сохранения полной модели с слитыми весами"""
        checkpoint = self._create_lora_checkpoint()
        mock_args.output_checkpoint_path = str(temp_dir / "full_output")
        mock_args.tokenizer_path = None
        mock_args.save_full_model = True
        mock_args.input_tp_size = 1

        with patch("mappers.get_peft_model") as mock_peft, patch(
            "mappers.AutoModelForCausalLM"
        ), patch("mappers.AutoTokenizer"), patch("mappers.AutoConfig"):
            mock_model = Mock()
            mock_merged = Mock()
            mock_model.merge_and_unload.return_value = mock_merged
            mock_peft.return_value = mock_model

            map_fsdp_lora_atapter_to_hf(checkpoint, args=mock_args)

            # Проверяем что merge_and_unload был вызван
            mock_model.merge_and_unload.assert_called_once()
            mock_merged.save_pretrained.assert_called_once()

    def test_qkv_unpacking_for_adapter(self, temp_dir, mock_args):
        """Тест распаковки qkv_proj в q/k/v для PEFT"""
        checkpoint = self._create_lora_checkpoint_with_qkv()
        mock_args.output_checkpoint_path = str(temp_dir / "adapter_output")
        mock_args.save_full_model = False
        mock_args.input_tp_size = 1

        with patch("mappers.get_peft_model") as mock_peft, patch(
            "mappers.AutoModelForCausalLM"
        ), patch("mappers.AutoTokenizer"), patch("mappers.AutoConfig"):
            mock_model = Mock()
            mock_peft.return_value = mock_model

            map_fsdp_lora_atapter_to_hf(checkpoint, args=mock_args)

            # Проверяем что модель была создана
            mock_peft.assert_called_once()

    def test_target_modules_converted(self, temp_dir, mock_args):
        """Тест конвертации target_modules (qkv_proj -> q,k,v)"""
        checkpoint = self._create_lora_checkpoint()
        checkpoint["state"]["integrations"]["huggingface"]["model"]["config"][
            "content"
        ]["lora_config"]["target_modules"] = ["qkv_proj"]

        mock_args.output_checkpoint_path = str(temp_dir / "adapter_output")
        mock_args.save_full_model = False
        mock_args.input_tp_size = 1

        with patch("mappers.get_peft_model") as mock_peft, patch(
            "mappers.AutoModelForCausalLM"
        ), patch("mappers.AutoTokenizer"), patch("mappers.AutoConfig"), patch(
            "mappers.LoraConfig"
        ) as mock_lora_config:
            map_fsdp_lora_atapter_to_hf(checkpoint, args=mock_args)

            # Проверяем что LoraConfig был создан с правильными target_modules
            call_args = mock_lora_config.call_args
            target_modules = call_args[1]["target_modules"]
            assert "q_proj" in target_modules
            assert "k_proj" in target_modules
            assert "v_proj" in target_modules
            assert "qkv_proj" not in target_modules

    def test_attention_type_lora_handling(self, temp_dir, mock_args):
        """Тест обработки LlamaPackedAttentionLoRA"""
        checkpoint = self._create_lora_checkpoint()
        checkpoint["state"]["integrations"]["huggingface"]["model"]["config"][
            "content"
        ]["attention_type"] = "LlamaPackedAttentionLoRA"

        mock_args.output_checkpoint_path = str(temp_dir / "adapter_output")
        mock_args.save_full_model = False
        mock_args.input_tp_size = 1

        with patch("mappers.get_peft_model") as mock_peft, patch(
            "mappers.AutoModelForCausalLM"
        ), patch("mappers.AutoTokenizer"), patch("mappers.AutoConfig"):
            mock_model = Mock()
            mock_peft.return_value = mock_model

            # Should not raise
            map_fsdp_lora_atapter_to_hf(checkpoint, args=mock_args)

    def test_model_type_conversion(self, temp_dir, mock_args):
        """Тест конвертации model_type для HF совместимости"""
        checkpoint = self._create_lora_checkpoint()
        mock_args.output_checkpoint_path = str(temp_dir / "adapter_output")
        mock_args.save_full_model = False
        mock_args.input_tp_size = 1

        with patch("mappers.get_peft_model") as mock_peft, patch(
            "mappers.AutoModelForCausalLM"
        ), patch("mappers.AutoTokenizer"), patch("mappers.AutoConfig") as mock_config:
            mock_model = Mock()
            mock_peft.return_value = mock_model

            map_fsdp_lora_atapter_to_hf(checkpoint, args=mock_args)

            # AutoConfig should be called
            mock_config.from_pretrained.assert_called()

    def _create_lora_checkpoint(self):
        hidden_size = 512
        return {
            "state": {
                "model": {
                    "model.embed_tokens.weight": torch.randn(1000, hidden_size),
                    "model.layers.0.self_attn.q_proj.base_layer.weight": torch.randn(
                        hidden_size, hidden_size
                    ),
                    "model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(
                        8, hidden_size
                    ),
                    "model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(
                        hidden_size, 8
                    ),
                    "model.lm_head.weight": torch.randn(1000, hidden_size),
                },
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "model_type": "gigar",
                                    "hidden_size": hidden_size,
                                    "num_attention_heads": 8,
                                    "num_key_value_heads": 8,
                                    "tp_size": 1,
                                    "attention_type": "LlamaPackedAttention",
                                    "lora_config": {
                                        "r": 8,
                                        "lora_alpha": 16,
                                        "target_modules": ["q_proj"],
                                    },
                                    "_name_or_path": "test_model",
                                }
                            }
                        }
                    }
                },
            }
        }

    def _create_lora_checkpoint_with_qkv(self):
        hidden_size = 512
        return {
            "state": {
                "model": {
                    "model.embed_tokens.weight": torch.randn(1000, hidden_size),
                    "model.layers.0.self_attn.qkv_proj.base_layer.weight": torch.randn(
                        hidden_size * 3, hidden_size
                    ),
                    "model.layers.0.self_attn.qkv_proj.lora_A.default.weight": torch.randn(
                        8, hidden_size
                    ),
                    "model.layers.0.self_attn.qkv_proj.lora_B.default.weight": torch.randn(
                        hidden_size * 3, 8
                    ),
                    "model.lm_head.weight": torch.randn(1000, hidden_size),
                },
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "model_type": "gigar",
                                    "hidden_size": hidden_size,
                                    "num_attention_heads": 8,
                                    "num_key_value_heads": 8,
                                    "tp_size": 1,
                                    "attention_type": "LlamaPackedAttention",
                                    "lora_config": {
                                        "r": 8,
                                        "lora_alpha": 16,
                                        "target_modules": ["qkv_proj"],
                                    },
                                    "_name_or_path": "test_model",
                                }
                            }
                        }
                    }
                },
            }
        }
