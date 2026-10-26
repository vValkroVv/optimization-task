import pytest
import torch
from unittest.mock import patch, MagicMock, Mock
from collections import OrderedDict
import argparse
import copy


class TestGetEmbeddingType:
    """Tests for embedding type detection logic"""

    def test_embedding_parallel_detection(self):
        """Test detection of EmbeddingParallelEmbedding pattern"""
        # hidden_size = 64, tp_size = 2, so shard_size should be 32
        embedding_shard = torch.randn(1000, 32)  # [vocab_size, hidden/tp]
        model_config = {"hidden_size": 64}
        input_tp_size = 2

        # Check the logic that would be used
        sharded_embedding_size = embedding_shard.size(1)
        expected_hidden_size = model_config["hidden_size"]

        if expected_hidden_size == sharded_embedding_size * input_tp_size:
            result = "EmbeddingParallelEmbedding"
        else:
            result = "VocabParallelEmbedding"

        assert result == "EmbeddingParallelEmbedding"

    def test_vocab_parallel_detection(self):
        """Test detection of VocabParallelEmbedding pattern"""
        # When shard size doesn't match hidden_size / tp_size
        embedding_shard = torch.randn(500, 64)  # vocab sharded, not embedding
        model_config = {"hidden_size": 64}
        input_tp_size = 2

        sharded_embedding_size = embedding_shard.size(1)
        expected_hidden_size = model_config["hidden_size"]

        if expected_hidden_size == sharded_embedding_size * input_tp_size:
            result = "EmbeddingParallelEmbedding"
        else:
            result = "VocabParallelEmbedding"

        assert result == "VocabParallelEmbedding"


class TestMapTpToMonolith:
    """Tests for map_tp_to_monolith function"""

    def _create_tp_checkpoints(self, tp_size=2):
        """Create mock TP checkpoints"""
        hidden_size = 64
        num_heads = 4
        head_dim = hidden_size // num_heads

        checkpoints = []
        for tp_rank in range(tp_size):
            ckpt = {
                "rng": {"state": torch.get_rng_state()},
                "state": {
                    "model": OrderedDict(
                        [
                            # Column parallel - sharded on dim 0
                            (
                                "model.layers.0.self_attn.q_proj.weight",
                                torch.randn(hidden_size // tp_size, hidden_size),
                            ),
                            (
                                "model.layers.0.self_attn.k_proj.weight",
                                torch.randn(hidden_size // tp_size, hidden_size),
                            ),
                            (
                                "model.layers.0.self_attn.v_proj.weight",
                                torch.randn(hidden_size // tp_size, hidden_size),
                            ),
                            # Row parallel - sharded on dim 1
                            (
                                "model.layers.0.self_attn.o_proj.weight",
                                torch.randn(hidden_size, hidden_size // tp_size),
                            ),
                            # MLP
                            (
                                "model.layers.0.mlp.gate_proj.weight",
                                torch.randn(hidden_size // tp_size, hidden_size),
                            ),
                            (
                                "model.layers.0.mlp.down_proj.weight",
                                torch.randn(hidden_size, hidden_size // tp_size),
                            ),
                            # Non-sharded
                            (
                                "model.layers.0.input_layernorm.weight",
                                torch.randn(hidden_size),
                            ),
                        ]
                    ),
                    "integrations": {
                        "huggingface": {
                            "model": {
                                "config": {
                                    "content": {
                                        "hidden_size": hidden_size,
                                        "num_attention_heads": num_heads,
                                        "num_key_value_heads": num_heads,
                                        "model_type": "gigar",
                                        "tp_size": tp_size,
                                    }
                                }
                            }
                        }
                    },
                },
            }
            checkpoints.append(ckpt)

        return checkpoints

    def test_merges_column_parallel(self):
        """Test that column parallel layers are merged correctly"""
        from mappers import map_tp_to_monolith

        tp_ckpts = self._create_tp_checkpoints(tp_size=2)
        args = argparse.Namespace(skip_optimizer=True, input_tp_size=2)

        result = map_tp_to_monolith(tp_ckpts, args=args)

        # q_proj should be concatenated on dim 0
        q_proj = result["state"]["model"]["model.layers.0.self_attn.q_proj.weight"]
        assert q_proj.shape[0] == 64  # Full hidden_size

    def test_merges_row_parallel(self):
        """Test that row parallel layers are merged correctly"""
        from mappers import map_tp_to_monolith

        tp_ckpts = self._create_tp_checkpoints(tp_size=2)
        args = argparse.Namespace(skip_optimizer=True, input_tp_size=2)

        result = map_tp_to_monolith(tp_ckpts, args=args)

        # o_proj should be concatenated on dim 1
        o_proj = result["state"]["model"]["model.layers.0.self_attn.o_proj.weight"]
        assert o_proj.shape[1] == 64  # Full hidden_size

    def test_raises_on_packed_attention(self):
        """Test that error is raised when packed attention is present"""
        from mappers import map_tp_to_monolith

        tp_ckpts = [
            {
                "state": {
                    "model": {
                        "model.layers.0.self_attn.qkv_proj.weight": torch.randn(192, 64)
                    },
                    "integrations": {
                        "huggingface": {"model": {"config": {"content": {}}}}
                    },
                }
            }
        ]
        args = argparse.Namespace(skip_optimizer=True)

        with pytest.raises(AssertionError):
            map_tp_to_monolith(tp_ckpts, args=args)


class TestMapMonolithToTp:
    """Tests for map_monolith_to_tp function"""

    def _create_monolith_checkpoint(self):
        """Create mock monolith checkpoint"""
        hidden_size = 64

        return {
            "rng": {"state": torch.get_rng_state()},
            "state": {
                "model": OrderedDict(
                    [
                        (
                            "model.layers.0.self_attn.q_proj.weight",
                            torch.randn(hidden_size, hidden_size),
                        ),
                        (
                            "model.layers.0.self_attn.k_proj.weight",
                            torch.randn(hidden_size, hidden_size),
                        ),
                        (
                            "model.layers.0.self_attn.v_proj.weight",
                            torch.randn(hidden_size, hidden_size),
                        ),
                        (
                            "model.layers.0.self_attn.o_proj.weight",
                            torch.randn(hidden_size, hidden_size),
                        ),
                        (
                            "model.layers.0.mlp.gate_proj.weight",
                            torch.randn(hidden_size, hidden_size),
                        ),
                        (
                            "model.layers.0.mlp.down_proj.weight",
                            torch.randn(hidden_size, hidden_size),
                        ),
                        (
                            "model.layers.0.input_layernorm.weight",
                            torch.randn(hidden_size),
                        ),
                        ("model.embed_tokens.weight", torch.randn(1000, hidden_size)),
                    ]
                ),
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "hidden_size": hidden_size,
                                    "num_attention_heads": 4,
                                    "num_key_value_heads": 4,
                                    "model_type": "gigar",
                                    "tp_size": 1,
                                }
                            }
                        }
                    }
                },
            },
        }

    def test_splits_to_tp(self):
        """Test that monolith is split into TP shards"""
        from mappers import map_monolith_to_tp

        mono_ckpt = self._create_monolith_checkpoint()
        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=2,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(mono_ckpt, args=args)

        assert len(result) == 2

        # Each shard should have half the q_proj on dim 0
        for shard in result:
            q_proj = shard["state"]["model"]["model.layers.0.self_attn.q_proj.weight"]
            assert q_proj.shape[0] == 32  # 64 / 2

    def test_updates_tp_size_in_config(self):
        """Test that tp_size is updated in config"""
        from mappers import map_monolith_to_tp

        mono_ckpt = self._create_monolith_checkpoint()
        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=4,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(mono_ckpt, args=args)

        for shard in result:
            config = shard["state"]["integrations"]["huggingface"]["model"]["config"][
                "content"
            ]
            assert config["tp_size"] == 4


class TestMapTpToTp:
    """Tests for map_tp_to_tp function"""

    def test_tp2_to_tp4(self):
        """Test conversion from TP=2 to TP=4"""
        from mappers import map_tp_to_tp

        hidden_size = 64

        # Create TP=2 checkpoints
        tp2_ckpts = []
        for tp_rank in range(2):
            ckpt = {
                "state": {
                    "model": OrderedDict(
                        [
                            (
                                "model.layers.0.self_attn.q_proj.weight",
                                torch.randn(hidden_size // 2, hidden_size),
                            ),
                        ]
                    ),
                    "integrations": {
                        "huggingface": {
                            "model": {
                                "config": {
                                    "content": {
                                        "hidden_size": hidden_size,
                                        "num_attention_heads": 4,
                                        "num_key_value_heads": 4,
                                        "model_type": "gigar",
                                        "tp_size": 2,
                                    }
                                }
                            }
                        }
                    },
                }
            }
            tp2_ckpts.append(ckpt)

        args = argparse.Namespace(
            skip_optimizer=True, output_tp_size=4, input_tp_size=2
        )

        result = map_tp_to_tp(tp2_ckpts, args=args)

        assert len(result) == 4

        # Each shard should have quarter of q_proj
        for shard in result:
            q_proj = shard["state"]["model"]["model.layers.0.self_attn.q_proj.weight"]
            assert q_proj.shape[0] == 16  # 64 / 4


class TestIsKeyContainedInModel:
    """Tests for is_key_contained_in_model helper"""

    def test_key_found(self):
        """Test when key is found in model"""
        from mappers import is_key_contained_in_model

        ckpt = {
            "state": {
                "model": {"model.layers.0.self_attn.qkv_proj.weight": torch.randn(4)}
            }
        }

        result = is_key_contained_in_model(ckpt, "qkv_proj")

        assert result is True

    def test_key_not_found(self):
        """Test when key is not found in model"""
        from mappers import is_key_contained_in_model

        ckpt = {
            "state": {
                "model": {"model.layers.0.self_attn.q_proj.weight": torch.randn(4)}
            }
        }

        result = is_key_contained_in_model(ckpt, "qkv_proj")

        assert result is False


class TestLoadCheckpoint:
    """Tests for load_checkpoint function"""

    @patch("mappers.torch.load")
    @patch("mappers.os.path.isdir")
    def test_loads_monolith_checkpoint(self, mock_isdir, mock_torch_load):
        """Test loading monolith checkpoint"""
        mock_isdir.return_value = False
        mock_torch_load.return_value = {"state": {"model": {}}}

        from mappers import load_checkpoint

        args = argparse.Namespace()
        result = load_checkpoint("/path/to/checkpoint.pt", args=args)

        mock_torch_load.assert_called_once()

    @patch("mappers.load_gigafsdp_checkpoint")
    @patch("mappers.os.path.isdir")
    def test_loads_sharded_checkpoint(self, mock_isdir, mock_load_gigafsdp):
        """Test loading sharded checkpoint directory"""
        mock_isdir.return_value = True
        mock_load_gigafsdp.return_value = {"state": {"model": {}}}

        from mappers import load_checkpoint

        args = argparse.Namespace()
        result = load_checkpoint("/path/to/checkpoint_dir", args=args)

        mock_load_gigafsdp.assert_called_once()


class TestRemoveFsdpOptimizer:
    """Tests for remove_fsdp_optimizer function"""

    def test_removes_optimizer_from_state(self):
        """Test that optimizer is removed from state"""
        from mappers import remove_fsdp_optimizer

        fsdp_state_dict = {
            "state": {
                "model": {"weight": torch.randn(4)},
                "optimizers": {"weight": {"exp_avg": torch.randn(4)}},
            }
        }
        args = argparse.Namespace()

        result = remove_fsdp_optimizer(fsdp_state_dict, args=args)

        assert (
            "optimizers" not in result["state"] or result["state"]["optimizers"] == {}
        )


class TestLoadHfModel:
    """Tests for load_hf_model function"""

    @patch("mappers.AutoModelForCausalLM")
    def test_loads_model(self, mock_auto_model):
        """Test that HF model is loaded"""
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        from mappers import load_hf_model

        # load_hf_model requires checkpoint_path as first argument
        result = load_hf_model(
            checkpoint_path="/path/to/model",
            args=argparse.Namespace(torch_dtype="bfloat16"),
        )

        mock_auto_model.from_pretrained.assert_called_once()
        assert result == mock_model


class TestMapCheckpointFunctionsRegistry:
    """Tests for MAP_CHECKPOINT_FUNCTIONS registry"""

    def test_all_functions_registered(self):
        """Test that all expected functions are in registry"""
        from mappers import MAP_CHECKPOINT_FUNCTIONS

        expected_functions = [
            "load_checkpoint",
            "map_base_to_packed_attn",
            "map_packed_to_base_attn",
            "map_tp_to_monolith",
            "map_monolith_to_tp",
            "map_fsdp_to_hf",
            "map_hf_to_fsdp",
            "map_merge_lora",
        ]

        for func_name in expected_functions:
            assert func_name in MAP_CHECKPOINT_FUNCTIONS
            assert callable(MAP_CHECKPOINT_FUNCTIONS[func_name])


class TestMapArchNames:
    """Tests for MAP_ARCH_NAMES constant"""

    def test_gigar_mapping(self):
        """Test gigar maps to llama"""
        from mappers import MAP_ARCH_NAMES

        assert MAP_ARCH_NAMES["gigar"] == "llama"

    def test_papayar_mapping(self):
        """Test papayar maps to llama"""
        from mappers import MAP_ARCH_NAMES

        assert MAP_ARCH_NAMES["papayar"] == "llama"


class TestMapModelTypeToArch:
    """Tests for MAP_MODEL_TYPE_TO_ARCH constant"""

    def test_llama_mapping(self):
        """Test llama maps to LlamaForCausalLM"""
        from mappers import MAP_MODEL_TYPE_TO_ARCH

        assert MAP_MODEL_TYPE_TO_ARCH["llama"] == "LlamaForCausalLM"

    def test_qwen2_mapping(self):
        """Test qwen2 maps to Qwen2ForCausalLM"""
        from mappers import MAP_MODEL_TYPE_TO_ARCH

        assert MAP_MODEL_TYPE_TO_ARCH["qwen2"] == "Qwen2ForCausalLM"

    def test_qwen3_mapping(self):
        """Test qwen3 maps to Qwen3ForCausalLM"""
        from mappers import MAP_MODEL_TYPE_TO_ARCH

        assert MAP_MODEL_TYPE_TO_ARCH["qwen3"] == "Qwen3ForCausalLM"
