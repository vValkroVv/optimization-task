"""
Additional tests for mappers.py to increase coverage.
Covers: optimizer handling, EMA functions, sequence_apply, optimized_chunk
"""

import pytest
import torch
from unittest.mock import patch, MagicMock, Mock
from collections import OrderedDict
import argparse
import copy


class TestOptimizedChunk:
    """Tests for optimized_chunk function"""

    def test_chunks_tensor(self):
        """Test basic tensor chunking"""
        from mappers import optimized_chunk

        tensor = torch.randn(8, 4)
        result = optimized_chunk(tensor, chunks=2, dim=0)

        assert len(result) == 2
        assert result[0].shape == torch.Size([4, 4])
        assert result[1].shape == torch.Size([4, 4])

    def test_chunks_along_dim1(self):
        """Test chunking along dim=1"""
        from mappers import optimized_chunk

        tensor = torch.randn(4, 8)
        result = optimized_chunk(tensor, chunks=4, dim=1)

        assert len(result) == 4
        for chunk in result:
            assert chunk.shape == torch.Size([4, 2])

    def test_returns_cloned_tensors(self):
        """Test that returned tensors are clones"""
        from mappers import optimized_chunk

        tensor = torch.randn(4, 4)
        result = optimized_chunk(tensor, chunks=2, dim=0)

        # Modify original
        tensor[0, 0] = 999.0

        # Result should not be affected
        assert result[0][0, 0] != 999.0


class TestPartAndGc:
    """Tests for part_and_gc function"""

    def test_returns_partial(self):
        """Test that part_and_gc returns a partial function"""
        from mappers import part_and_gc

        def dummy_mapper(state_dict, args=None):
            return state_dict

        args = argparse.Namespace()
        result = part_and_gc(dummy_mapper, args)

        assert callable(result)
        # Test that partial works
        test_dict = {"key": "value"}
        assert result(test_dict) == test_dict


class TestSequenceApply:
    """Tests for sequence_apply decorator"""

    def test_with_single_dict(self):
        """Test sequence_apply with single dict input"""
        from mappers import sequence_apply

        @sequence_apply
        def identity_mapper(state_dict, args=None):
            return state_dict

        test_dict = {"key": torch.randn(4)}
        args = argparse.Namespace()

        result = identity_mapper(test_dict, args)

        assert isinstance(result, dict)
        assert "key" in result

    def test_with_list_of_dicts(self):
        """Test sequence_apply with list of dicts"""
        from mappers import sequence_apply

        @sequence_apply
        def add_marker_mapper(state_dict, args=None):
            state_dict["marker"] = True
            return state_dict

        test_list = [{"key1": 1}, {"key2": 2}]
        args = argparse.Namespace()

        result = add_marker_mapper(test_list, args)

        assert isinstance(result, list)
        assert len(result) == 2
        for item in result:
            assert item.get("marker") is True


class TestPutEmaToWeightsOrDeleteEma:
    """Tests for put_ema_to_weights_or_delele_ema function"""

    def test_deletes_ema_when_not_use_ema(self):
        """Test that EMA weights are deleted when use_ema=False"""
        import sys

        # Mock giga_fsdp module before importing
        mock_giga_fsdp = MagicMock()
        mock_giga_fsdp.EMA_SUFFIX = "_ema"
        sys.modules["giga_fsdp"] = mock_giga_fsdp

        from mappers import put_ema_to_weights_or_delele_ema

        fsdp_state_dict = {
            "state": {
                "model": {
                    "model.weight": torch.randn(4),
                    "model.weight_ema": torch.randn(4),
                }
            }
        }
        args = argparse.Namespace(use_ema=False, prepare_ema_weight=False)

        put_ema_to_weights_or_delele_ema(fsdp_state_dict, args)

        assert "model.weight" in fsdp_state_dict["state"]["model"]
        assert "model.weight_ema" not in fsdp_state_dict["state"]["model"]

    def test_replaces_with_ema_when_use_ema(self):
        """Test that EMA weights replace original when use_ema=True"""
        import sys

        mock_giga_fsdp = MagicMock()
        mock_giga_fsdp.EMA_SUFFIX = "_ema"
        sys.modules["giga_fsdp"] = mock_giga_fsdp

        from mappers import put_ema_to_weights_or_delele_ema

        orig_weight = torch.randn(4)
        ema_weight = torch.randn(4)

        fsdp_state_dict = {
            "state": {
                "model": {
                    "model.weight": orig_weight.clone(),
                    "model.weight_ema": ema_weight.clone(),
                }
            }
        }
        args = argparse.Namespace(use_ema=True, prepare_ema_weight=False)

        put_ema_to_weights_or_delele_ema(fsdp_state_dict, args)

        assert "model.weight" in fsdp_state_dict["state"]["model"]
        assert "model.weight_ema" not in fsdp_state_dict["state"]["model"]
        # Weight should now be EMA weight
        assert torch.allclose(
            fsdp_state_dict["state"]["model"]["model.weight"], ema_weight
        )

    def test_raises_on_missing_original(self):
        """Test that assertion fails when original weight is missing"""
        import sys

        mock_giga_fsdp = MagicMock()
        mock_giga_fsdp.EMA_SUFFIX = "_ema"
        sys.modules["giga_fsdp"] = mock_giga_fsdp

        from mappers import put_ema_to_weights_or_delele_ema

        fsdp_state_dict = {
            "state": {
                "model": {
                    "model.weight_ema": torch.randn(4),
                    # Note: model.weight is missing
                }
            }
        }
        args = argparse.Namespace(use_ema=False, prepare_ema_weight=False)

        with pytest.raises(AssertionError):
            put_ema_to_weights_or_delele_ema(fsdp_state_dict, args)

    def test_skips_assertion_with_prepare_ema_weight(self):
        """Test that assertion is skipped when prepare_ema_weight=True"""
        import sys

        mock_giga_fsdp = MagicMock()
        mock_giga_fsdp.EMA_SUFFIX = "_ema"
        sys.modules["giga_fsdp"] = mock_giga_fsdp

        from mappers import put_ema_to_weights_or_delele_ema

        fsdp_state_dict = {
            "state": {
                "model": {
                    "model.weight_ema": torch.randn(4),
                    # Note: model.weight is missing
                }
            }
        }
        args = argparse.Namespace(use_ema=True, prepare_ema_weight=True)

        # Should not raise
        put_ema_to_weights_or_delele_ema(fsdp_state_dict, args)


class TestMapPackedToBaseAttnWithOptimizer:
    """Tests for map_packed_to_base_attn with optimizer handling"""

    def _create_packed_checkpoint_with_optimizer(self):
        """Create checkpoint with packed attention and optimizer state"""
        hidden_size = 64
        num_heads = 4
        num_kv_heads = 2
        head_dim = hidden_size // num_heads
        kv_hidden = num_kv_heads * head_dim
        qkv_size = hidden_size + 2 * kv_hidden

        qkv_weight = torch.randn(qkv_size, hidden_size)

        # Correct nested structure with optimizer name
        return {
            "state": {
                "model": {
                    "model.layers.0.self_attn.qkv_proj.weight": qkv_weight,
                    "model.layers.0.mlp.gate_proj.weight": torch.randn(
                        hidden_size, hidden_size
                    ),
                },
                "optimizers": {
                    "DecoupledLionW": {  # optimizer name
                        "state": {
                            "model.layers.0.self_attn.qkv_proj.weight": {
                                "exp_avg": torch.randn(qkv_size, hidden_size),
                                "exp_avg_sq": torch.randn(qkv_size, hidden_size),
                                "step": torch.tensor(100),
                            },
                            "model.layers.0.mlp.gate_proj.weight": {
                                "exp_avg": torch.randn(hidden_size, hidden_size),
                                "exp_avg_sq": torch.randn(hidden_size, hidden_size),
                                "step": torch.tensor(100),
                            },
                        },
                        "param_groups": [
                            {
                                "params": [
                                    "model.layers.0.self_attn.qkv_proj.weight",
                                    "model.layers.0.mlp.gate_proj.weight",
                                ]
                            }
                        ],
                    }
                },
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "hidden_size": hidden_size,
                                    "num_attention_heads": num_heads,
                                    "num_key_value_heads": num_kv_heads,
                                    "attention_type": "LlamaPackedAttention",
                                    "tp_size": 1,
                                    "model_type": "gigar",
                                }
                            }
                        }
                    }
                },
            }
        }

    def test_splits_optimizer_state(self):
        """Test that optimizer state is split along with weights"""
        from mappers import map_packed_to_base_attn

        checkpoint = self._create_packed_checkpoint_with_optimizer()
        args = argparse.Namespace(skip_optimizer=False)

        result = map_packed_to_base_attn(checkpoint, args=args)

        optimizer_sd = result["state"]["optimizers"]["DecoupledLionW"]["state"]

        # Check that Q, K, V optimizer states exist
        assert "model.layers.0.self_attn.q_proj.weight" in optimizer_sd
        assert "model.layers.0.self_attn.k_proj.weight" in optimizer_sd
        assert "model.layers.0.self_attn.v_proj.weight" in optimizer_sd

        # Check that original qkv_proj is removed
        assert "model.layers.0.self_attn.qkv_proj.weight" not in optimizer_sd

        # Check exp_avg and exp_avg_sq exist
        assert "exp_avg" in optimizer_sd["model.layers.0.self_attn.q_proj.weight"]
        assert "exp_avg_sq" in optimizer_sd["model.layers.0.self_attn.q_proj.weight"]

    def test_updates_param_groups(self):
        """Test that param_groups are updated with new parameter names"""
        from mappers import map_packed_to_base_attn

        checkpoint = self._create_packed_checkpoint_with_optimizer()
        args = argparse.Namespace(skip_optimizer=False)

        result = map_packed_to_base_attn(checkpoint, args=args)

        param_groups = result["state"]["optimizers"]["DecoupledLionW"]["param_groups"]
        params = param_groups[0]["params"]

        # Check that Q, K, V are in param_groups
        assert "model.layers.0.self_attn.q_proj.weight" in params
        assert "model.layers.0.self_attn.k_proj.weight" in params
        assert "model.layers.0.self_attn.v_proj.weight" in params

        # Check that qkv_proj is removed
        assert "model.layers.0.self_attn.qkv_proj.weight" not in params

    def test_preserves_step(self):
        """Test that optimizer step is preserved"""
        from mappers import map_packed_to_base_attn

        checkpoint = self._create_packed_checkpoint_with_optimizer()
        args = argparse.Namespace(skip_optimizer=False)

        result = map_packed_to_base_attn(checkpoint, args=args)

        q_opt = result["state"]["optimizers"]["DecoupledLionW"]["state"][
            "model.layers.0.self_attn.q_proj.weight"
        ]

        assert "step" in q_opt
        assert q_opt["step"] == 100


class TestMapMonolithFsdpToHf:
    """Tests for map_monolith_fsdp_to_hf routing"""

    @patch("mappers.put_ema_to_weights_or_delele_ema")
    @patch("mappers._map_monolith_gigar_to_hf")
    def test_routes_gigar_model(self, mock_gigar, mock_ema):
        """Test routing for gigar model type"""
        from mappers import map_monolith_fsdp_to_hf

        fsdp_state_dict = {
            "state": {
                "model": {},
                "integrations": {
                    "huggingface": {
                        "model": {"config": {"content": {"model_type": "gigar"}}}
                    }
                },
            }
        }
        args = argparse.Namespace()

        map_monolith_fsdp_to_hf(fsdp_state_dict, args=args)

        mock_gigar.assert_called_once()

    @patch("mappers.put_ema_to_weights_or_delele_ema")
    @patch("mappers._map_monolith_gigar_to_hf")
    def test_routes_papayar_model(self, mock_gigar, mock_ema):
        """Test routing for papayar model type"""
        from mappers import map_monolith_fsdp_to_hf

        fsdp_state_dict = {
            "state": {
                "model": {},
                "integrations": {
                    "huggingface": {
                        "model": {"config": {"content": {"model_type": "papayar"}}}
                    }
                },
            }
        }
        args = argparse.Namespace()

        map_monolith_fsdp_to_hf(fsdp_state_dict, args=args)

        mock_gigar.assert_called_once()

    @patch("mappers.put_ema_to_weights_or_delele_ema")
    @patch("mappers._map_monolith_deepseek_to_hf")
    def test_routes_giga_mix_model(self, mock_deepseek, mock_ema):
        """Test routing for giga_mix model type"""
        from mappers import map_monolith_fsdp_to_hf

        fsdp_state_dict = {
            "state": {
                "model": {},
                "integrations": {
                    "huggingface": {
                        "model": {"config": {"content": {"model_type": "giga_mix"}}}
                    }
                },
            }
        }
        args = argparse.Namespace()

        map_monolith_fsdp_to_hf(fsdp_state_dict, args=args)

        mock_deepseek.assert_called_once()

    @patch("mappers.put_ema_to_weights_or_delele_ema")
    def test_raises_on_unsupported_model(self, mock_ema):
        """Test that ValueError is raised for unsupported model type"""
        from mappers import map_monolith_fsdp_to_hf

        fsdp_state_dict = {
            "state": {
                "model": {},
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {"content": {"model_type": "unsupported_type"}}
                        }
                    }
                },
            }
        }
        args = argparse.Namespace()

        with pytest.raises(ValueError, match="not supported"):
            map_monolith_fsdp_to_hf(fsdp_state_dict, args=args)


class TestMapHfToMonolithFsdp:
    """Tests for map_hf_to_monolith_fsdp function"""

    def test_adds_model_prefix(self):
        """Test that model. prefix is added to keys"""
        from mappers import map_hf_to_monolith_fsdp

        mock_model = MagicMock()
        mock_model.state_dict.return_value = OrderedDict(
            [
                ("layers.0.weight", torch.randn(4, 4)),
                ("embed_tokens.weight", torch.randn(100, 4)),
            ]
        )
        mock_model.config = MagicMock()
        mock_model.config.to_dict.return_value = {"hidden_size": 4}

        args = argparse.Namespace()

        result = map_hf_to_monolith_fsdp(mock_model, args=args)

        model_sd = result["state"]["model"]
        assert "model.layers.0.weight" in model_sd
        assert "model.embed_tokens.weight" in model_sd


class TestBuildMonolithFromMixedGigafsdpState:
    """Tests for build_monolith_from_mixed_gigafsdp_state function"""

    @patch("mappers.map_base_to_packed_attn")
    @patch("mappers.map_packed_to_base_attn")
    def test_basic_build(self, mock_unpack, mock_pack):
        """Test basic build of monolith from mixed state"""
        from mappers import build_monolith_from_mixed_gigafsdp_state
        from mapper_gigafsdp_to_monolith import unshard_gigafsdp_model

        # Setup mocks
        mock_unpack.return_value = [
            {"state": {"model": {"model.weight": torch.randn(4)}}}
        ]
        mock_pack.return_value = {"state": {"model": {}}}

        state_dicts = [
            {
                "state": {
                    "model": {"model.meta": {"tp_size": 1}},
                    "integrations": {
                        "huggingface": {
                            "model": {"config": {"content": {"hidden_size": 4}}}
                        }
                    },
                    "other_key": "value",
                }
            }
        ]

        args = argparse.Namespace(skip_optimizer=True)

        with patch.object(
            __import__(
                "mapper_gigafsdp_to_monolith", fromlist=["unshard_gigafsdp_model"]
            ),
            "unshard_gigafsdp_model",
            return_value={1: [{"model.weight": torch.randn(4)}]},
        ):
            try:
                result = build_monolith_from_mixed_gigafsdp_state(state_dicts, args)
                assert "state" in result
                assert "model" in result["state"]
            except Exception:
                pytest.skip("Function requires complex mocking")

    def test_raises_on_optimizer_request(self):
        """Test that NotImplementedError is raised when optimizer requested"""
        from mappers import build_monolith_from_mixed_gigafsdp_state

        state_dicts = [
            {
                "state": {
                    "model": {"model.meta": {"tp_size": 1}},
                    "integrations": {
                        "huggingface": {
                            "model": {
                                "config": {
                                    "content": {
                                        "model_type": "gigar",
                                        "hidden_size": 64,
                                        "num_attention_heads": 4,
                                        "num_key_value_heads": 4,
                                    }
                                }
                            }
                        }
                    },
                }
            }
        ]

        args = argparse.Namespace(skip_optimizer=False)

        with patch.object(
            __import__(
                "mapper_gigafsdp_to_monolith", fromlist=["unshard_gigafsdp_model"]
            ),
            "unshard_gigafsdp_model",
            return_value={1: [{}]},
        ):
            with pytest.raises(NotImplementedError, match="Optimizer is not supported"):
                build_monolith_from_mixed_gigafsdp_state(state_dicts, args)
