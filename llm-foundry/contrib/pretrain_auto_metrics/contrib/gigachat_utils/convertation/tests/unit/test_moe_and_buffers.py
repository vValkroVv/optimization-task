import pytest
import torch
from unittest.mock import patch, MagicMock
from collections import OrderedDict
import argparse
import sys


class TestUnshardFlatTensor:
    """Tests for unshard_flat_tensor with different param_groups (buffer types)"""

    def test_unshard_main_param_group(self):
        """Test unsharding main (trainable) parameter group"""
        from mapper_gigafsdp_to_monolith import unshard_flat_tensor

        # Meta for main parameters - 1 layer with 16 elements -> 4x4 tensor
        # shard_size = 16 means each rank has 16 elements total for this layer
        meta = {
            "main_parameter_shard_bounaries": [16],
            "main_parameter_orig_shapes": [[[4, 4]]],
            "main_shards_fqns": [["layers.0.q_proj.weight"]],
        }

        # 2 FSDP ranks, each with shard_size=16 elements
        # Total = 32 elements, reshaped to 4x4 (takes first 16)
        flatten_tensors = [
            torch.arange(16, dtype=torch.float32),  # rank 0: 16 elements
            torch.arange(16, 32, dtype=torch.float32),  # rank 1: 16 elements
        ]

        result = unshard_flat_tensor(meta, flatten_tensors, param_group="main")

        assert "model.layers.0.q_proj.weight" in result
        assert result["model.layers.0.q_proj.weight"].shape == torch.Size([4, 4])

    def test_unshard_layernorm_param_group(self):
        """Test unsharding layernorm (separate) parameter group"""
        from mapper_gigafsdp_to_monolith import unshard_flat_tensor

        # shard_size = 8, each rank has 8 elements
        meta = {
            "layernorm_parameter_shard_bounaries": [8],
            "layernorm_parameter_orig_shapes": [[[8]]],
            "layernorm_shards_fqns": [["layers.0.input_layernorm.weight"]],
        }

        # 2 FSDP ranks, each with 8 elements
        flatten_tensors = [
            torch.arange(8, dtype=torch.float32),
            torch.arange(8, 16, dtype=torch.float32),
        ]

        result = unshard_flat_tensor(meta, flatten_tensors, param_group="layernorm")

        assert "model.layers.0.input_layernorm.weight" in result
        assert result["model.layers.0.input_layernorm.weight"].shape == torch.Size([8])

    def test_unshard_freezed_param_group(self):
        """Test unsharding freezed parameter group"""
        from mapper_gigafsdp_to_monolith import unshard_flat_tensor

        # shard_size = 16, each rank has 16 elements
        meta = {
            "freezed_parameter_shard_bounaries": [16],
            "freezed_parameter_orig_shapes": [[[4, 4]]],
            "freezed_shards_fqns": [["embed_tokens.weight"]],
        }

        flatten_tensors = [
            torch.arange(16, dtype=torch.float32),
            torch.arange(16, 32, dtype=torch.float32),
        ]

        result = unshard_flat_tensor(meta, flatten_tensors, param_group="freezed")

        assert "model.embed_tokens.weight" in result
        assert result["model.embed_tokens.weight"].shape == torch.Size([4, 4])

    def test_unshard_layernormfreezed_param_group(self):
        """Test unsharding layernormfreezed parameter group"""
        from mapper_gigafsdp_to_monolith import unshard_flat_tensor

        # shard_size = 8
        meta = {
            "layernormfreezed_parameter_shard_bounaries": [8],
            "layernormfreezed_parameter_orig_shapes": [[[8]]],
            "layernormfreezed_shards_fqns": [
                ["layers.0.post_attention_layernorm.weight"]
            ],
        }

        flatten_tensors = [
            torch.randn(8),
            torch.randn(8),
        ]

        result = unshard_flat_tensor(
            meta, flatten_tensors, param_group="layernormfreezed"
        )

        assert "model.layers.0.post_attention_layernorm.weight" in result
        assert result[
            "model.layers.0.post_attention_layernorm.weight"
        ].shape == torch.Size([8])

    def test_unshard_with_ema_suffix(self):
        """Test unsharding with EMA weight suffix"""
        from mapper_gigafsdp_to_monolith import unshard_flat_tensor

        # shard_size = 8
        meta = {
            "main_parameter_shard_bounaries": [8],
            "main_parameter_orig_shapes": [[[8]]],
            "main_shards_fqns": [["layers.0.weight"]],
        }

        flatten_tensors = [
            torch.arange(8, dtype=torch.float32),
            torch.arange(8, 16, dtype=torch.float32),
        ]

        result = unshard_flat_tensor(
            meta, flatten_tensors, param_group="main", param_name_suffix="_ema"
        )

        assert "model.layers.0.weight_ema" in result
        assert "model.layers.0.weight" not in result

    def test_unshard_multiple_params_per_layer(self):
        """Test unsharding layer with multiple parameters"""
        from mapper_gigafsdp_to_monolith import unshard_flat_tensor

        # One layer with 2 params: weight (4x4=16) and bias (4) = 20 total
        # shard_size = 20
        meta = {
            "main_parameter_shard_bounaries": [20],
            "main_parameter_orig_shapes": [[[4, 4], [4]]],
            "main_shards_fqns": [["layers.0.proj.weight", "layers.0.proj.bias"]],
        }

        # 2 FSDP ranks, each with 20 elements
        flatten_tensors = [
            torch.arange(20, dtype=torch.float32),
            torch.arange(20, 40, dtype=torch.float32),
        ]

        result = unshard_flat_tensor(meta, flatten_tensors, param_group="main")

        assert "model.layers.0.proj.weight" in result
        assert "model.layers.0.proj.bias" in result
        assert result["model.layers.0.proj.weight"].shape == torch.Size([4, 4])
        assert result["model.layers.0.proj.bias"].shape == torch.Size([4])


class TestLegacyUnshardGigafsdpModel:
    """Tests for legacy_unshard_gigafsdp_model with all buffer types"""

    @patch("mapper_gigafsdp_to_monolith.giga_fsdp")
    def test_processes_all_buffer_types(self, mock_giga_fsdp):
        """Test that all buffer types are processed"""
        mock_giga_fsdp.EMA_SUFFIX = "_ema"

        from mapper_gigafsdp_to_monolith import legacy_unshard_gigafsdp_model

        # Create state dicts with all buffer types
        # Each rank has shard_size elements for each buffer
        state_dicts = [
            OrderedDict(
                [
                    ("model._main_parameter", torch.arange(16, dtype=torch.float32)),
                    (
                        "model._layernorm_parameter",
                        torch.arange(8, dtype=torch.float32),
                    ),
                    ("model._freezed_parameter", torch.arange(16, dtype=torch.float32)),
                ]
            ),
            OrderedDict(
                [
                    (
                        "model._main_parameter",
                        torch.arange(16, 32, dtype=torch.float32),
                    ),
                    (
                        "model._layernorm_parameter",
                        torch.arange(8, 16, dtype=torch.float32),
                    ),
                    (
                        "model._freezed_parameter",
                        torch.arange(16, 32, dtype=torch.float32),
                    ),
                ]
            ),
        ]

        meta = {
            "main_parameter_shard_bounaries": [16],
            "main_parameter_orig_shapes": [[[4, 4]]],
            "main_shards_fqns": [["layers.0.q_proj.weight"]],
            "layernorm_parameter_shard_bounaries": [8],
            "layernorm_parameter_orig_shapes": [[[8]]],
            "layernorm_shards_fqns": [["layers.0.input_layernorm.weight"]],
            "freezed_parameter_shard_bounaries": [16],
            "freezed_parameter_orig_shapes": [[[4, 4]]],
            "freezed_shards_fqns": [["embed_tokens.weight"]],
            "fsdp_group_size": 2,
        }

        result = legacy_unshard_gigafsdp_model(state_dicts, meta)

        # Result structure: {tp_size: [state_dict]}
        assert 1 in result
        model_sd = result[1][0]

        # Check all buffer types are in result
        assert "model.layers.0.q_proj.weight" in model_sd  # main
        assert "model.layers.0.input_layernorm.weight" in model_sd  # layernorm
        assert "model.embed_tokens.weight" in model_sd  # freezed

    @patch("mapper_gigafsdp_to_monolith.giga_fsdp")
    def test_skips_missing_buffer_types(self, mock_giga_fsdp):
        """Test that missing buffer types are skipped gracefully"""
        mock_giga_fsdp.EMA_SUFFIX = "_ema"

        from mapper_gigafsdp_to_monolith import legacy_unshard_gigafsdp_model

        # Only main buffer present, each rank has 16 elements
        state_dicts = [
            OrderedDict(
                [("model._main_parameter", torch.arange(16, dtype=torch.float32))]
            ),
            OrderedDict(
                [("model._main_parameter", torch.arange(16, 32, dtype=torch.float32))]
            ),
        ]

        meta = {
            "main_parameter_shard_bounaries": [16],
            "main_parameter_orig_shapes": [[[4, 4]]],
            "main_shards_fqns": [["layers.0.weight"]],
            # layernorm meta present but not in state_dicts
            "layernorm_parameter_shard_bounaries": [8],
            "layernorm_parameter_orig_shapes": [[[8]]],
            "layernorm_shards_fqns": [["layers.0.ln.weight"]],
            "fsdp_group_size": 2,
        }

        # Should not raise
        result = legacy_unshard_gigafsdp_model(state_dicts, meta)

        assert 1 in result
        model_sd = result[1][0]
        assert "model.layers.0.weight" in model_sd
        assert "model.layers.0.ln.weight" not in model_sd  # skipped

    @patch("mapper_gigafsdp_to_monolith.giga_fsdp")
    def test_handles_ema_weights(self, mock_giga_fsdp):
        """Test that EMA weights are processed alongside regular weights"""
        mock_giga_fsdp.EMA_SUFFIX = "_ema"

        from mapper_gigafsdp_to_monolith import legacy_unshard_gigafsdp_model

        # Each rank has 16 elements for both main and EMA
        state_dicts = [
            OrderedDict(
                [
                    ("model._main_parameter", torch.arange(16, dtype=torch.float32)),
                    (
                        "model._main_parameter_ema",
                        torch.arange(100, 116, dtype=torch.float32),
                    ),
                ]
            ),
            OrderedDict(
                [
                    (
                        "model._main_parameter",
                        torch.arange(16, 32, dtype=torch.float32),
                    ),
                    (
                        "model._main_parameter_ema",
                        torch.arange(116, 132, dtype=torch.float32),
                    ),
                ]
            ),
        ]

        meta = {
            "main_parameter_shard_bounaries": [16],
            "main_parameter_orig_shapes": [[[4, 4]]],
            "main_shards_fqns": [["layers.0.weight"]],
            "fsdp_group_size": 2,
        }

        result = legacy_unshard_gigafsdp_model(state_dicts, meta)

        model_sd = result[1][0]
        assert "model.layers.0.weight" in model_sd
        assert "model.layers.0.weight_ema" in model_sd

        # Verify EMA has different values
        assert not torch.allclose(
            model_sd["model.layers.0.weight"], model_sd["model.layers.0.weight_ema"]
        )


class TestCheckBufferParams:
    """Tests for check_buffer_params function"""

    def test_detects_expert_bias_buffer(self):
        """Test detection of expert_bias buffer pattern"""
        from mapper_gigafsdp_to_monolith import check_buffer_params

        tensor = torch.randn(8)

        result_name, result_val = check_buffer_params(
            [tensor, tensor.clone()], "model.layers.0._gate.expert_bias"
        )

        assert result_name == "model.layers.0._gate.expert_bias"
        assert torch.equal(result_val, tensor)

    def test_returns_none_for_non_buffer(self):
        """Test that non-buffer patterns return None"""
        from mapper_gigafsdp_to_monolith import check_buffer_params

        tensor = torch.randn(64, 64)

        result_name, result_val = check_buffer_params(
            [tensor], "model.layers.0.self_attn.q_proj.weight"
        )

        assert result_name is None
        assert result_val is None

    def test_validates_buffer_consistency(self):
        """Test that buffer values must match across ranks"""
        from mapper_gigafsdp_to_monolith import check_buffer_params

        tensor1 = torch.randn(8)
        tensor2 = torch.randn(8)  # Different values

        # Should raise RuntimeError if buffers don't match
        with pytest.raises(RuntimeError, match="differs across FSDP ranks"):
            check_buffer_params([tensor1, tensor2], "model.layers.0._gate.expert_bias")


class TestNormalizeFunctions:
    """Tests for normalization functions"""

    def test_normalize_module_name(self):
        """Test _normalize_module_name removes _module and root_module"""
        from mapper_gigafsdp_to_monolith import _normalize_module_name

        assert _normalize_module_name("model._module.layers.0") == "model.layers.0"
        assert _normalize_module_name("model.root_module.layers.0") == "model.layers.0"
        assert (
            _normalize_module_name("model._module.root_module.layers.0")
            == "model.layers.0"
        )
        assert _normalize_module_name("model.layers.0") == "model.layers.0"

    def test_normalize_key_prefix(self):
        """Test _normalize_key_prefix handles model.model. prefix"""
        from mapper_gigafsdp_to_monolith import _normalize_key_prefix

        assert _normalize_key_prefix("model.model.layers.0") == "model.layers.0"
        assert _normalize_key_prefix("model.layers.0") == "model.layers.0"


# ============================================================================
# Test MoE Expert Parallelism
# ============================================================================


class TestMoEParallelKeyClassification:
    """Tests for MoE key classification in COLUMN_PARALLEL_KEYS and ROW_PARALLEL_KEYS"""

    def test_expert_weights_are_column_parallel(self):
        """Test that expert weights are in COLUMN_PARALLEL_KEYS"""
        from mappers import COLUMN_PARALLEL_KEYS

        moe_column_keys = [
            "block_sparse_moe.experts.weight1",
            "block_sparse_moe.experts.weight2",
            "block_sparse_moe.shared_w1",
            "block_sparse_moe.shared_w3",
            "block_sparse_moe.w2",
        ]

        for key in moe_column_keys:
            assert (
                key in COLUMN_PARALLEL_KEYS
            ), f"{key} should be in COLUMN_PARALLEL_KEYS"

    def test_moe_gate_is_row_parallel(self):
        """Test that MoE gate and some shared weights are in ROW_PARALLEL_KEYS"""
        from mappers import ROW_PARALLEL_KEYS

        moe_row_keys = [
            "block_sparse_moe.gate",
            "block_sparse_moe.shared_w2",
            "block_sparse_moe.w1",
            "block_sparse_moe.w3",
        ]

        for key in moe_row_keys:
            assert key in ROW_PARALLEL_KEYS, f"{key} should be in ROW_PARALLEL_KEYS"


class TestMapTpToMonolithWithMoE:
    """Tests for map_tp_to_monolith with MoE expert layers"""

    def _create_tp_moe_checkpoints(self, tp_size=2, num_experts=4, hidden_size=64):
        """Create TP-sharded MoE checkpoints"""
        intermediate_size = hidden_size * 2
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
                            # MoE shared experts - column parallel
                            (
                                "model.layers.0.block_sparse_moe.shared_w1",
                                torch.randn(intermediate_size // tp_size, hidden_size),
                            ),
                            (
                                "model.layers.0.block_sparse_moe.shared_w3",
                                torch.randn(intermediate_size // tp_size, hidden_size),
                            ),
                            # MoE shared experts - row parallel
                            (
                                "model.layers.0.block_sparse_moe.shared_w2",
                                torch.randn(hidden_size, intermediate_size // tp_size),
                            ),
                            # Router gate - row parallel (sharded on dim 1)
                            (
                                "model.layers.0.block_sparse_moe.gate",
                                torch.randn(hidden_size, num_experts // tp_size),
                            ),
                            # LayerNorm - not sharded
                            (
                                "model.layers.0.input_layernorm.weight",
                                torch.randn(hidden_size),
                            ),
                            (
                                "model.layers.0.post_attention_layernorm.weight",
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
                                        "intermediate_size": intermediate_size,
                                        "num_attention_heads": 4,
                                        "num_key_value_heads": 4,
                                        "num_routed_experts": num_experts,
                                        "model_type": "giga_mix",
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

    def test_merges_attention_layers(self):
        """Test that attention layers are merged correctly"""
        from mappers import map_tp_to_monolith

        tp_size = 2
        hidden_size = 64
        checkpoints = self._create_tp_moe_checkpoints(
            tp_size=tp_size, hidden_size=hidden_size
        )

        args = argparse.Namespace(skip_optimizer=True, input_tp_size=tp_size)

        result = map_tp_to_monolith(checkpoints, args=args)

        # Q projection should be merged on dim 0
        q_proj = result["state"]["model"]["model.layers.0.self_attn.q_proj.weight"]
        assert q_proj.shape == torch.Size([hidden_size, hidden_size])

        # O projection should be merged on dim 1
        o_proj = result["state"]["model"]["model.layers.0.self_attn.o_proj.weight"]
        assert o_proj.shape == torch.Size([hidden_size, hidden_size])

    def test_merges_moe_shared_experts(self):
        """Test that MoE shared experts are merged correctly"""
        from mappers import map_tp_to_monolith

        tp_size = 2
        hidden_size = 64
        intermediate_size = hidden_size * 2
        checkpoints = self._create_tp_moe_checkpoints(
            tp_size=tp_size, hidden_size=hidden_size
        )

        args = argparse.Namespace(skip_optimizer=True, input_tp_size=tp_size)

        result = map_tp_to_monolith(checkpoints, args=args)

        # shared_w1 is column parallel - merged on dim 0
        shared_w1 = result["state"]["model"][
            "model.layers.0.block_sparse_moe.shared_w1"
        ]
        assert shared_w1.shape == torch.Size([intermediate_size, hidden_size])

        # shared_w2 is row parallel - merged on dim 1
        shared_w2 = result["state"]["model"][
            "model.layers.0.block_sparse_moe.shared_w2"
        ]
        assert shared_w2.shape == torch.Size([hidden_size, intermediate_size])

    def test_merges_moe_router_gate(self):
        """Test that MoE router gate is merged correctly"""
        from mappers import map_tp_to_monolith

        tp_size = 2
        hidden_size = 64
        num_experts = 4
        checkpoints = self._create_tp_moe_checkpoints(
            tp_size=tp_size, hidden_size=hidden_size, num_experts=num_experts
        )

        args = argparse.Namespace(skip_optimizer=True, input_tp_size=tp_size)

        result = map_tp_to_monolith(checkpoints, args=args)

        # Gate is row parallel - merged on dim 1
        gate = result["state"]["model"]["model.layers.0.block_sparse_moe.gate"]
        assert gate.shape == torch.Size([hidden_size, num_experts])

    def test_layernorm_not_sharded(self):
        """Test that LayerNorm is taken from first shard (not concatenated)"""
        from mappers import map_tp_to_monolith

        tp_size = 2
        hidden_size = 64
        checkpoints = self._create_tp_moe_checkpoints(
            tp_size=tp_size, hidden_size=hidden_size
        )

        args = argparse.Namespace(skip_optimizer=True, input_tp_size=tp_size)

        result = map_tp_to_monolith(checkpoints, args=args)

        # LayerNorm should have original size (not doubled)
        ln = result["state"]["model"]["model.layers.0.input_layernorm.weight"]
        assert ln.shape == torch.Size([hidden_size])

    def test_tp_size_set_to_1(self):
        """Test that tp_size is set to 1 after merge"""
        from mappers import map_tp_to_monolith

        checkpoints = self._create_tp_moe_checkpoints(tp_size=2)
        args = argparse.Namespace(skip_optimizer=True, input_tp_size=2)

        result = map_tp_to_monolith(checkpoints, args=args)

        config = result["state"]["integrations"]["huggingface"]["model"]["config"][
            "content"
        ]
        assert config["tp_size"] == 1


class TestMapMonolithToTpWithMoE:
    """Tests for map_monolith_to_tp with MoE layers"""

    def _create_monolith_moe_checkpoint(self, num_experts=4, hidden_size=64):
        """Create monolith MoE checkpoint"""
        intermediate_size = hidden_size * 2

        return {
            "rng": {"state": torch.get_rng_state()},
            "state": {
                "model": OrderedDict(
                    [
                        # Full attention
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
                        # MoE shared experts
                        (
                            "model.layers.0.block_sparse_moe.shared_w1",
                            torch.randn(intermediate_size, hidden_size),
                        ),
                        (
                            "model.layers.0.block_sparse_moe.shared_w2",
                            torch.randn(hidden_size, intermediate_size),
                        ),
                        (
                            "model.layers.0.block_sparse_moe.shared_w3",
                            torch.randn(intermediate_size, hidden_size),
                        ),
                        # Router gate
                        (
                            "model.layers.0.block_sparse_moe.gate",
                            torch.randn(hidden_size, num_experts),
                        ),
                        # LayerNorm
                        (
                            "model.layers.0.input_layernorm.weight",
                            torch.randn(hidden_size),
                        ),
                        # Embeddings
                        ("model.embed_tokens.weight", torch.randn(1000, hidden_size)),
                    ]
                ),
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "hidden_size": hidden_size,
                                    "intermediate_size": intermediate_size,
                                    "num_attention_heads": 4,
                                    "num_key_value_heads": 4,
                                    "num_routed_experts": num_experts,
                                    "model_type": "giga_mix",
                                    "tp_size": 1,
                                }
                            }
                        }
                    }
                },
            },
        }

    def test_splits_attention_to_tp(self):
        """Test that attention is split correctly across TP ranks"""
        from mappers import map_monolith_to_tp

        tp_size = 2
        hidden_size = 64
        checkpoint = self._create_monolith_moe_checkpoint(hidden_size=hidden_size)

        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=tp_size,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(checkpoint, args=args)

        assert len(result) == tp_size

        # Q projection should be split on dim 0
        for shard in result:
            q_proj = shard["state"]["model"]["model.layers.0.self_attn.q_proj.weight"]
            assert q_proj.shape == torch.Size([hidden_size // tp_size, hidden_size])

    def test_splits_moe_shared_experts(self):
        """Test that MoE shared experts are split correctly"""
        from mappers import map_monolith_to_tp

        tp_size = 2
        hidden_size = 64
        intermediate_size = hidden_size * 2
        checkpoint = self._create_monolith_moe_checkpoint(hidden_size=hidden_size)

        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=tp_size,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(checkpoint, args=args)

        for shard in result:
            # shared_w1 is column parallel - split on dim 0
            shared_w1 = shard["state"]["model"][
                "model.layers.0.block_sparse_moe.shared_w1"
            ]
            assert shared_w1.shape == torch.Size(
                [intermediate_size // tp_size, hidden_size]
            )

            # shared_w2 is row parallel - split on dim 1
            shared_w2 = shard["state"]["model"][
                "model.layers.0.block_sparse_moe.shared_w2"
            ]
            assert shared_w2.shape == torch.Size(
                [hidden_size, intermediate_size // tp_size]
            )

    def test_splits_router_gate(self):
        """Test that router gate is split correctly"""
        from mappers import map_monolith_to_tp

        tp_size = 2
        hidden_size = 64
        num_experts = 4
        checkpoint = self._create_monolith_moe_checkpoint(
            hidden_size=hidden_size, num_experts=num_experts
        )

        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=tp_size,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(checkpoint, args=args)

        for shard in result:
            # Gate is row parallel - split on dim 1
            gate = shard["state"]["model"]["model.layers.0.block_sparse_moe.gate"]
            assert gate.shape == torch.Size([hidden_size, num_experts // tp_size])

    def test_tp_size_updated_in_config(self):
        """Test that tp_size is updated in config after split"""
        from mappers import map_monolith_to_tp

        tp_size = 4
        checkpoint = self._create_monolith_moe_checkpoint()

        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=tp_size,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(checkpoint, args=args)

        for shard in result:
            config = shard["state"]["integrations"]["huggingface"]["model"]["config"][
                "content"
            ]
            assert config["tp_size"] == tp_size


class TestIsSubstrOfStrings:
    """Tests for is_substr_of_strings helper"""

    def test_finds_substring(self):
        """Test that substring is found in list"""
        from mappers import is_substr_of_strings

        assert is_substr_of_strings(
            "model.layers.0.self_attn.q_proj.weight", ["q_proj", "k_proj"]
        )
        assert is_substr_of_strings("block_sparse_moe.gate.weight", ["gate"])

    def test_returns_false_when_not_found(self):
        """Test that False is returned when no substring matches"""
        from mappers import is_substr_of_strings

        assert not is_substr_of_strings(
            "model.layers.0.mlp.weight", ["q_proj", "k_proj"]
        )


class TestIsKeyContainedInModel:
    """Tests for is_key_contained_in_model helper"""

    def test_finds_key_in_model(self):
        """Test finding key substring in model state dict"""
        from mappers import is_key_contained_in_model

        state_dict = {
            "state": {
                "model": {
                    "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
                    "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 64),
                }
            }
        }

        assert is_key_contained_in_model(state_dict, "q_proj")
        assert is_key_contained_in_model(state_dict, "self_attn")
        assert not is_key_contained_in_model(state_dict, "qkv_proj")


# ============================================================================
# Additional Buffer Tests
# ============================================================================


class TestProcessLayer:
    """Tests for process_layer function"""

    def test_process_single_param_layer(self):
        """Test processing layer with single parameter"""
        from mapper_gigafsdp_to_monolith import process_layer

        # 2 FSDP ranks, each with shard_size=8 elements
        flatten_tensors = [
            torch.arange(8, dtype=torch.float32),  # rank 0
            torch.arange(8, 16, dtype=torch.float32),  # rank 1
        ]

        args = (
            8,  # shard_size - each rank contributes 8 elements
            [[4, 4]],  # param_shapes - final 4x4 tensor
            ["layers.0.weight"],  # param_names
            0,  # offset
        )

        result = process_layer(args, flatten_tensors)

        assert "model.layers.0.weight" in result
        assert result["model.layers.0.weight"].shape == torch.Size([4, 4])

    def test_process_layer_with_suffix(self):
        """Test processing layer with EMA suffix"""
        from mapper_gigafsdp_to_monolith import process_layer

        # Each rank has 8 elements
        flatten_tensors = [
            torch.arange(8, dtype=torch.float32),
            torch.arange(8, 16, dtype=torch.float32),
        ]

        args = (8, [[4, 4]], ["layers.0.weight"], 0)

        result = process_layer(args, flatten_tensors, param_name_suffix="_ema")

        assert "model.layers.0.weight_ema" in result
        assert "model.layers.0.weight" not in result

    def test_process_layer_with_multiple_params(self):
        """Test processing layer with weight and bias"""
        from mapper_gigafsdp_to_monolith import process_layer

        # Total: 16 (weight) + 4 (bias) = 20 elements
        # Each rank has 20 elements
        flatten_tensors = [
            torch.arange(20, dtype=torch.float32),
            torch.arange(20, 40, dtype=torch.float32),
        ]

        args = (
            20,  # shard_size
            [[4, 4], [4]],  # weight 4x4, bias 4
            ["layers.0.proj.weight", "layers.0.proj.bias"],
            0,
        )

        result = process_layer(args, flatten_tensors)

        assert "model.layers.0.proj.weight" in result
        assert "model.layers.0.proj.bias" in result
        assert result["model.layers.0.proj.weight"].shape == torch.Size([4, 4])
        assert result["model.layers.0.proj.bias"].shape == torch.Size([4])

    def test_process_layer_with_offset(self):
        """Test processing layer at non-zero offset"""
        from mapper_gigafsdp_to_monolith import process_layer

        # Each rank has 16 elements total, we start at offset 8
        flatten_tensors = [
            torch.arange(16, dtype=torch.float32),
            torch.arange(16, 32, dtype=torch.float32),
        ]

        args = (8, [[4, 4]], ["layers.1.weight"], 8)  # offset=8, shard_size=8

        result = process_layer(args, flatten_tensors)

        assert "model.layers.1.weight" in result
        # Values from offset 8: rank0[8:16] + rank1[8:16] = [8..15, 24..31]
        assert result["model.layers.1.weight"][0, 0].item() == 8.0


class TestCleanParamName:
    """Tests for clean_param_name function"""

    def test_returns_name_unchanged(self):
        """Test that clean_param_name returns name as-is (current implementation)"""
        from mapper_gigafsdp_to_monolith import clean_param_name

        # Current implementation just returns the name unchanged
        assert clean_param_name("layers.0.weight") == "layers.0.weight"
        assert clean_param_name("embed_tokens.weight") == "embed_tokens.weight"
        assert (
            clean_param_name("_fsdp_wrapped_module.layers.0.weight")
            == "_fsdp_wrapped_module.layers.0.weight"
        )


class TestBufferPatternMatching:
    """Additional tests for buffer pattern matching"""

    def test_multiple_buffer_patterns(self):
        """Test various buffer patterns"""
        from mapper_gigafsdp_to_monolith import check_buffer_params

        # expert_bias pattern
        tensor = torch.randn(8)
        name, val = check_buffer_params(
            [tensor, tensor.clone()], "model.layers.0._gate.expert_bias"
        )
        assert name == "model.layers.0._gate.expert_bias"

    def test_buffer_must_start_with_model(self):
        """Test that buffer must start with 'model.'"""
        from mapper_gigafsdp_to_monolith import check_buffer_params

        tensor = torch.randn(8)
        # Doesn't start with 'model.'
        name, val = check_buffer_params([tensor], "layers.0._gate.expert_bias")
        assert name is None

    def test_buffer_must_be_tensor(self):
        """Test that buffer value must be a tensor"""
        from mapper_gigafsdp_to_monolith import check_buffer_params

        # Not a tensor
        name, val = check_buffer_params(
            [{"key": "value"}], "model.layers.0._gate.expert_bias"
        )
        assert name is None


class TestUnshardWithMixedTpSizes:
    """Tests for unsharding with different TP sizes in meta"""

    def test_unshard_with_tp_size_in_pg_size(self):
        """Test unsharding when meta has parameter_pg_size"""
        from mapper_gigafsdp_to_monolith import unshard_flat_tensor

        # shard_size = 16, each rank has 16 elements
        meta = {
            "main_parameter_shard_bounaries": [16],
            "main_parameter_orig_shapes": [[[4, 4]]],
            "main_shards_fqns": [["layers.0.weight"]],
            "main_parameter_pg_size": [(2, 1)],  # (dp_size, tp_size)
            "fsdp_group_size": 2,
        }

        flatten_tensors = [
            torch.arange(16, dtype=torch.float32),
            torch.arange(16, 32, dtype=torch.float32),
        ]

        result = unshard_flat_tensor(meta, flatten_tensors, param_group="main")

        assert "model.layers.0.weight" in result


# ============================================================================
# Additional MoE Expert Tests
# ============================================================================


class TestMoEExpertsWeight1Hotfix:
    """Tests for the experts.weight1 special handling in map_tp_to_monolith"""

    def _create_tp_checkpoints_with_experts(
        self, tp_size=2, num_experts=8, hidden_size=64
    ):
        """Create TP checkpoints with expert weights"""
        intermediate_size = hidden_size * 2
        num_local_experts = (
            num_experts  # All experts on each TP rank before hotfix reshape
        )

        checkpoints = []
        for tp_rank in range(tp_size):
            # experts.weight1 before hotfix: [hidden_size // tp_size, num_experts * intermediate_size]
            # After view: [num_local_experts, hidden_size, intermediate_size // tp_size]
            experts_weight1 = torch.randn(
                hidden_size // tp_size, num_experts * intermediate_size
            )

            ckpt = {
                "rng": {"state": torch.get_rng_state()},
                "state": {
                    "model": OrderedDict(
                        [
                            (
                                "model.layers.0.self_attn.q_proj.weight",
                                torch.randn(hidden_size // tp_size, hidden_size),
                            ),
                            (
                                "model.layers.0.block_sparse_moe.experts.weight1",
                                experts_weight1,
                            ),
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
                                        "intermediate_size": intermediate_size,
                                        "num_attention_heads": 4,
                                        "num_key_value_heads": 4,
                                        "num_routed_experts": num_experts,
                                        "model_type": "giga_mix",
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

    def test_experts_weight1_reshape(self):
        """Test that experts.weight1 is correctly reshaped during merge"""
        from mappers import map_tp_to_monolith

        tp_size = 2
        hidden_size = 64
        num_experts = 8
        intermediate_size = hidden_size * 2

        checkpoints = self._create_tp_checkpoints_with_experts(
            tp_size=tp_size, num_experts=num_experts, hidden_size=hidden_size
        )

        args = argparse.Namespace(skip_optimizer=True, input_tp_size=tp_size)

        result = map_tp_to_monolith(checkpoints, args=args)

        weight1 = result["state"]["model"][
            "model.layers.0.block_sparse_moe.experts.weight1"
        ]
        # After hotfix reshape: [hidden_size, num_experts * intermediate_size]
        assert weight1.shape[0] == hidden_size


class TestMoEExpertsWeight2:
    """Tests for experts.weight2 handling"""

    def _create_tp_checkpoints_with_experts2(
        self, tp_size=2, num_experts=4, hidden_size=64
    ):
        """Create TP checkpoints with experts.weight2"""
        intermediate_size = hidden_size * 2

        checkpoints = []
        for tp_rank in range(tp_size):
            # experts.weight2 is column parallel, sharded on dim 0
            experts_weight2 = torch.randn(intermediate_size // tp_size, hidden_size)

            ckpt = {
                "rng": {"state": torch.get_rng_state()},
                "state": {
                    "model": OrderedDict(
                        [
                            (
                                "model.layers.0.self_attn.q_proj.weight",
                                torch.randn(hidden_size // tp_size, hidden_size),
                            ),
                            (
                                "model.layers.0.block_sparse_moe.experts.weight2",
                                experts_weight2,
                            ),
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
                                        "intermediate_size": intermediate_size,
                                        "num_attention_heads": 4,
                                        "num_key_value_heads": 4,
                                        "num_routed_experts": num_experts,
                                        "model_type": "giga_mix",
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

    def test_experts_weight2_merge(self):
        """Test that experts.weight2 is merged correctly"""
        from mappers import map_tp_to_monolith

        tp_size = 2
        hidden_size = 64
        intermediate_size = hidden_size * 2

        checkpoints = self._create_tp_checkpoints_with_experts2(
            tp_size=tp_size, hidden_size=hidden_size
        )

        args = argparse.Namespace(skip_optimizer=True, input_tp_size=tp_size)

        result = map_tp_to_monolith(checkpoints, args=args)

        weight2 = result["state"]["model"][
            "model.layers.0.block_sparse_moe.experts.weight2"
        ]
        # Column parallel merged on dim 0
        assert weight2.shape == torch.Size([intermediate_size, hidden_size])


class TestMoEW1W2W3:
    """Tests for block_sparse_moe.w1, w2, w3 handling"""

    def _create_tp_checkpoints_with_w123(self, tp_size=2, hidden_size=64):
        """Create TP checkpoints with w1, w2, w3"""
        intermediate_size = hidden_size * 2

        checkpoints = []
        for tp_rank in range(tp_size):
            ckpt = {
                "rng": {"state": torch.get_rng_state()},
                "state": {
                    "model": OrderedDict(
                        [
                            (
                                "model.layers.0.self_attn.q_proj.weight",
                                torch.randn(hidden_size // tp_size, hidden_size),
                            ),
                            # w1 is row parallel
                            (
                                "model.layers.0.block_sparse_moe.w1",
                                torch.randn(hidden_size, intermediate_size // tp_size),
                            ),
                            # w2 is column parallel
                            (
                                "model.layers.0.block_sparse_moe.w2",
                                torch.randn(intermediate_size // tp_size, hidden_size),
                            ),
                            # w3 is row parallel
                            (
                                "model.layers.0.block_sparse_moe.w3",
                                torch.randn(hidden_size, intermediate_size // tp_size),
                            ),
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
                                        "intermediate_size": intermediate_size,
                                        "num_attention_heads": 4,
                                        "num_key_value_heads": 4,
                                        "model_type": "giga_mix",
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

    def test_w1_row_parallel_merge(self):
        """Test that w1 is merged as row parallel (dim 1)"""
        from mappers import map_tp_to_monolith

        tp_size = 2
        hidden_size = 64
        intermediate_size = hidden_size * 2

        checkpoints = self._create_tp_checkpoints_with_w123(
            tp_size=tp_size, hidden_size=hidden_size
        )

        args = argparse.Namespace(skip_optimizer=True, input_tp_size=tp_size)

        result = map_tp_to_monolith(checkpoints, args=args)

        w1 = result["state"]["model"]["model.layers.0.block_sparse_moe.w1"]
        assert w1.shape == torch.Size([hidden_size, intermediate_size])

    def test_w2_column_parallel_merge(self):
        """Test that w2 is merged as column parallel (dim 0)"""
        from mappers import map_tp_to_monolith

        tp_size = 2
        hidden_size = 64
        intermediate_size = hidden_size * 2

        checkpoints = self._create_tp_checkpoints_with_w123(
            tp_size=tp_size, hidden_size=hidden_size
        )

        args = argparse.Namespace(skip_optimizer=True, input_tp_size=tp_size)

        result = map_tp_to_monolith(checkpoints, args=args)

        w2 = result["state"]["model"]["model.layers.0.block_sparse_moe.w2"]
        assert w2.shape == torch.Size([intermediate_size, hidden_size])

    def test_w3_row_parallel_merge(self):
        """Test that w3 is merged as row parallel (dim 1)"""
        from mappers import map_tp_to_monolith

        tp_size = 2
        hidden_size = 64
        intermediate_size = hidden_size * 2

        checkpoints = self._create_tp_checkpoints_with_w123(
            tp_size=tp_size, hidden_size=hidden_size
        )

        args = argparse.Namespace(skip_optimizer=True, input_tp_size=tp_size)

        result = map_tp_to_monolith(checkpoints, args=args)

        w3 = result["state"]["model"]["model.layers.0.block_sparse_moe.w3"]
        assert w3.shape == torch.Size([hidden_size, intermediate_size])


class TestMoEConfigFields:
    """Tests for MoE config field handling"""

    def test_num_routed_experts_preserved(self):
        """Test that num_routed_experts is preserved in config"""
        from mappers import map_tp_to_monolith

        num_experts = 8
        checkpoints = [
            {
                "rng": {"state": torch.get_rng_state()},
                "state": {
                    "model": OrderedDict(
                        [
                            (
                                "model.layers.0.self_attn.q_proj.weight",
                                torch.randn(32, 64),
                            ),
                        ]
                    ),
                    "integrations": {
                        "huggingface": {
                            "model": {
                                "config": {
                                    "content": {
                                        "hidden_size": 64,
                                        "num_attention_heads": 4,
                                        "num_key_value_heads": 4,
                                        "num_routed_experts": num_experts,
                                        "model_type": "giga_mix",
                                        "tp_size": 1,
                                    }
                                }
                            }
                        }
                    },
                },
            }
        ]

        args = argparse.Namespace(skip_optimizer=True, input_tp_size=1)

        result = map_tp_to_monolith(checkpoints, args=args)

        config = result["state"]["integrations"]["huggingface"]["model"]["config"][
            "content"
        ]
        assert config["num_routed_experts"] == num_experts


class TestMoEMonolithSplit:
    """Additional tests for map_monolith_to_tp with MoE"""

    def _create_full_moe_checkpoint(self, num_experts=8, hidden_size=64):
        """Create full MoE checkpoint with all expert types"""
        intermediate_size = hidden_size * 2

        return {
            "rng": {"state": torch.get_rng_state()},
            "state": {
                "model": OrderedDict(
                    [
                        # Attention
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
                        # Routed experts
                        (
                            "model.layers.0.block_sparse_moe.experts.weight1",
                            torch.randn(hidden_size, num_experts * intermediate_size),
                        ),
                        (
                            "model.layers.0.block_sparse_moe.experts.weight2",
                            torch.randn(num_experts * intermediate_size, hidden_size),
                        ),
                        # Non-shared MoE weights
                        (
                            "model.layers.0.block_sparse_moe.w1",
                            torch.randn(hidden_size, intermediate_size),
                        ),
                        (
                            "model.layers.0.block_sparse_moe.w2",
                            torch.randn(intermediate_size, hidden_size),
                        ),
                        (
                            "model.layers.0.block_sparse_moe.w3",
                            torch.randn(hidden_size, intermediate_size),
                        ),
                        # Shared experts
                        (
                            "model.layers.0.block_sparse_moe.shared_w1",
                            torch.randn(intermediate_size, hidden_size),
                        ),
                        (
                            "model.layers.0.block_sparse_moe.shared_w2",
                            torch.randn(hidden_size, intermediate_size),
                        ),
                        (
                            "model.layers.0.block_sparse_moe.shared_w3",
                            torch.randn(intermediate_size, hidden_size),
                        ),
                        # Router
                        (
                            "model.layers.0.block_sparse_moe.gate",
                            torch.randn(hidden_size, num_experts),
                        ),
                        # LayerNorms
                        (
                            "model.layers.0.input_layernorm.weight",
                            torch.randn(hidden_size),
                        ),
                        (
                            "model.layers.0.post_attention_layernorm.weight",
                            torch.randn(hidden_size),
                        ),
                        # Embeddings
                        ("model.embed_tokens.weight", torch.randn(1000, hidden_size)),
                        ("model.lm_head.weight", torch.randn(1000, hidden_size)),
                    ]
                ),
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "hidden_size": hidden_size,
                                    "intermediate_size": intermediate_size,
                                    "num_attention_heads": 4,
                                    "num_key_value_heads": 4,
                                    "num_routed_experts": num_experts,
                                    "model_type": "giga_mix",
                                    "tp_size": 1,
                                }
                            }
                        }
                    }
                },
            },
        }

    def test_split_w1_w3_row_parallel(self):
        """Test w1 and w3 are split as row parallel"""
        from mappers import map_monolith_to_tp

        tp_size = 2
        hidden_size = 64
        intermediate_size = hidden_size * 2

        checkpoint = self._create_full_moe_checkpoint(hidden_size=hidden_size)

        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=tp_size,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(checkpoint, args=args)

        for shard in result:
            # w1 row parallel - split on dim 1
            w1 = shard["state"]["model"]["model.layers.0.block_sparse_moe.w1"]
            assert w1.shape == torch.Size([hidden_size, intermediate_size // tp_size])

            # w3 row parallel - split on dim 1
            w3 = shard["state"]["model"]["model.layers.0.block_sparse_moe.w3"]
            assert w3.shape == torch.Size([hidden_size, intermediate_size // tp_size])

    def test_split_w2_column_parallel(self):
        """Test w2 is split as column parallel"""
        from mappers import map_monolith_to_tp

        tp_size = 2
        hidden_size = 64
        intermediate_size = hidden_size * 2

        checkpoint = self._create_full_moe_checkpoint(hidden_size=hidden_size)

        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=tp_size,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(checkpoint, args=args)

        for shard in result:
            # w2 column parallel - split on dim 0
            w2 = shard["state"]["model"]["model.layers.0.block_sparse_moe.w2"]
            assert w2.shape == torch.Size([intermediate_size // tp_size, hidden_size])

    def test_layernorm_replicated(self):
        """Test that LayerNorm weights are replicated (not split)"""
        from mappers import map_monolith_to_tp

        tp_size = 2
        hidden_size = 64

        checkpoint = self._create_full_moe_checkpoint(hidden_size=hidden_size)
        original_ln = checkpoint["state"]["model"][
            "model.layers.0.input_layernorm.weight"
        ].clone()

        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=tp_size,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(checkpoint, args=args)

        for shard in result:
            ln = shard["state"]["model"]["model.layers.0.input_layernorm.weight"]
            # Should be same size, not split
            assert ln.shape == torch.Size([hidden_size])
            # Should be same values
            assert torch.allclose(ln, original_ln)

    def test_lm_head_split(self):
        """Test that lm_head is split correctly"""
        from mappers import map_monolith_to_tp

        tp_size = 2
        hidden_size = 64
        vocab_size = 1000

        checkpoint = self._create_full_moe_checkpoint(hidden_size=hidden_size)

        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=tp_size,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(checkpoint, args=args)

        for shard in result:
            # lm_head is column parallel - split on dim 0
            lm_head = shard["state"]["model"]["model.lm_head.weight"]
            assert lm_head.shape == torch.Size([vocab_size // tp_size, hidden_size])


class TestGigaOmniMoEConfig:
    """Tests for gigaomni model type config handling"""

    def _create_gigaomni_checkpoint(self, hidden_size=64):
        """Create gigaomni checkpoint with nested config"""
        return {
            "rng": {"state": torch.get_rng_state()},
            "state": {
                "model": OrderedDict(
                    [
                        (
                            "model.layers.0.self_attn.q_proj.weight",
                            torch.randn(hidden_size, hidden_size),
                        ),
                    ]
                ),
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "model_type": "gigaomni",
                                    "language_model": {
                                        "hidden_size": hidden_size,
                                        "num_attention_heads": 4,
                                        "num_key_value_heads": 4,
                                        "tp_size": 1,
                                    },
                                }
                            }
                        }
                    }
                },
            },
        }

    def test_gigaomni_tp_size_update(self):
        """Test that gigaomni nested config tp_size is updated"""
        from mappers import map_monolith_to_tp

        tp_size = 2
        checkpoint = self._create_gigaomni_checkpoint()

        args = argparse.Namespace(
            skip_optimizer=True,
            output_tp_size=tp_size,
            input_tp_size=1,
            output_tp_embedding_type="VocabParallelEmbedding",
        )

        result = map_monolith_to_tp(checkpoint, args=args)

        for shard in result:
            config = shard["state"]["integrations"]["huggingface"]["model"]["config"][
                "content"
            ]
            # gigaomni has nested config
            assert config["language_model"]["tp_size"] == tp_size
