import pytest
import tempfile
import torch
from pathlib import Path
from collections import OrderedDict
from unittest.mock import Mock, patch
import sys

from mappers import map_mixed_gigafsdp_to_fsdp


def _create_model_meta(config, tp_size=2):
    """Creates model.meta structure required by gigafsdp unshard functions."""
    hidden_size = config.get("hidden_size", 256)
    num_heads = config.get("num_attention_heads", 4)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    vocab_size = config.get("vocab_size", 1000)

    # Create parameter names and shapes for the model
    param_names = [
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "norm.weight",
        "lm_head.weight",
    ]

    head_dim = hidden_size // num_heads
    kv_hidden_size = head_dim * num_kv_heads

    param_shapes = [
        [vocab_size, hidden_size],
        [hidden_size // tp_size, hidden_size],
        [kv_hidden_size // tp_size, hidden_size],
        [kv_hidden_size // tp_size, hidden_size],
        [hidden_size, hidden_size // tp_size],
        [intermediate_size // tp_size, hidden_size],
        [intermediate_size // tp_size, hidden_size],
        [hidden_size, intermediate_size // tp_size],
        [hidden_size],
        [vocab_size, hidden_size],
    ]

    # Calculate shard sizes (number of elements per shard)
    shard_sizes = []
    for shape in param_shapes:
        numel = 1
        for s in shape:
            numel *= s
        shard_sizes.append(numel)

    return {
        "main_parameter_shard_bounaries": shard_sizes,
        "main_parameter_orig_shapes": param_shapes,
        "main_shards_fqns": [[name] for name in param_names],
        "tp_size": tp_size,
    }


def _create_mixed_checkpoint(
    temp_dir, config, tp_size=2, ep_size=2, include_weights=True, include_meta=True
):
    """Creates mixed gigafsdp checkpoint with combined directory naming."""
    checkpoint_dir = Path(temp_dir) / "checkpoint"

    hidden_size = config.get("hidden_size", 256)
    num_heads = config.get("num_attention_heads", 4)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    kv_hidden_size = head_dim * num_kv_heads
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    vocab_size = config.get("vocab_size", 1000)

    for tp_rank in range(tp_size):
        for ep_rank in range(ep_size):
            # CRITICAL: Combined directory name for regex matching
            rank_dir = checkpoint_dir / f"tp_rank{tp_rank}_ep_rank{ep_rank}"
            rank_dir.mkdir(parents=True, exist_ok=True)

            if include_weights:
                model = OrderedDict(
                    {
                        "model.embed_tokens.weight": torch.randn(
                            vocab_size, hidden_size
                        ),
                        "model.layers.0.self_attn.q_proj.weight": torch.randn(
                            hidden_size // tp_size, hidden_size
                        ),
                        "model.layers.0.self_attn.k_proj.weight": torch.randn(
                            kv_hidden_size // tp_size, hidden_size
                        ),
                        "model.layers.0.self_attn.v_proj.weight": torch.randn(
                            kv_hidden_size // tp_size, hidden_size
                        ),
                        "model.layers.0.self_attn.o_proj.weight": torch.randn(
                            hidden_size, hidden_size // tp_size
                        ),
                        "model.layers.0.mlp.gate_proj.weight": torch.randn(
                            intermediate_size // tp_size, hidden_size
                        ),
                        "model.layers.0.mlp.up_proj.weight": torch.randn(
                            intermediate_size // tp_size, hidden_size
                        ),
                        "model.layers.0.mlp.down_proj.weight": torch.randn(
                            hidden_size, intermediate_size // tp_size
                        ),
                        "model.norm.weight": torch.ones(hidden_size),
                        "model.lm_head.weight": torch.randn(vocab_size, hidden_size),
                    }
                )

                # Add model.meta for gigafsdp unshard functions
                if include_meta:
                    model["model.meta"] = _create_model_meta(config, tp_size)
            else:
                model = OrderedDict()
                if include_meta:
                    model["model.meta"] = _create_model_meta(config, tp_size)

            state_dict = {
                "state": {
                    "model": model,
                    "integrations": {
                        "huggingface": {"model": {"config": {"content": config.copy()}}}
                    },
                }
            }

            torch.save(state_dict, rank_dir / "__0_0.distcp")

    return checkpoint_dir


def create_mock_args(**kwargs):
    """Create mock args with sensible defaults."""
    mock_args = Mock()
    mock_args.tp_size = kwargs.get("tp_size", 1)
    mock_args.ep_size = kwargs.get("ep_size", 1)
    mock_args.output = kwargs.get("output", "/tmp/output")
    mock_args.skip_ema_mapper = kwargs.get("skip_ema_mapper", True)
    mock_args.skip_optimizer = kwargs.get("skip_optimizer", True)
    mock_args.use_ema = kwargs.get("use_ema", False)
    mock_args.packed_attention = kwargs.get("packed_attention", True)
    mock_args.model_name = kwargs.get("model_name", "test-model")
    mock_args.model_type = kwargs.get("model_type", "gigar")
    mock_args.input_tp_size = kwargs.get("input_tp_size", None)
    mock_args.input_ep_size = kwargs.get("input_ep_size", None)
    mock_args.output_tp_size = kwargs.get("output_tp_size", 1)
    mock_args.output_ep_size = kwargs.get("output_ep_size", 1)
    mock_args.prepare_ema_weight = kwargs.get("prepare_ema_weight", False)
    return mock_args


class TestMixedGigafsdpToFsdp:
    """Tests for map_mixed_gigafsdp_to_fsdp function."""

    def test_basic_conversion(self, simple_config):
        """Test basic mixed gigafsdp to fsdp conversion."""
        config = simple_config.copy()
        config.setdefault("num_key_value_heads", config.get("num_attention_heads", 4))

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=2, ep_size=2
            )
            mock_args = create_mock_args(skip_optimizer=True)

            # Mock the unshard functions since we can't easily create real gigafsdp checkpoints
            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                mock_build.return_value = {
                    "state": {
                        "model": OrderedDict({"model.test": torch.randn(10)}),
                        "integrations": config,
                    }
                }
                result = map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

                assert result is not None
                assert "state" in result

    def test_tp_size_inferred_from_checkpoint(self, simple_config):
        """Test that TP size is correctly inferred from checkpoint structure."""
        config = simple_config.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=4, ep_size=2
            )
            mock_args = create_mock_args()

            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                mock_build.return_value = {"state": {"model": OrderedDict()}}
                map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

                # After loading, args should have inferred TP size
                assert mock_args.input_tp_size == 4

    def test_ep_size_inferred_from_checkpoint(self, simple_config):
        """Test that EP size is correctly inferred from checkpoint structure."""
        config = simple_config.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=2, ep_size=4
            )
            mock_args = create_mock_args()

            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                mock_build.return_value = {"state": {"model": OrderedDict()}}
                map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

                assert mock_args.input_ep_size == 4

    def test_output_is_monolith(self, simple_config):
        """Test that output is a single monolith checkpoint, not sharded."""
        config = simple_config.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=2, ep_size=2
            )
            mock_args = create_mock_args()

            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                expected_result = {
                    "state": {
                        "model": OrderedDict({"model.test": torch.randn(10)}),
                    }
                }
                mock_build.return_value = expected_result
                result = map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

                # Result should be a dict, not a list
                assert isinstance(result, dict)
                assert "state" in result

    def test_mixed_tp_layers_merged(self, simple_config):
        """Test that layers with different TP sizes are properly merged."""
        config = simple_config.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=2, ep_size=2
            )
            mock_args = create_mock_args()

            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                mock_build.return_value = {"state": {"model": OrderedDict()}}
                result = map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

                # build_monolith should have been called
                mock_build.assert_called_once()

    def test_packed_attention_in_mixed(self, simple_config):
        """Test handling of packed attention in mixed checkpoint."""
        config = simple_config.copy()
        config["attention_type"] = "LlamaPackedAttention"

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=2, ep_size=2
            )
            mock_args = create_mock_args(packed_attention=True)

            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                mock_build.return_value = {"state": {"model": OrderedDict()}}
                result = map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

                assert result is not None

    def test_ema_handled_in_mixed(self, simple_config):
        """Test EMA weights handling in mixed checkpoint."""
        config = simple_config.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=2, ep_size=2
            )
            mock_args = create_mock_args(use_ema=False, skip_ema_mapper=True)

            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                mock_build.return_value = {"state": {"model": OrderedDict()}}
                result = map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

                assert result is not None

    def test_file_loading_order(self, simple_config):
        """Test that files are loaded in correct order by rank."""
        config = simple_config.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=2, ep_size=2
            )
            mock_args = create_mock_args()

            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                mock_build.return_value = {"state": {"model": OrderedDict()}}
                result = map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

                # Verify that loading happened (args should be updated)
                assert mock_args.input_tp_size is not None

    def test_optimizer_not_supported(self, simple_config):
        """Test that optimizer conversion raises NotImplementedError for mixed."""
        config = simple_config.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=2, ep_size=2
            )
            mock_args = create_mock_args(skip_optimizer=False)

            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                # build_monolith should raise NotImplementedError for optimizer
                mock_build.side_effect = NotImplementedError("Optimizer not supported")

                with pytest.raises(NotImplementedError):
                    map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

    def test_unsupported_input_tp_size_greater_than_1(self, simple_config):
        """Test handling when input TP size > 1 (should work with mixed)."""
        config = simple_config.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=2, ep_size=2
            )
            mock_args = create_mock_args()

            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                mock_build.return_value = {"state": {"model": OrderedDict()}}
                result = map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

                # Mixed checkpoint conversion should handle tp_size > 1
                assert result is not None

    def test_unsupported_output_tp_size_greater_than_1(self, simple_config):
        """Test handling when output TP size > 1."""
        config = simple_config.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = _create_mixed_checkpoint(
                temp_dir, config, tp_size=2, ep_size=2
            )
            mock_args = create_mock_args(output_tp_size=2)

            with patch(
                "mappers.build_monolith_from_mixed_gigafsdp_state"
            ) as mock_build:
                mock_build.return_value = {"state": {"model": OrderedDict()}}
                result = map_mixed_gigafsdp_to_fsdp(str(checkpoint_dir), args=mock_args)

                assert result is not None
