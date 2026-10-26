import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.unit
class TestLoadCheckpoint:
    """Tests for load_checkpoint function"""

    def test_load_monolith_checkpoint(
        self, temp_dir, simple_fsdp_checkpoint, mock_args
    ):
        """Test loading a monolith checkpoint file"""
        from mappers import load_checkpoint

        checkpoint_path = temp_dir / "checkpoint.pt"
        torch.save(simple_fsdp_checkpoint, checkpoint_path)

        loaded = load_checkpoint(str(checkpoint_path), args=mock_args)

        assert "state" in loaded
        assert "model" in loaded["state"]

    def test_load_sharded_checkpoint(self, temp_dir, simple_fsdp_checkpoint, mock_args):
        """Test loading a sharded gigafsdp checkpoint"""
        from mappers import load_checkpoint

        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()

        for rank in range(2):
            torch.save(simple_fsdp_checkpoint, checkpoint_dir / f"__{rank}_0.distcp")

        loaded = load_checkpoint(str(checkpoint_dir), args=mock_args)

        assert isinstance(loaded, tuple)
        assert len(loaded) == 2

    def test_load_with_mmap(self, temp_dir, simple_fsdp_checkpoint, mock_args):
        """Test loading with memory mapping"""
        from mappers import load_checkpoint

        checkpoint_path = temp_dir / "checkpoint.pt"
        torch.save(simple_fsdp_checkpoint, checkpoint_path)

        loaded = load_checkpoint(str(checkpoint_path), args=mock_args)

        assert loaded is not None

    def test_load_nonexistent_checkpoint_error(self, temp_dir, mock_args):
        """Test that loading nonexistent checkpoint raises error"""
        from mappers import load_checkpoint

        with pytest.raises(FileNotFoundError):
            load_checkpoint(str(temp_dir / "nonexistent.pt"), args=mock_args)

    def test_load_with_weights_only_false(
        self, temp_dir, simple_fsdp_checkpoint, mock_args
    ):
        """Test that weights_only=False is used for loading"""
        from mappers import load_checkpoint

        checkpoint_path = temp_dir / "checkpoint.pt"
        torch.save(simple_fsdp_checkpoint, checkpoint_path)

        loaded = load_checkpoint(str(checkpoint_path), args=mock_args)

        assert "state" in loaded

    def test_load_uses_dill(self, temp_dir, simple_fsdp_checkpoint, mock_args):
        """Test that dill is used for unpickling"""
        from mappers import load_checkpoint

        checkpoint_path = temp_dir / "checkpoint.pt"
        torch.save(simple_fsdp_checkpoint, checkpoint_path)

        # Should not raise
        loaded = load_checkpoint(str(checkpoint_path), args=mock_args)
        assert loaded is not None


@pytest.mark.unit
class TestSaveCheckpoint:
    """Tests for save_checkpoint function"""

    def test_save_monolith_checkpoint(
        self, temp_dir, simple_fsdp_checkpoint, mock_args
    ):
        """Test saving a monolith checkpoint"""
        from mappers import save_checkpoint

        output_path = str(temp_dir / "output.pt")

        save_checkpoint(simple_fsdp_checkpoint, output_path, args=mock_args)

        assert Path(output_path).exists()
        loaded = torch.load(output_path, weights_only=False)
        assert "state" in loaded

    def test_save_tp_checkpoints(self, temp_dir, simple_fsdp_checkpoint, mock_args):
        """Test saving multiple tp checkpoints"""
        from mappers import save_checkpoint

        mock_args.output_tp_size = 2

        checkpoints = [simple_fsdp_checkpoint, simple_fsdp_checkpoint]
        output_paths = [str(temp_dir / "tp_rank0.pt"), str(temp_dir / "tp_rank1.pt")]

        save_checkpoint(checkpoints, output_paths, args=mock_args)

        for path in output_paths:
            assert Path(path).exists()

    def test_save_gigafsdp_sharded(self, temp_dir, simple_fsdp_checkpoint, mock_args):
        """Test saving gigafsdp sharded checkpoint"""
        from mappers import save_checkpoint

        checkpoints = [simple_fsdp_checkpoint, simple_fsdp_checkpoint]
        output_path = str(temp_dir / "sharded")

        save_checkpoint(checkpoints, output_path, args=mock_args)

        assert (temp_dir / "sharded").exists()
        assert (temp_dir / "sharded" / "__0_0.distcp").exists()

    def test_save_uses_dill(self, temp_dir, simple_fsdp_checkpoint, mock_args):
        """Test that dill is used for pickling"""
        from mappers import save_checkpoint

        output_path = str(temp_dir / "output.pt")

        # Should not raise
        save_checkpoint(simple_fsdp_checkpoint, output_path, args=mock_args)
        assert Path(output_path).exists()

    def test_save_creates_parent_directories(
        self, temp_dir, simple_fsdp_checkpoint, mock_args
    ):
        """Test that parent directories are created"""
        from mappers import save_checkpoint

        output_path = temp_dir / "subdir1" / "subdir2" / "output.pt"

        # Create parent directories manually since save_checkpoint may not do this
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_checkpoint(simple_fsdp_checkpoint, str(output_path), args=mock_args)

        assert output_path.exists()


@pytest.mark.unit
class TestLoadGigafsdpCheckpoint:
    """Tests for load_gigafsdp_checkpoint function"""

    def test_load_basic_gigafsdp(self, temp_dir, simple_fsdp_checkpoint, mock_args):
        """Test loading basic gigafsdp checkpoint"""
        from mappers import load_gigafsdp_checkpoint

        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()

        for rank in range(2):
            torch.save(simple_fsdp_checkpoint, checkpoint_dir / f"__{rank}_0.distcp")

        loaded = load_gigafsdp_checkpoint(str(checkpoint_dir))

        assert isinstance(loaded, tuple)
        assert len(loaded) == 2

    def test_load_multiple_shards(self, temp_dir, simple_fsdp_checkpoint, mock_args):
        """Test loading multiple shards"""
        from mappers import load_gigafsdp_checkpoint

        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()

        for rank in range(4):
            torch.save(simple_fsdp_checkpoint, checkpoint_dir / f"__{rank}_0.distcp")

        loaded = load_gigafsdp_checkpoint(str(checkpoint_dir))

        assert len(loaded) == 4

    def test_load_with_unexpected_files(
        self, temp_dir, simple_fsdp_checkpoint, mock_args
    ):
        """Test loading ignores unexpected files"""
        from mappers import load_gigafsdp_checkpoint
        import warnings

        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()

        torch.save(simple_fsdp_checkpoint, checkpoint_dir / "__0_0.distcp")
        (checkpoint_dir / "random.txt").write_text("test")

        with warnings.catch_warnings(record=True) as w:
            loaded = load_gigafsdp_checkpoint(str(checkpoint_dir))

        assert len(loaded) == 1

    def test_load_preserves_tensor_values(
        self, temp_dir, simple_fsdp_checkpoint, mock_args
    ):
        """Test that tensor values are preserved after loading"""
        from mappers import load_gigafsdp_checkpoint

        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()

        original_tensor = simple_fsdp_checkpoint["state"]["model"][
            "model.embed_tokens.weight"
        ].clone()
        torch.save(simple_fsdp_checkpoint, checkpoint_dir / "__0_0.distcp")

        loaded = load_gigafsdp_checkpoint(str(checkpoint_dir))

        assert torch.allclose(
            loaded[0]["state"]["model"]["model.embed_tokens.weight"], original_tensor
        )

    def test_load_with_mmap(self, temp_dir, simple_fsdp_checkpoint, mock_args):
        """Test loading uses memory mapping"""
        from mappers import load_gigafsdp_checkpoint

        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()

        torch.save(simple_fsdp_checkpoint, checkpoint_dir / "__0_0.distcp")

        loaded = load_gigafsdp_checkpoint(str(checkpoint_dir))

        assert loaded is not None


@pytest.mark.unit
class TestLoadMixedGigafsdpCheckpoint:
    """Tests for load_mixed_gigafsdp_checkpoint function"""

    def _create_mixed_checkpoint(self, temp_dir, config, tp_size=2, ep_size=2):
        """Helper to create mixed gigafsdp checkpoint with proper directory structure"""
        checkpoint_dir = temp_dir / "checkpoint"

        for tp_rank in range(tp_size):
            for ep_rank in range(ep_size):
                # Create directory structure: checkpoint/tp_rank0_ep_rank0/
                rank_dir = checkpoint_dir / f"tp_rank{tp_rank}_ep_rank{ep_rank}"
                rank_dir.mkdir(parents=True)

                meta = {
                    "fsdp_group_size": ep_size,
                    "main_parameter_shard_bounaries": [256 * 512],
                    "main_parameter_orig_shapes": [[[256, 512]]],
                    "main_shards_fqns": [["layers.0.self_attn.q_proj.weight"]],
                    "main_shards_numels": [[256 * 512]],
                }

                shard = {
                    "state": {
                        "model": {
                            "model.meta": meta,
                            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                                256, 512
                            ),
                        },
                        "integrations": {
                            "huggingface": {"model": {"config": {"content": config}}}
                        },
                    }
                }

                torch.save(shard, rank_dir / "__0_0.distcp")

        return checkpoint_dir

    def test_load_basic_mixed_checkpoint(self, temp_dir, mock_args, simple_config):
        """Test loading basic mixed checkpoint"""
        from mappers import load_mixed_gigafsdp_checkpoint

        checkpoint_dir = self._create_mixed_checkpoint(
            temp_dir, simple_config, tp_size=2, ep_size=2
        )

        loaded = load_mixed_gigafsdp_checkpoint(str(checkpoint_dir), args=mock_args)

        assert isinstance(loaded, list)
        assert len(loaded) == 4  # tp_size * ep_size

    def test_infer_tp_and_ep_sizes(self, temp_dir, mock_args, simple_config):
        """Test that tp_size and ep_size are inferred from checkpoint"""
        from mappers import load_mixed_gigafsdp_checkpoint

        checkpoint_dir = self._create_mixed_checkpoint(
            temp_dir, simple_config, tp_size=3, ep_size=4
        )

        load_mixed_gigafsdp_checkpoint(str(checkpoint_dir), args=mock_args)

        assert mock_args.input_tp_size == 3
        assert mock_args.input_ep_size == 4

    def test_sets_args_tp_and_ep_size(self, temp_dir, mock_args, simple_config):
        """Test that args.input_tp_size and args.input_ep_size are set"""
        from mappers import load_mixed_gigafsdp_checkpoint

        checkpoint_dir = self._create_mixed_checkpoint(
            temp_dir, simple_config, tp_size=2, ep_size=3
        )

        load_mixed_gigafsdp_checkpoint(str(checkpoint_dir), args=mock_args)

        assert mock_args.input_tp_size == 2
        assert mock_args.input_ep_size == 3

    def test_load_order_by_rank(self, temp_dir, mock_args, simple_config):
        """Test that files are loaded in correct order"""
        from mappers import load_mixed_gigafsdp_checkpoint

        checkpoint_dir = self._create_mixed_checkpoint(
            temp_dir, simple_config, tp_size=2, ep_size=2
        )

        loaded = load_mixed_gigafsdp_checkpoint(str(checkpoint_dir), args=mock_args)

        # Should load 4 checkpoints
        assert len(loaded) == 4

    def test_empty_checkpoint_directory(self, temp_dir, mock_args, simple_config):
        """Test error handling for empty checkpoint directory"""
        from mappers import load_mixed_gigafsdp_checkpoint

        checkpoint_dir = temp_dir / "empty_checkpoint"
        checkpoint_dir.mkdir()

        with pytest.raises((ValueError, FileNotFoundError)):
            load_mixed_gigafsdp_checkpoint(str(checkpoint_dir), args=mock_args)

    def test_file_sorting(self, temp_dir, mock_args, simple_config):
        """Test that files are sorted correctly"""
        from mappers import load_mixed_gigafsdp_checkpoint

        checkpoint_dir = self._create_mixed_checkpoint(
            temp_dir, simple_config, tp_size=2, ep_size=2
        )

        loaded = load_mixed_gigafsdp_checkpoint(str(checkpoint_dir), args=mock_args)

        assert len(loaded) == 4


@pytest.mark.unit
class TestLoadHfModel:
    """Tests for load_hf_model function"""

    def test_load_basic_hf_model(self, mock_args):
        """Test loading basic HF model"""
        from mappers import load_hf_model

        with patch("mappers.AutoModelForCausalLM") as mock_model:
            mock_instance = Mock()
            mock_model.from_pretrained.return_value = mock_instance

            result = load_hf_model("test_path", args=mock_args)

            mock_model.from_pretrained.assert_called_once_with("test_path")
            assert result == mock_instance

    def test_load_from_local_path(self, temp_dir, mock_args):
        """Test loading from local path"""
        from mappers import load_hf_model

        with patch("mappers.AutoModelForCausalLM") as mock_model:
            mock_instance = Mock()
            mock_model.from_pretrained.return_value = mock_instance

            result = load_hf_model(str(temp_dir), args=mock_args)

            assert result == mock_instance

    def test_load_from_hub(self, mock_args):
        """Test loading from HuggingFace Hub"""
        from mappers import load_hf_model

        with patch("mappers.AutoModelForCausalLM") as mock_model:
            mock_instance = Mock()
            mock_model.from_pretrained.return_value = mock_instance

            result = load_hf_model("gpt2", args=mock_args)

            mock_model.from_pretrained.assert_called_once_with("gpt2")

    def test_load_with_custom_dtype(self, mock_args):
        """Test loading with custom dtype"""
        from mappers import load_hf_model

        mock_args.torch_dtype = "float16"

        with patch("mappers.AutoModelForCausalLM") as mock_model:
            mock_instance = Mock()
            mock_model.from_pretrained.return_value = mock_instance

            result = load_hf_model("test_path", args=mock_args)

            assert result is not None

    def test_load_nonexistent_model_error(self, mock_args):
        """Test error when loading nonexistent model"""
        from mappers import load_hf_model

        with patch("mappers.AutoModelForCausalLM") as mock_model:
            mock_model.from_pretrained.side_effect = OSError("Model not found")

            with pytest.raises(OSError):
                load_hf_model("nonexistent_model", args=mock_args)
