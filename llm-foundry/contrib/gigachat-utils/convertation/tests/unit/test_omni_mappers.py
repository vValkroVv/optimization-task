import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch
from collections import OrderedDict


@pytest.mark.unit
class TestFsdpToOmniFsdp:
    """Tests for map_fsdp_to_omni_fspd"""

    def test_rename_layers_for_gigaomni(self, simple_fsdp_checkpoint, mock_args):
        """Test renaming layers to GigaOmni format"""
        from mappers import map_fsdp_to_omni_fsdp

        result = map_fsdp_to_omni_fsdp(simple_fsdp_checkpoint, args=mock_args)

        model = result["state"]["model"]

        # Check that language_model prefix is added for lm_head
        assert "model.language_model.lm_head.weight" in model

        # Old keys should be removed (model.model.* becomes model.language_model.model.*)
        # But model.lm_head -> model.language_model.lm_head
        assert "model.lm_head.weight" not in model

    def test_all_model_layers_renamed(self, simple_fsdp_checkpoint, mock_args):
        """Test that all model layers are renamed"""
        from mappers import map_fsdp_to_omni_fsdp

        result = map_fsdp_to_omni_fsdp(simple_fsdp_checkpoint, args=mock_args)

        model = result["state"]["model"]

        # Check that NO keys with old format exist
        for key in model.keys():
            if key.startswith("model.") and "lm_head" in key:
                assert "language_model" in key, f"Key {key} not renamed properly"

    def test_works_with_packed_attention(self, simple_fsdp_checkpoint, mock_args):
        """Test working with packed attention"""
        from mappers import map_base_to_packed_attn, map_fsdp_to_omni_fsdp

        mock_args.skip_optimizer = True
        packed = map_base_to_packed_attn(simple_fsdp_checkpoint, args=mock_args)

        result = map_fsdp_to_omni_fsdp(packed, args=mock_args)

        model = result["state"]["model"]
        # Should have renamed lm_head
        assert "model.language_model.lm_head.weight" in model

    def test_works_with_tp_size_greater_than_1(self, mock_args):
        """Test working with tp_size > 1"""
        from mappers import map_fsdp_to_omni_fsdp

        tp_checkpoints = self._create_tp2_checkpoints()

        result = map_fsdp_to_omni_fsdp(tp_checkpoints, args=mock_args)

        assert isinstance(result, list)
        assert len(result) == 2

        for ckpt in result:
            model = ckpt["state"]["model"]
            assert any("language_model" in key for key in model.keys())

    def _create_tp2_checkpoints(self):
        """Creates 2 checkpoints with tp_size=2"""
        checkpoints = []
        for rank in range(2):
            ckpt = {
                "state": {
                    "model": OrderedDict(
                        {
                            "model.embed_tokens.weight": torch.randn(1000, 256),
                            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                                256, 512
                            ),
                            "model.lm_head.weight": torch.randn(1000, 512),
                        }
                    ),
                    "integrations": {
                        "huggingface": {
                            "model": {
                                "config": {
                                    "content": {
                                        "model_type": "gigar",
                                        "tp_size": 2,
                                    }
                                }
                            }
                        }
                    },
                }
            }
            checkpoints.append(ckpt)
        return checkpoints


@pytest.mark.unit
class TestExpandEmbeddings:
    """Tests for map_expand_embeddings_and_lm_head_fsdp"""

    def test_expand_embeddings_basic(self, simple_fsdp_checkpoint, mock_args):
        from mappers import map_expand_embeddings_and_lm_head_fsdp

        original_vocab_size = simple_fsdp_checkpoint["state"]["model"][
            "model.embed_tokens.weight"
        ].shape[0]
        expand_size = 100
        mock_args.embeddings_expand_size = expand_size

        result = map_expand_embeddings_and_lm_head_fsdp(
            simple_fsdp_checkpoint, args=mock_args
        )

        new_embed = result["state"]["model"]["model.embed_tokens.weight"]
        new_lm_head = result["state"]["model"]["model.lm_head.weight"]

        assert new_embed.shape[0] == original_vocab_size + expand_size
        assert new_lm_head.shape[0] == original_vocab_size + expand_size

    def test_original_weights_preserved(self, simple_fsdp_checkpoint, mock_args):
        """Test that original weights are preserved"""
        from mappers import map_expand_embeddings_and_lm_head_fsdp

        original_embed = simple_fsdp_checkpoint["state"]["model"][
            "model.embed_tokens.weight"
        ].clone()
        mock_args.embeddings_expand_size = 50

        result = map_expand_embeddings_and_lm_head_fsdp(
            simple_fsdp_checkpoint, args=mock_args
        )

        new_embed = result["state"]["model"]["model.embed_tokens.weight"]

        assert torch.allclose(new_embed[: original_embed.shape[0]], original_embed)

    def test_new_weights_initialized(self, simple_fsdp_checkpoint, mock_args):
        """Test that new weights are initialized"""
        from mappers import map_expand_embeddings_and_lm_head_fsdp

        original_vocab_size = simple_fsdp_checkpoint["state"]["model"][
            "model.embed_tokens.weight"
        ].shape[0]
        expand_size = 50
        mock_args.embeddings_expand_size = expand_size

        result = map_expand_embeddings_and_lm_head_fsdp(
            simple_fsdp_checkpoint, args=mock_args
        )

        new_embed = result["state"]["model"]["model.embed_tokens.weight"]

        new_rows = new_embed[original_vocab_size:]
        assert not torch.allclose(new_rows, torch.zeros_like(new_rows))

    def test_fails_with_tp_size_greater_than_1(self, mock_args):
        """Test that fails with tp_size > 1"""
        from mappers import map_expand_embeddings_and_lm_head_fsdp

        checkpoint = {
            "state": {
                "model": OrderedDict(
                    {
                        "model.embed_tokens.weight": torch.randn(1000, 512),
                        "model.lm_head.weight": torch.randn(1000, 512),
                    }
                ),
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "model_type": "gigar",
                                    "tp_size": 2,
                                }
                            }
                        }
                    }
                },
            }
        }

        mock_args.embeddings_expand_size = 100

        with pytest.raises(AssertionError, match="tp_size"):
            map_expand_embeddings_and_lm_head_fsdp(checkpoint, args=mock_args)

    def test_works_with_gigaomni_config(self, mock_args):
        """Test working with GigaOmni config"""
        from mappers import map_expand_embeddings_and_lm_head_fsdp

        checkpoint = {
            "state": {
                "model": OrderedDict(
                    {
                        "model.embed_tokens.weight": torch.randn(1000, 512),
                        "model.lm_head.weight": torch.randn(1000, 512),
                    }
                ),
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": {
                                    "model_type": "gigaomni",
                                    "language_model": {
                                        "tp_size": 1,
                                    },
                                }
                            }
                        }
                    }
                },
            }
        }

        mock_args.embeddings_expand_size = 50

        result = map_expand_embeddings_and_lm_head_fsdp(checkpoint, args=mock_args)

        assert result["state"]["model"]["model.embed_tokens.weight"].shape[0] == 1050
