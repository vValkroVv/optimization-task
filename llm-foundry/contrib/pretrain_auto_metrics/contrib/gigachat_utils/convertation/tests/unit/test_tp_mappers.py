import pytest
import torch
import copy
from collections import OrderedDict


@pytest.mark.unit
class TestTpToTp:

    def test_same_tp_size_no_conversion(self, simple_fsdp_checkpoint, mock_args):
        """Test that same tp_size doesn't change anything"""
        from mappers import map_tp_to_tp

        mock_args.input_tp_size = 1
        mock_args.output_tp_size = 1

        original_keys = set(simple_fsdp_checkpoint["state"]["model"].keys())
        result = map_tp_to_tp(simple_fsdp_checkpoint, args=mock_args)

        # Keys should be the same
        result_keys = set(result["state"]["model"].keys())
        assert original_keys == result_keys

    def test_tp1_to_tp2_conversion(self, simple_fsdp_checkpoint, mock_args):
        """Test conversion from tp_size=1 to tp_size=2"""
        from mappers import map_tp_to_tp

        mock_args.input_tp_size = 1
        mock_args.output_tp_size = 2
        mock_args.skip_optimizer = True

        result = map_tp_to_tp(simple_fsdp_checkpoint, args=mock_args)

        assert isinstance(result, list)
        assert len(result) == 2
        # Weights should be present in each shard
        assert "model.embed_tokens.weight" in result[0]["state"]["model"]

    def test_tp2_to_tp1_conversion(self, mock_args):
        """Test conversion from tp_size=2 to tp_size=1"""
        from mappers import map_tp_to_tp

        tp_checkpoints = self._create_tp2_checkpoints()
        mock_args.input_tp_size = 2
        mock_args.output_tp_size = 1
        mock_args.skip_optimizer = True

        result = map_tp_to_tp(tp_checkpoints, args=mock_args)

        assert isinstance(result, dict)
        assert "state" in result
        assert "model" in result["state"]

    def test_tp2_to_tp4_conversion(self, mock_args):
        """Test conversion from tp_size=2 to tp_size=4"""
        from mappers import map_tp_to_tp

        tp_checkpoints = self._create_tp2_checkpoints()
        mock_args.input_tp_size = 2
        mock_args.output_tp_size = 4
        mock_args.skip_optimizer = True

        result = map_tp_to_tp(tp_checkpoints, args=mock_args)

        assert isinstance(result, list)
        assert len(result) == 4

    def test_packed_attention_handled(self, simple_fsdp_checkpoint, mock_args):
        """Test packed attention handling when changing tp_size"""
        from mappers import map_base_to_packed_attn, map_tp_to_tp

        # Convert to packed attention
        mock_args.skip_optimizer = True
        packed = map_base_to_packed_attn(simple_fsdp_checkpoint, args=mock_args)

        mock_args.input_tp_size = 1
        mock_args.output_tp_size = 2

        result = map_tp_to_tp(packed, args=mock_args)

        # Should remain in packed format
        assert "model.layers.0.self_attn.qkv_proj.weight" in result[0]["state"]["model"]

    def test_embedding_parallel_split(self, simple_fsdp_checkpoint, mock_args):
        """Test correct EmbeddingParallelEmbedding split"""
        from mappers import map_tp_to_tp

        mock_args.input_tp_size = 1
        mock_args.output_tp_size = 2
        mock_args.output_tp_embedding_type = "EmbeddingParallelEmbedding"
        mock_args.skip_optimizer = True

        result = map_tp_to_tp(simple_fsdp_checkpoint, args=mock_args)

        # Embedding should be split by dim=1
        emb0 = result[0]["state"]["model"]["model.embed_tokens.weight"]
        emb1 = result[1]["state"]["model"]["model.embed_tokens.weight"]

        assert emb0.shape[1] == emb1.shape[1]
        assert (
            emb0.shape[1] * 2
            == simple_fsdp_checkpoint["state"]["model"][
                "model.embed_tokens.weight"
            ].shape[1]
        )

    def test_vocab_parallel_split(self, simple_fsdp_checkpoint, mock_args):
        """Test correct VocabParallelEmbedding split"""
        from mappers import map_tp_to_tp

        mock_args.input_tp_size = 1
        mock_args.output_tp_size = 2
        mock_args.output_tp_embedding_type = "VocabParallelEmbedding"
        mock_args.skip_optimizer = True

        result = map_tp_to_tp(simple_fsdp_checkpoint, args=mock_args)

        # Embedding should be split by dim=0
        emb0 = result[0]["state"]["model"]["model.embed_tokens.weight"]
        emb1 = result[1]["state"]["model"]["model.embed_tokens.weight"]

        # Both shards should have same shape when split by vocab
        assert emb0.shape[0] == emb1.shape[0]

    def test_optimizer_preserved_when_not_skipped(
        self, simple_fsdp_checkpoint, mock_args
    ):
        """Test optimizer preserved when skip_optimizer=False"""
        from mappers import map_tp_to_tp

        mock_args.input_tp_size = 1
        mock_args.output_tp_size = 2
        mock_args.skip_optimizer = False

        result = map_tp_to_tp(simple_fsdp_checkpoint, args=mock_args)

        # Optimizer should be present in both shards
        assert "optimizers" in result[0]["state"]
        assert "optimizers" in result[1]["state"]

    def test_optimizer_removed_when_skipped(self, simple_fsdp_checkpoint, mock_args):
        """Test optimizer removed when skip_optimizer=True"""
        from mappers import map_tp_to_tp

        mock_args.input_tp_size = 1
        mock_args.output_tp_size = 2
        mock_args.skip_optimizer = True

        # Remove optimizer first
        del simple_fsdp_checkpoint["state"]["optimizers"]

        result = map_tp_to_tp(simple_fsdp_checkpoint, args=mock_args)

        assert "optimizers" not in result[0]["state"]

    def test_config_tp_size_updated(self, simple_fsdp_checkpoint, mock_args):
        """Test config tp_size is updated"""
        from mappers import map_tp_to_tp

        mock_args.input_tp_size = 1
        mock_args.output_tp_size = 2
        mock_args.skip_optimizer = True

        result = map_tp_to_tp(simple_fsdp_checkpoint, args=mock_args)

        config0 = result[0]["state"]["integrations"]["huggingface"]["model"]["config"][
            "content"
        ]
        assert config0["tp_size"] == 2

    def _create_tp2_checkpoints(self):
        """Creates tp_size=2 checkpoints"""
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
                            "model.layers.0.self_attn.k_proj.weight": torch.randn(
                                256, 512
                            ),
                            "model.layers.0.self_attn.v_proj.weight": torch.randn(
                                256, 512
                            ),
                            "model.layers.0.self_attn.o_proj.weight": torch.randn(
                                512, 256
                            ),
                        }
                    ),
                    "integrations": {
                        "huggingface": {
                            "model": {
                                "config": {
                                    "content": {
                                        "model_type": "gigar",
                                        "hidden_size": 512,
                                        "num_attention_heads": 8,
                                        "num_key_value_heads": 8,
                                        "tp_size": 2,
                                        "architectures": ["LlamaForCausalLM"],
                                    }
                                }
                            }
                        }
                    },
                }
            }
            checkpoints.append(ckpt)
        return checkpoints
