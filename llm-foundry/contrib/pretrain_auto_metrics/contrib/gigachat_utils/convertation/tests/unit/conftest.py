import pytest
import sys
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock
from collections import OrderedDict


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")


@pytest.fixture
def temp_dir():
    """Fixture that provides a temporary directory that is cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_config():
    """Simple model configuration for tests."""
    return {
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 1024,
        "vocab_size": 1000,
        "head_dim": 64,
        "attention_type": "LlamaPackedAttention",
        "model_type": "gigar",
        "tp_size": 1,
        "ep_size": 1,
        "architectures": ["LlamaForCausalLM"],
        "_name_or_path": "test-model",
    }


@pytest.fixture
def mock_args():
    """Mock args with all necessary attributes for mappers."""
    args = Mock()
    # Basic args
    args.tp_size = 1
    args.ep_size = 1
    args.output = "/tmp/test_output"
    args.skip_ema_mapper = True
    args.skip_optimizer = True
    args.use_ema = False
    args.packed_attention = True
    args.model_name = "test-model"
    args.model_type = "gigar"
    args.tokenizer_path = None
    args.hf_config = None
    args.save_adapter_only = False
    args.chunk_layers = 10
    args.rank = 0
    args.prepare_ema_weight = False

    # TP/EP conversion args
    args.input_tp_size = 1
    args.input_ep_size = 1
    args.output_tp_size = 1
    args.output_ep_size = 1
    args.input_tp_embedding_type = "EmbeddingParallelEmbedding"
    args.output_tp_embedding_type = "EmbeddingParallelEmbedding"

    # Safetensor args
    args.global_rank = 0
    args.world_size = 1
    args.output_checkpoint_path = "/tmp/test_output"

    # Embedding expansion args
    args.embeddings_expand_size = None

    # Torch dtype
    args.torch_dtype = "bfloat16"

    return args


@pytest.fixture
def simple_fsdp_checkpoint(simple_config):
    """
    Creates a simple FSDP checkpoint with the correct nested structure.
    Config is at state["integrations"]["huggingface"]["model"]["config"]["content"]
    """
    hidden_size = simple_config["hidden_size"]
    num_heads = simple_config["num_attention_heads"]
    num_kv_heads = simple_config["num_key_value_heads"]
    head_dim = hidden_size // num_heads
    kv_hidden_size = head_dim * num_kv_heads
    vocab_size = simple_config["vocab_size"]
    intermediate_size = simple_config["intermediate_size"]

    # Create model with two layers to match num_hidden_layers=2
    model = OrderedDict(
        {
            "model.embed_tokens.weight": torch.randn(vocab_size, hidden_size),
            # Layer 0
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                hidden_size, hidden_size
            ),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(
                kv_hidden_size, hidden_size
            ),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(
                kv_hidden_size, hidden_size
            ),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(
                hidden_size, hidden_size
            ),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(
                intermediate_size, hidden_size
            ),
            "model.layers.0.mlp.up_proj.weight": torch.randn(
                intermediate_size, hidden_size
            ),
            "model.layers.0.mlp.down_proj.weight": torch.randn(
                hidden_size, intermediate_size
            ),
            "model.layers.0.input_layernorm.weight": torch.ones(hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden_size),
            # Layer 1
            "model.layers.1.self_attn.q_proj.weight": torch.randn(
                hidden_size, hidden_size
            ),
            "model.layers.1.self_attn.k_proj.weight": torch.randn(
                kv_hidden_size, hidden_size
            ),
            "model.layers.1.self_attn.v_proj.weight": torch.randn(
                kv_hidden_size, hidden_size
            ),
            "model.layers.1.self_attn.o_proj.weight": torch.randn(
                hidden_size, hidden_size
            ),
            "model.layers.1.mlp.gate_proj.weight": torch.randn(
                intermediate_size, hidden_size
            ),
            "model.layers.1.mlp.up_proj.weight": torch.randn(
                intermediate_size, hidden_size
            ),
            "model.layers.1.mlp.down_proj.weight": torch.randn(
                hidden_size, intermediate_size
            ),
            "model.layers.1.input_layernorm.weight": torch.ones(hidden_size),
            "model.layers.1.post_attention_layernorm.weight": torch.ones(hidden_size),
            # Final layers
            "model.norm.weight": torch.ones(hidden_size),
            "model.lm_head.weight": torch.randn(vocab_size, hidden_size),
        }
    )

    # Create optimizer state
    optimizer_state = {}
    for key, tensor in model.items():
        if "weight" in key:
            optimizer_state[key] = {
                "exp_avg": torch.zeros_like(tensor),
                "exp_avg_sq": torch.zeros_like(tensor),
                "step": torch.tensor(1000),
            }

    # Proper nested structure that mappers expect
    return {
        "state": {
            "model": model,
            "integrations": {
                "huggingface": {"model": {"config": {"content": simple_config.copy()}}}
            },
            "optimizers": {
                "DecoupledLionW": {
                    "state": optimizer_state,
                    "param_groups": [{"params": list(model.keys())}],
                }
            },
        }
    }


@pytest.fixture
def sample_checkpoint(simple_config):
    """Alias for simple_fsdp_checkpoint for backward compatibility."""
    hidden_size = simple_config["hidden_size"]
    num_heads = simple_config["num_attention_heads"]
    num_kv_heads = simple_config["num_key_value_heads"]
    head_dim = hidden_size // num_heads
    kv_hidden_size = head_dim * num_kv_heads

    model = OrderedDict(
        {
            "model.embed_tokens.weight": torch.randn(1000, hidden_size),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                hidden_size, hidden_size
            ),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(
                kv_hidden_size, hidden_size
            ),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(
                kv_hidden_size, hidden_size
            ),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(
                hidden_size, hidden_size
            ),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(
                hidden_size * 4, hidden_size
            ),
            "model.layers.0.mlp.up_proj.weight": torch.randn(
                hidden_size * 4, hidden_size
            ),
            "model.layers.0.mlp.down_proj.weight": torch.randn(
                hidden_size, hidden_size * 4
            ),
            "model.norm.weight": torch.ones(hidden_size),
            "model.lm_head.weight": torch.randn(1000, hidden_size),
        }
    )

    return {
        "state": {
            "model": model,
            "integrations": {
                "huggingface": {"model": {"config": {"content": simple_config.copy()}}}
            },
        }
    }


@pytest.fixture
def packed_checkpoint(simple_config):
    """Checkpoint with packed QKV attention."""
    hidden_size = simple_config["hidden_size"]
    num_heads = simple_config["num_attention_heads"]
    num_kv_heads = simple_config["num_key_value_heads"]
    head_dim = hidden_size // num_heads
    kv_hidden_size = head_dim * num_kv_heads
    qkv_size = hidden_size + 2 * kv_hidden_size

    model = OrderedDict(
        {
            "model.embed_tokens.weight": torch.randn(1000, hidden_size),
            "model.layers.0.self_attn.qkv_proj.weight": torch.randn(
                qkv_size, hidden_size
            ),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(
                hidden_size, hidden_size
            ),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(
                hidden_size * 4, hidden_size
            ),
            "model.layers.0.mlp.up_proj.weight": torch.randn(
                hidden_size * 4, hidden_size
            ),
            "model.layers.0.mlp.down_proj.weight": torch.randn(
                hidden_size, hidden_size * 4
            ),
            "model.layers.1.self_attn.qkv_proj.weight": torch.randn(
                qkv_size, hidden_size
            ),
            "model.layers.1.self_attn.o_proj.weight": torch.randn(
                hidden_size, hidden_size
            ),
            "model.layers.1.mlp.gate_proj.weight": torch.randn(
                hidden_size * 4, hidden_size
            ),
            "model.layers.1.mlp.up_proj.weight": torch.randn(
                hidden_size * 4, hidden_size
            ),
            "model.layers.1.mlp.down_proj.weight": torch.randn(
                hidden_size, hidden_size * 4
            ),
            "model.norm.weight": torch.ones(hidden_size),
            "model.lm_head.weight": torch.randn(1000, hidden_size),
        }
    )

    return {
        "state": {
            "model": model,
            "integrations": {
                "huggingface": {"model": {"config": {"content": simple_config.copy()}}}
            },
        }
    }


@pytest.fixture
def checkpoint_with_ema(simple_config):
    """Checkpoint with EMA weights."""
    hidden_size = simple_config["hidden_size"]

    model = OrderedDict(
        {
            "model.embed_tokens.weight": torch.randn(1000, hidden_size),
            "model.embed_tokens.weight.ema": torch.randn(1000, hidden_size),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                hidden_size, hidden_size
            ),
            "model.layers.0.self_attn.q_proj.weight.ema": torch.randn(
                hidden_size, hidden_size
            ),
            "model.lm_head.weight": torch.randn(1000, hidden_size),
            "model.lm_head.weight.ema": torch.randn(1000, hidden_size),
        }
    )

    return {
        "state": {
            "model": model,
            "integrations": {
                "huggingface": {"model": {"config": {"content": simple_config.copy()}}}
            },
        }
    }


@pytest.fixture
def checkpoint_with_optimizer(simple_config):
    """Checkpoint with optimizer state."""
    hidden_size = simple_config["hidden_size"]

    model = OrderedDict(
        {
            "model.embed_tokens.weight": torch.randn(1000, hidden_size),
            "model.lm_head.weight": torch.randn(1000, hidden_size),
        }
    )

    optimizer_state = {}
    for key in model.keys():
        optimizer_state[key] = {
            "exp_avg": torch.zeros_like(model[key]),
            "exp_avg_sq": torch.zeros_like(model[key]),
        }

    return {
        "state": {
            "model": model,
            "integrations": {
                "huggingface": {"model": {"config": {"content": simple_config.copy()}}}
            },
            "optimizers": {"DecoupledLionW": {"state": optimizer_state}},
        }
    }
