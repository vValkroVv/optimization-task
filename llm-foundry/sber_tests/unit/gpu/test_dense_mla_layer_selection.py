import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("the DSA kernels are Hopper-only (sm90)", allow_module_level=True)
pytest.importorskip(
    "llmfoundry.models.layers.dsa_operator",
    reason="the DSA kernel packages are not installed in this image",
)

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.layers.attention import LlamaLatentAttention
from llmfoundry.models.layers.blocks import LlamaDecoderLayer
from llmfoundry.models.layers.dsa_attention import LlamaLatentSparseAttention


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_dense_mla_and_sparse_dsa_layers_forward_backward():
    config = GigaMixConfig(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_krope_heads=1,
        max_position_embeddings=128,
        attention_type="LlamaLatentSparseAttention",
        dense_attention_layers=[0],
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        q_lora_rank=512,
        indexer_n_heads=64,
        indexer_head_dim=128,
        indexer_topk=128,
        aux_kl_loss_koeff=1.0,
        dsa_stage=2,
        apply_torch_compile_to_projections=False,
        use_custom_rotary_kernel=False,
        tp_size=1,
        init_device="cuda",
    )
    layers = torch.nn.ModuleList(
        [LlamaDecoderLayer(config, layer_idx=i) for i in range(2)]
    ).to(dtype=torch.bfloat16)

    assert isinstance(layers[0].self_attn, LlamaLatentAttention)
    assert not isinstance(layers[0].self_attn, LlamaLatentSparseAttention)
    assert isinstance(layers[1].self_attn, LlamaLatentSparseAttention)

    hidden_states = torch.randn(
        1,
        128,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    for layer in layers:
        hidden_states = layer(hidden_states)[0]

    loss = hidden_states.float().square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert layers[0].self_attn.dq_proj.weight.grad is not None
    assert layers[1].self_attn.indexer.wq.weight.grad is not None
