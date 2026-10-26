

import torch
from transformers import PretrainedConfig
import os
from flash_attn.layers.rotary import apply_rotary_emb

from llmfoundry.models.layers.custom_embedding import (
    LlamaRotaryEmbedding,
    apply_rotary_emb_mla,
)

torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

bsz, seq_len, num_heads, qk_head_dim = 4, 32, 32, 512

# NOTE: inplace=True causes non-determinism!!!

N_TIMES = 1000


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # To ensure reproducibility (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

seed = 0
set_seed(seed)

for i in range(N_TIMES):

    # seed = i + 42
    # set_seed(seed)

    qk_nope_head_dim=qk_rope_head_dim = qk_head_dim // 2

    # Инициализация на CPU/GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = PretrainedConfig(
        max_position_embeddings=8192,
        rope_theta=10000.0,
        rope_scaling={"rope_type": "linear", "factor": 2.0},
        hidden_size=qk_rope_head_dim,  # Указываем размерность RoPE-части
        num_attention_heads=num_heads,
        head_dim=qk_rope_head_dim  # Явно задаем размерность головы
    )

    rotary_emb = LlamaRotaryEmbedding(config, device=device)
    dtype = torch.bfloat16

    query = torch.randn(bsz, seq_len, num_heads, qk_head_dim, device=device, dtype=torch.bfloat16)

    # CUSTOM KERNEL
    query_custom = query.clone().requires_grad_(True)

    cos_custom, sin_custom = rotary_emb(query_custom, seq_len=seq_len)

    cos_custom = cos_custom.contiguous()
    sin_custom = sin_custom.contiguous()

    query_custom_final = apply_rotary_emb_mla(query_custom, cos_custom, sin_custom, inplace=False, head_offset=qk_nope_head_dim)


    # DEFAULT KERNEL
    query_default = query.clone().requires_grad_(True)

    q_nope, q_rope = torch.split(query_default, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    cos_default, sin_default = rotary_emb(q_rope, seq_len=seq_len)

    q_rope = apply_rotary_emb(q_rope, cos_default, sin_default, inplace=False)
    query_default_final = torch.cat([q_nope, q_rope], dim=-1)

    grad_output = torch.randn_like(query_default_final)
    query_custom_final.backward(grad_output.clone())
    query_default_final.backward(grad_output.clone())

    assert torch.allclose(query_custom_final, query_default_final) # type: ignore
    assert torch.allclose(query_custom.grad, query_default.grad) # type: ignore

    print(f"{i+1}/{N_TIMES} PASSED")

