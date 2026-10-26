import os
import torch
import numpy as np
import random


from llmfoundry.models.layers.attention import LlamaPackedAttention
from llmfoundry.models.gigar.configuration_gigar import GigarConfig

os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["NUMEXPR_MAX_THREADS"] = "16"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def test_llama_packed_attention():
    config_1 = GigarConfig(
        hidden_size=64,
        intermediate_size=64,
        num_attention_heads=2,
        sp_size=1,
        determinisitc_attention=True,
    )
    config_2 = GigarConfig(
        hidden_size=64,
        intermediate_size=64,
        num_attention_heads=2,
        sp_size=2,
        determinisitc_attention=True,
    )

    seed_all(42)
    attn_1 = LlamaPackedAttention(config_1)
    attn_1.to("cuda:0")
    attn_1 = attn_1.to(torch.bfloat16)
    seed_all(42)

    attn_2 = LlamaPackedAttention(config_2)
    attn_2.to("cuda:0")
    attn_2 = attn_2.to(torch.bfloat16)

    for _ in range(10):
        input_x = torch.randn(size=(16, 32, 64), device="cuda:0", dtype=torch.bfloat16)

        logits_1 = attn_1(input_x)[0]
        logits_2 = attn_2(input_x)[0]

        assert torch.allclose(logits_1, logits_2, atol=1e-8)
