# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
#PYTHONPATH=$(pwd)/:$(pwd)/contrib/composer:$(pwd)/contrib/streaming
#TP=1: PYTHONPATH=$PYTHONPATH python3 -m pytest --disable-pytest-warnings -m target sber_tests/test_fused_qkv_attn.py
#TP=2: PYTHONPATH=$PYTHONPATH composer -n 2  -v --stdout {rank}.txt sber_tests/test_fused_qkv_attn.py

import pytest
import torch
import torch.distributed as dist
from composer.utils import reproducibility
from composer.utils.dist import initialize_dist, get_tp_group
from omegaconf import OmegaConf as om
from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.gigar.configuration_gigar import GigarConfig
from typing import Optional
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2


def assert_allclose(x, y, atol, rtol, pct=0.5):
    mask = torch.isclose(x, y, atol=atol, rtol=rtol)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print("{:.2f}% of values not close.".format(pct_diff))
        assert False, "{:.2f}% of values not close.".format(pct_diff)
    assert True


@pytest.mark.gpu
@pytest.mark.parametrize('seq_len', [256, 1024])
@pytest.mark.parametrize('hidden_size', [1024])
@pytest.mark.parametrize('num_attention_heads', [8, 16, 32])
@pytest.mark.parametrize('num_key_value_heads', [4, 8, 16])
@pytest.mark.parametrize('dtype', [torch.float, torch.bfloat16])
@pytest.mark.parametrize('tp_size', [1])
def test_three_attn_impl(seq_len: int,
                   hidden_size: int,
                   num_attention_heads: int,
                   num_key_value_heads: int,
                   tp_size: int,
                   dtype: torch.dtype,
                   device: str = 'cuda'):
    """Compare all attn impl with each other.

    Includes testing with and without attn_clip_qkv, attn_qk_ln, and alibi.
    """
    reproducibility.seed_all(7)
    FORWARD_ATOL = {torch.float: 1e-3, torch.half: 1e-3, torch.bfloat16: 1e-3}
    BACKWARD_ATOL = {torch.float: 1e-2, torch.half: 1e-2, torch.bfloat16: 1e-2}

    FORWARD_RTOL = {torch.float: 2e-3, torch.half: 4e-3, torch.bfloat16: 2e-3}
    BACKWARD_RTOL = {torch.float: 2e-2, torch.half: 1e-2, torch.bfloat16: 2e-2}

    config = GigarConfig(hidden_size=hidden_size,
                        num_attention_heads=num_attention_heads,
                        tp_size=tp_size,
                        num_key_value_heads=num_key_value_heads)
    head_dim = hidden_size // num_attention_heads
    kv_hidden_size = head_dim * num_key_value_heads

    if num_key_value_heads > num_attention_heads:
        pytest.skip()
    
    if head_dim > 256:
        pytest.skip()
    
    input_config = om.create({
        'bs': 1,
        'seq_len': seq_len,
        'hidden_size': hidden_size,
        'cu_seqlens' : None,
        'max_seqlen' : None
    })

    base_attn = ATTN_CLASS_REGISTRY['LlamaAttention'](config)
    llama_attn = LlamaFlashAttention2(config)
    fused_attn = ATTN_CLASS_REGISTRY['LlamaPackedAttention'](config)
    with torch.no_grad():
        llama_attn.load_state_dict(base_attn.state_dict())
        
        fused_attn.qkv_proj.weight.data.copy_(torch.cat([base_attn.q_proj.weight.data, 
                                                        base_attn.k_proj.weight.data,
                                                        base_attn.v_proj.weight.data], axis=0))
        fused_attn.o_proj.weight.copy_(base_attn.o_proj.weight)


    base_attn.to(device).to(dtype)
    fused_attn.to(device).to(dtype)
    llama_attn.to(device).to(dtype)

    assert base_attn.q_proj.weight.dtype == dtype

    x0 = torch.randn(input_config.bs, 
                     input_config.seq_len,
                     input_config.hidden_size, dtype=dtype).to(device)
    x1 = x0.clone().detach()
    x2 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True
    x2.requires_grad = True

    with torch.autocast(x0.device.type):
        y0, _, _ = base_attn(x0)                             
        y1, _, _ = fused_attn(x1)
        y2, _, _ = llama_attn(x2)
        
        loss0 = y0.sum()
        loss1 = y1.sum()
        loss2 = y2.sum()

    loss0.backward()
    loss1.backward()
    loss2.backward()

    assert_allclose(y2, y0, atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype])
    assert_allclose(y1, y0, atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype])
    assert_allclose(y1, y2, atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype])
    
    #grad compare
    assert base_attn.q_proj.weight.grad is not None
    assert base_attn.k_proj.weight.grad is not None
    assert base_attn.v_proj.weight.grad is not None
    assert base_attn.o_proj.weight.grad is not None
    assert fused_attn.qkv_proj.weight.grad is not None
    assert fused_attn.o_proj.weight.grad is not None
    
    assert_allclose(fused_attn.o_proj.weight.grad, 
                    base_attn.o_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])

    assert_allclose(fused_attn.qkv_proj.weight.grad[:head_dim * num_attention_heads], 
                    base_attn.q_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])
    
    key_slice = slice(hidden_size, hidden_size + kv_hidden_size)
    assert_allclose(fused_attn.qkv_proj.weight.grad[key_slice], 
                    base_attn.k_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])

    value_slice = slice(hidden_size + kv_hidden_size, 
                        hidden_size + 2 * kv_hidden_size)
    assert_allclose(fused_attn.qkv_proj.weight.grad[value_slice], 
                    llama_attn.v_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])
    assert_allclose(base_attn.v_proj.weight.grad, 
                    llama_attn.v_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])
    assert_allclose(fused_attn.qkv_proj.weight.grad[value_slice], 
                    base_attn.v_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])

    assert x0.grad is not None
    assert x1.grad is not None
    assert_allclose(x1.grad, x0.grad, atol=BACKWARD_ATOL[dtype], rtol=BACKWARD_RTOL[dtype])


@pytest.mark.target
@pytest.mark.parametrize('packed_seq', [True, False])
@pytest.mark.parametrize('masked', [True, False])
@pytest.mark.parametrize('seq_len', [256, 1024])
@pytest.mark.parametrize('hidden_size', [1024])
@pytest.mark.parametrize('num_attention_heads', [8, 16, 32])
@pytest.mark.parametrize('num_key_value_heads', [4, 8, 16])
@pytest.mark.parametrize('dtype', [torch.float, torch.bfloat16])
@pytest.mark.parametrize('tp_size', [1])
def test_attn_impl(seq_len: int,
                   hidden_size: int,
                   num_attention_heads: int,
                   num_key_value_heads: int,
                   tp_size: int,
                   dtype: torch.dtype,
                   masked: bool,
                   packed_seq: bool,
                   device: str = 'cuda'):
    """Compare all attn impl with each other.

    Includes testing with and without attn_clip_qkv, attn_qk_ln, and alibi.
    """
    reproducibility.seed_all(7)
    FORWARD_ATOL = {torch.float: 1e-3, torch.half: 1e-3, torch.bfloat16: 1e-3}
    BACKWARD_ATOL = {torch.float: 1e-2, torch.half: 1e-2, torch.bfloat16: 1e-2}

    FORWARD_RTOL = {torch.float: 2e-3, torch.half: 4e-3, torch.bfloat16: 2e-3}
    BACKWARD_RTOL = {torch.float: 2e-2, torch.half: 1e-2, torch.bfloat16: 2e-2}

    config = GigarConfig(hidden_size=hidden_size,
                        num_attention_heads=num_attention_heads,
                        tp_size=tp_size,
                        num_key_value_heads=num_key_value_heads)
    head_dim = hidden_size // num_attention_heads
    kv_hidden_size = head_dim * num_key_value_heads

    if num_key_value_heads > num_attention_heads:
        pytest.skip()
    
    if head_dim > 256:
        pytest.skip()
    
    input_config = om.create({
        'bs': 1 if masked else 2,
        'seq_len': seq_len,
        'hidden_size': hidden_size,
        'cu_seqlens' : None,
        'max_seqlen' : None
    })

    if packed_seq:
        cu_seqlens_q = torch.arange(0, (input_config.bs + 1) * input_config.seq_len, 
                                    step=input_config.seq_len, 
                                    dtype=torch.int32,
                                    device=device)
        max_seqlen = input_config.seq_len
    else:
        cu_seqlens_q = None
        max_seqlen = None
    
    base_attn = ATTN_CLASS_REGISTRY['LlamaAttention'](config)
    fused_attn = ATTN_CLASS_REGISTRY['LlamaPackedAttention'](config)
    with torch.no_grad():
        #add weights mapping    
        #fused_attn.load_state_dict(base_attn.state_dict())
        fused_attn.qkv_proj.weight.data.copy_(torch.cat([base_attn.q_proj.weight.data, 
                                                        base_attn.k_proj.weight.data,
                                                        base_attn.v_proj.weight.data], axis=0))
        fused_attn.o_proj.weight.copy_(base_attn.o_proj.weight)

    base_attn.to(device).to(dtype)
    fused_attn.to(device).to(dtype)

    assert base_attn.q_proj.weight.dtype == dtype

    x0 = torch.randn(input_config.bs, 
                     input_config.seq_len,
                     input_config.hidden_size, dtype=dtype).to(device)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    if masked:
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
        idx = torch.randint(seq_len // 2, seq_len, size=(1,))[0].item()
        attention_mask[idx:] = 0
    else:
        attention_mask = None

    with torch.autocast(x0.device.type):
        y0, _, _ = base_attn(x0, 
                             attention_mask=attention_mask,
                             cu_seqlens=cu_seqlens_q,
                             max_seqlen=max_seqlen)

        y1, _, _ = fused_attn(x1, 
                              attention_mask=attention_mask,
                              cu_seqlens=cu_seqlens_q,
                              max_seqlen=max_seqlen)
        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    assert_allclose(y1, y0, atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype])
    
    #grad compare
    assert base_attn.q_proj.weight.grad is not None
    assert base_attn.k_proj.weight.grad is not None
    assert base_attn.v_proj.weight.grad is not None
    assert base_attn.o_proj.weight.grad is not None
    assert fused_attn.qkv_proj.weight.grad is not None
    assert fused_attn.o_proj.weight.grad is not None
    
    assert_allclose(fused_attn.o_proj.weight.grad, 
                    base_attn.o_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])

    assert_allclose(fused_attn.qkv_proj.weight.grad[:head_dim * num_attention_heads], 
                    base_attn.q_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])
    
    key_slice = slice(hidden_size, hidden_size + kv_hidden_size)
    assert_allclose(fused_attn.qkv_proj.weight.grad[key_slice], 
                    base_attn.k_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])

    value_slice = slice(hidden_size + kv_hidden_size, 
                        hidden_size + 2 * kv_hidden_size)
    assert_allclose(fused_attn.qkv_proj.weight.grad[value_slice], 
                    base_attn.v_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])

    assert x0.grad is not None
    assert x1.grad is not None
    assert_allclose(x1.grad, x0.grad, atol=BACKWARD_ATOL[dtype], rtol=BACKWARD_RTOL[dtype])


@pytest.mark.llama
@pytest.mark.parametrize('seq_len', [256, 1024])
@pytest.mark.parametrize('hidden_size', [1024])
@pytest.mark.parametrize('num_attention_heads', [8, 16, 32])
@pytest.mark.parametrize('num_key_value_heads', [4, 8, 16])
@pytest.mark.parametrize('dtype', [torch.float, torch.bfloat16])
@pytest.mark.parametrize('tp_size', [1])
def test_llama_impl(seq_len: int,
                   hidden_size: int,
                   num_attention_heads: int,
                   num_key_value_heads: int,
                   tp_size: int,
                   dtype: torch.dtype,
                   device: str = 'cuda'):
    """Compare all attn impl with each other.

    Includes testing with and without attn_clip_qkv, attn_qk_ln, and alibi.
    """
    reproducibility.seed_all(7)
    FORWARD_ATOL = {torch.float: 1e-3, torch.half: 1e-3, torch.bfloat16: 1e-3}
    BACKWARD_ATOL = {torch.float: 1e-2, torch.half: 1e-2, torch.bfloat16: 1e-2}

    FORWARD_RTOL = {torch.float: 2e-3, torch.half: 4e-3, torch.bfloat16: 2e-3}
    BACKWARD_RTOL = {torch.float: 2e-2, torch.half: 1e-2, torch.bfloat16: 2e-2}

    config = GigarConfig(hidden_size=hidden_size,
                        num_attention_heads=num_attention_heads,
                        tp_size=tp_size,
                        num_key_value_heads=num_key_value_heads)
    head_dim = hidden_size // num_attention_heads
    kv_hidden_size = head_dim * num_key_value_heads
    STD_INIT = 0.1

    if num_key_value_heads > num_attention_heads:
        pytest.skip()
    
    if head_dim > 256:
        pytest.skip()
    
    input_config = om.create({
        'bs': 4,
        'seq_len': seq_len,
        'hidden_size': hidden_size,
        'cu_seqlens' : None,
        'max_seqlen' : None
    })

    base_attn = ATTN_CLASS_REGISTRY['LlamaAttention'](config)
    #base_attn = LlamaAttention(config)
    fused_attn = LlamaFlashAttention2(config)
    #ATTN_CLASS_REGISTRY['LlamaPackedAttention'](config)

    fused_attn.load_state_dict(base_attn.state_dict())

    base_attn.to(device).to(dtype)
    fused_attn.to(device).to(dtype)
    
    assert base_attn.q_proj.weight.dtype == dtype

    x0 = torch.randn(input_config.bs, 
                     input_config.seq_len,
                     input_config.hidden_size, dtype=dtype).to(device)
    #x0.normal_(0, 0.1)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    with torch.autocast(x0.device.type):
        y0, _, _ = base_attn(x0)
        y1, _, _ = fused_attn(x1)
        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    assert_allclose(y1, y0, atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype])

    #grad compare
    base_name_param_map = {n: p for n, p in base_attn.named_parameters()}
    for n, p in fused_attn.named_parameters():
        tp = base_name_param_map[n]
        assert p.grad is not None
        assert tp.grad is not None
        assert_allclose(p, tp, atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype])
        assert_allclose(p.grad, tp.grad, atol=BACKWARD_ATOL[dtype], rtol=BACKWARD_RTOL[dtype])

    assert x0.grad is not None
    assert x1.grad is not None
    assert_allclose(x1.grad, x0.grad, atol=BACKWARD_ATOL[dtype], rtol=BACKWARD_RTOL[dtype])

### TP_SIZE=2 TESTS

def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


def test_attn_tp(seq_len: int,
                hidden_size: int,
                num_attention_heads: int,
                num_key_value_heads: int,
                tp_size: int,
                dtype: torch.dtype,
                masked: bool,
                packed_seq: bool,
                device: str = 'cuda'):

    if tp_size > 1:
        initialize_dist("gpu", tp_size=tp_size)
        rank=dist.get_rank()
        device=f'cuda:{rank}'

    print(rank, device, get_tp_group().size())
    reproducibility.seed_all(7)

    FORWARD_ATOL = {torch.float: 1e-3, torch.half: 1e-3, torch.bfloat16: 1e-3}
    BACKWARD_ATOL = {torch.float: 1e-2, torch.half: 1e-2, torch.bfloat16: 1e-2}

    FORWARD_RTOL = {torch.float: 2e-3, torch.half: 4e-3, torch.bfloat16: 2e-3}
    BACKWARD_RTOL = {torch.float: 2e-2, torch.half: 1e-2, torch.bfloat16: 2e-2}

    config = GigarConfig(hidden_size=hidden_size,
                        num_attention_heads=num_attention_heads,
                        tp_size=tp_size,
                        num_key_value_heads=num_key_value_heads)
    head_dim = hidden_size // num_attention_heads
    kv_hidden_size = head_dim * num_key_value_heads

    input_config = om.create({
        'bs': 1 if masked else 2,
        'seq_len': seq_len,
        'hidden_size': hidden_size,
        'cu_seqlens' : None,
        'max_seqlen' : None
    })

    if packed_seq:
        cu_seqlens_q = torch.arange(0, (input_config.bs + 1) * input_config.seq_len, 
                                    step=input_config.seq_len, 
                                    dtype=torch.int32,
                                    device=device)
        max_seqlen = input_config.seq_len
    else:
        cu_seqlens_q = None
        max_seqlen = None
    
    base_attn = ATTN_CLASS_REGISTRY['LlamaAttention'](config)
    fused_attn = ATTN_CLASS_REGISTRY['LlamaPackedAttention'](config)
    with torch.no_grad():
        #add weights mapping
        fused_attn.qkv_proj.weight.data.copy_(torch.cat([base_attn.q_proj.weight.data, 
                                                        base_attn.k_proj.weight.data,
                                                        base_attn.v_proj.weight.data], axis=0))
        fused_attn.o_proj.weight.copy_(base_attn.o_proj.weight)

    print(rank, "q shape", base_attn.q_proj.weight.shape, base_attn.q_proj.weight)

    base_attn.to(device).to(dtype)
    fused_attn.to(device).to(dtype)

    assert base_attn.q_proj.weight.dtype == dtype

    x0 = torch.randn(input_config.bs, 
                     input_config.seq_len,
                     input_config.hidden_size, dtype=dtype).to(device)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    if masked:
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
        idx = torch.randint(seq_len // 2, seq_len, size=(1,))[0].item()
        attention_mask[idx:] = 0
    else:
        attention_mask = None

    get_tp_group().barrier()

    with torch.autocast(x0.device.type):
        y0, _, _ = base_attn(x0, 
                             attention_mask=attention_mask,
                             cu_seqlens=cu_seqlens_q,
                             max_seqlen=max_seqlen)

        y1, _, _ = fused_attn(x1, 
                              attention_mask=attention_mask,
                              cu_seqlens=cu_seqlens_q,
                              max_seqlen=max_seqlen)
        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    get_tp_group().barrier()
    if rank == 0:
        print("#" * 30)
        print("# asserts:")
        print("#" * 30)

    assert_allclose(y1, y0, atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype])
    
    log("y1", y1, rank0_only=True)
    log("y0", y0, rank0_only=True)
    log("fwd diff", y1 - y0)
    
    #print("predict:", y0, y1, y0.shape)

    #grad compare
    assert base_attn.q_proj.weight.grad is not None
    assert base_attn.k_proj.weight.grad is not None
    assert base_attn.v_proj.weight.grad is not None
    assert base_attn.o_proj.weight.grad is not None
    assert fused_attn.qkv_proj.weight.grad is not None
    assert fused_attn.o_proj.weight.grad is not None
    
    assert_allclose(fused_attn.o_proj.weight.grad, 
                    base_attn.o_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])

    assert_allclose(fused_attn.qkv_proj.weight.grad[:head_dim * num_attention_heads // tp_size], 
                    base_attn.q_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])

    #print("d_q:", fused_attn.qkv_proj.weight.grad[:head_dim * num_attention_heads // tp_size], base_attn.q_proj.weight.grad)
    log("d_q diff", fused_attn.qkv_proj.weight.grad[:head_dim * num_attention_heads // tp_size] - base_attn.q_proj.weight.grad)

    key_slice = slice(hidden_size // tp_size , (hidden_size + kv_hidden_size) // tp_size)
    assert_allclose(fused_attn.qkv_proj.weight.grad[key_slice], 
                    base_attn.k_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])

    log("d_k diff", fused_attn.qkv_proj.weight.grad[key_slice] - base_attn.k_proj.weight.grad)

    value_slice = slice((hidden_size + kv_hidden_size) // tp_size, 
                        (hidden_size + 2 * kv_hidden_size) // tp_size)
    assert_allclose(fused_attn.qkv_proj.weight.grad[value_slice], 
                    base_attn.v_proj.weight.grad,
                    atol=BACKWARD_ATOL[dtype], 
                    rtol=BACKWARD_RTOL[dtype])

    assert x0.grad is not None
    assert x1.grad is not None
    assert_allclose(x1.grad, x0.grad, atol=BACKWARD_ATOL[dtype], rtol=BACKWARD_RTOL[dtype])


if __name__ == "__main__":
    sq = [256, 1024]
    num_attention_heads = [8]
    num_key_value_heads = [4, 8]

    for seq_len in sq:
        for nh in num_attention_heads:
            for nh_kv in num_key_value_heads:
                try:
                    test_attn_tp(seq_len=seq_len,
                                hidden_size=1024,
                                num_attention_heads=nh,
                                num_key_value_heads=nh_kv,
                                tp_size=2,
                                dtype=torch.float,
                                masked=False,
                                packed_seq=False,
                                device='cuda')
                except Exception as e:
                    print(e)
