import pickle
import argparse
import os
from functools import partial
import time

import torch
from torch import nn
import numpy as np
import torch.distributed as dist
from torch.utils import benchmark

from llmfoundry.models.layers.attention import LlamaPackedAttention, LlamaPackedLatentAttention
from composer.utils.dist import (
    initialize_dist,
    get_tp_group,
    get_tp_group_size,
    # get_tp_group_rank,
)

def get_tp_group_rank():
    return 0

from composer.utils.device import get_device

from benchmark import benchmark_module

from sber_benchmarks.references.deepseek_v3.modeling_deepseek import DeepseekV3Attention
from sber_benchmarks.references.deepseek_v3.configuration_deepseek import DeepseekV3Config


print = partial(print, flush=True)

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

torch.cuda.set_device(LOCAL_RANK)
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=2)
torch.manual_seed(3 + RANK)
torch.cuda.manual_seed_all(3 + RANK)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

np.random.seed(3 + RANK)

def setup_dist():
    initialize_dist(get_device(None), tp_size=1, sp_size=1, timeout=999999)
    torch.distributed.barrier()
    torch.cuda.set_device(get_tp_group_rank())
    torch.cuda.synchronize()

def seed_everything(seed: int):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def print_diffs(lhs: torch.Tensor, rhs: torch.Tensor, name: str = "") -> None:
    assert lhs.shape == rhs.shape
    mean_diff = (lhs - rhs).abs().mean().item()
    max_diff = (lhs - rhs).abs().max().item()
    print(f"{name}: mean_diff={'{:.8f}'.format(mean_diff)}, max_diff={'{:.8f}'.format(max_diff)}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=2, required=False)
    parser.add_argument("--seq_len", type=int, default=8192, required=False)
    parser.add_argument("--hidden_size", type=int, default=4096, required=False)
    parser.add_argument("--num_attention_heads", type=int, default=32, required=False)
    parser.add_argument("--num_key_value_heads", type=int, default=32, required=False)
    parser.add_argument("--qk_nope_head_dim", type=int, default=64, required=False)
    parser.add_argument("--qk_rope_head_dim", type=int, default=64, required=False)
    parser.add_argument("--v_head_dim", type=int, default=128, required=False)
    parser.add_argument("--kv_lora_rank", type=int, default=1024, required=False)
    parser.add_argument("--q_lora_rank", type=int, default=1024, required=False)
    parser.add_argument("--max_position_embeddings", type=int, default=8192, required=False)
    parser.add_argument("--rope_theta", type=float, default=10000.0, required=False)
    parser.add_argument("--attention_bias", type=bool, default=False, required=False)
    parser.add_argument("--norm_type", type=str, default="rmsnorm", required=False)
    parser.add_argument("--apply_torch_compile_to_projections", type=bool, default=True, required=False)
    parser.add_argument("--use_custom_rotary_kernel", type=bool, default=True, required=False)


    
    parser.add_argument("--warmup_iters", type=int, default=10, required=False)
    parser.add_argument("--iters", type=int, default=10, required=False)
    parser.add_argument("--trace", action="store_true", default=True)
    parser.add_argument("--profile_memory", action="store_true", default=True)

    args = parser.parse_args()

    tp_size = 1
    assert args.seq_len % tp_size == 0
    assert args.hidden_size % tp_size == 0
    assert args.num_attention_heads % tp_size == 0
    assert args.num_key_value_heads % tp_size == 0

    return args

def make_config(args: argparse.Namespace):
    config_dict = {
        'attention_type': 'LlamaPackedLatentAttention',
        'hidden_size': args.hidden_size,
        'attention_hidden_size': None,  # Use hidden_size
        'num_attention_heads': args.num_attention_heads,
        'num_key_value_heads': args.num_key_value_heads,
        'qk_nope_head_dim': args.qk_nope_head_dim,
        'qk_rope_head_dim': args.qk_rope_head_dim,
        'v_head_dim': args.v_head_dim,
        'kv_lora_rank': args.kv_lora_rank,
        'q_lora_rank': args.q_lora_rank,
        'max_position_embeddings': args.max_position_embeddings,
        'rope_theta': args.rope_theta,
        'attention_bias': args.attention_bias,
        'norm_type': args.norm_type,
        'deterministic_attention': True,
        'init_device': 'cpu',
        'apply_torch_compile_to_projections': args.apply_torch_compile_to_projections,
        'use_custom_rotary_kernel': args.use_custom_rotary_kernel,
    }
    
    config = type('Config', (object,), config_dict)
    return config

def make_deepseek_config(args: argparse.Namespace):
    # taken from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/configuration_deepseek.py
    return DeepseekV3Config(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        attention_bias=args.attention_bias,
        _attn_implementation="eager",  # Use eager implementation for testing
        deterministic_attention=True,
    )

def make_latent_attention(args: argparse.Namespace):
    config = make_config(args)
    return LlamaPackedLatentAttention(config).cuda()

def make_packed_attention(args: argparse.Namespace):
    config = make_config(args)
    return LlamaPackedAttention(config).cuda()

def make_deepseek_attention(args: argparse.Namespace):
    config = make_deepseek_config(args)
    return DeepseekV3Attention(config, layer_idx=0).cuda()

def create_causal_mask(bsz, seq_len, device):
    # mask = torch.ones(bsz, 1, seq_len, seq_len, device=device)
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device) * -10000.0, diagonal=1)

    mask = mask.unsqueeze(0).unsqueeze(0)
    return torch.cat([mask, mask], dim=0)

def make_input(args: argparse.Namespace):
    hidden_states = torch.randn(
        args.batch_size,
        args.seq_len,
        args.hidden_size,
        dtype=torch.float16,
        device=torch.cuda.current_device(),
        requires_grad=True,
    )
    
    attention_mask = create_causal_mask(args.batch_size, args.seq_len, torch.cuda.current_device())
    
    position_ids = torch.arange(
        0, args.seq_len, dtype=torch.long, device=torch.cuda.current_device()
    ).unsqueeze(0).expand(args.batch_size, -1)
    
    return hidden_states, attention_mask, position_ids

def copy_weights_from_our_to_deepseek(our_module, deepseek_module):
    print("Copying weights from our implementation to DeepSeek...")
    
    # 1. Handle q projections
    if our_module.q_lora_rank == 0:
        # Direct copy for q_proj
        assert hasattr(deepseek_module, 'q_proj'), "DeepSeek module missing q_proj"
        deepseek_module.q_proj.weight.data.copy_(our_module.q_proj.weight.data)
        if hasattr(deepseek_module.q_proj, 'bias') and deepseek_module.q_proj.bias is not None:
            deepseek_module.q_proj.bias.data.copy_(our_module.q_proj.bias.data)
    else:
        # Copy lora components
        deepseek_module.q_a_proj.weight.data.copy_(our_module.dq_proj.weight.data)
        if hasattr(deepseek_module.q_a_proj, 'bias') and deepseek_module.q_a_proj.bias is not None:
            deepseek_module.q_a_proj.bias.data.copy_(our_module.dq_proj.bias.data)
            
        # Copy q norm params
        deepseek_module.q_a_layernorm.weight.data.copy_(our_module.q_norm.weight.data)
        
        # Copy q_b_proj (second part of lora)
        deepseek_module.q_b_proj.weight.data.copy_(our_module.uq_proj.weight.data)
        if hasattr(deepseek_module.q_b_proj, 'bias') and deepseek_module.q_b_proj.bias is not None:
            deepseek_module.q_b_proj.bias.data.copy_(our_module.uq_proj.bias.data)
    
    # 2. Handle kv projections
    
    # For kv_a_proj_with_mqa in DeepSeek, we need to combine:
    # - dkv_proj (latent part)
    # - kr_proj (rope part, after reshaping)
    
    # First, prepare the kr_proj weight by reshaping it
    our_kr_weight = our_module.kr_proj.weight.data  # [num_heads*qk_rope_head_dim, hidden_size]
    
    # DeepSeek expects just rope_dim for the kr part, not multiplied by num_heads
    rope_dim = our_module.qk_rope_head_dim
    hidden_size = our_module.hidden_size
    num_heads = our_module.num_heads
    
    # Reshape our kr_proj weight for use in DeepSeek's kv_a_proj_with_mqa
    # We need just one set of rope weights, not per head
    our_kr_weight_reshaped = our_kr_weight.view(num_heads, rope_dim, hidden_size)
    our_kr_weight_reshaped = our_kr_weight_reshaped[0]  # Take just one head's worth [rope_dim, hidden_size]
    
    # Concatenate dkv_proj and reshaped kr_proj weights
    combined_kv_a_weight = torch.cat([
        our_module.dkv_proj.weight.data,  # [kv_lora_rank, hidden_size]
        our_kr_weight_reshaped              # [rope_dim, hidden_size]
    ], dim=0)
    
    # Copy to DeepSeek's kv_a_proj_with_mqa
    deepseek_module.kv_a_proj_with_mqa.weight.data.copy_(combined_kv_a_weight)
    if hasattr(deepseek_module.kv_a_proj_with_mqa, 'bias') and deepseek_module.kv_a_proj_with_mqa.bias is not None:
        # Combine biases if they exist
        combined_kv_a_bias = torch.cat([
            our_module.dkv_proj.bias.data,
            our_module.kr_proj.bias.data[:rope_dim]  # Take just one head's worth of bias
        ], dim=0)
        deepseek_module.kv_a_proj_with_mqa.bias.data.copy_(combined_kv_a_bias)
    
    # Copy kv norm params
    deepseek_module.kv_a_layernorm.weight.data.copy_(our_module.kv_norm.weight.data)
    
    # For kv_b_proj in DeepSeek, we need to combine:
    # - uk_proj (key part)
    # - uv_proj (value part)
    combined_kv_b_weight = torch.cat([
        our_module.uk_proj.weight.data,  # [num_heads*qk_nope_head_dim, kv_lora_rank]
        our_module.uv_proj.weight.data   # [num_heads*v_head_dim, kv_lora_rank]
    ], dim=0)
    
    # Copy to DeepSeek's kv_b_proj
    deepseek_module.kv_b_proj.weight.data.copy_(combined_kv_b_weight)
    if hasattr(deepseek_module.kv_b_proj, 'bias') and deepseek_module.kv_b_proj.bias is not None:
        # Combine biases if they exist
        combined_kv_b_bias = torch.cat([
            our_module.uk_proj.bias.data,
            our_module.uv_proj.bias.data
        ], dim=0)
        deepseek_module.kv_b_proj.bias.data.copy_(combined_kv_b_bias)
    
    # 3. Copy o_proj weights
    deepseek_module.o_proj.weight.data.copy_(our_module.o_proj.weight.data)
    if hasattr(deepseek_module.o_proj, 'bias') and deepseek_module.o_proj.bias is not None:
        deepseek_module.o_proj.bias.data.copy_(our_module.o_proj.bias.data)
    
    print("Weight copying completed")
    return deepseek_module

def copy_weights_from_deepseek_to_our(deepseek_module, our_module):
    print("Copying weights from DeepSeek implementation to ours...")
    
    if our_module.q_lora_rank == 0:
        our_module.q_proj.weight.data.copy_(deepseek_module.q_proj.weight.data)
        if hasattr(our_module.q_proj, 'bias') and our_module.q_proj.bias is not None:
            our_module.q_proj.bias.data.copy_(deepseek_module.q_proj.bias.data)
    else:
        our_module.dq_proj.weight.data.copy_(deepseek_module.q_a_proj.weight.data)
        if hasattr(our_module.dq_proj, 'bias') and our_module.dq_proj.bias is not None:
            our_module.dq_proj.bias.data.copy_(deepseek_module.q_a_proj.bias.data)
            
        our_module.q_norm.weight.data.copy_(deepseek_module.q_a_layernorm.weight.data)
        
        # Copy second part of lora
        our_module.uq_proj.weight.data.copy_(deepseek_module.q_b_proj.weight.data)
        if hasattr(our_module.uq_proj, 'bias') and our_module.uq_proj.bias is not None:
            our_module.uq_proj.bias.data.copy_(deepseek_module.q_b_proj.bias.data)
    
    # 2. Handle kv projections
    
    # Split DeepSeek's kv_a_proj_with_mqa into latent and rope parts
    kv_lora_rank = our_module.kv_lora_rank
    rope_dim = our_module.qk_rope_head_dim
    
    # Extract parts from DeepSeek's combined projection
    deepseek_dkv_weight = deepseek_module.kv_a_proj_with_mqa.weight.data[:kv_lora_rank]
    deepseek_kr_weight_single = deepseek_module.kv_a_proj_with_mqa.weight.data[kv_lora_rank:]
    
    # Copy latent projection
    our_module.dkv_proj.weight.data.copy_(deepseek_dkv_weight)
    if hasattr(our_module.dkv_proj, 'bias') and our_module.dkv_proj.bias is not None:
        our_module.dkv_proj.bias.data.copy_(deepseek_module.kv_a_proj_with_mqa.bias.data[:kv_lora_rank])
    
    # For the rope part, we need to replicate the single head weights across all heads
    num_heads = our_module.num_heads
    hidden_size = our_module.hidden_size
    
    # Replicate the single RoPE weights for all heads
    deepseek_kr_weight_all_heads = deepseek_kr_weight_single.repeat(num_heads, 1)
    
    # Reshape to match our kr_proj's expected shape [num_heads*rope_dim, hidden_size]
    our_module.kr_proj.weight.data.copy_(deepseek_kr_weight_all_heads)
    if hasattr(our_module.kr_proj, 'bias') and our_module.kr_proj.bias is not None:
        # Replicate bias too if present
        deepseek_kr_bias_single = deepseek_module.kv_a_proj_with_mqa.bias.data[kv_lora_rank:]
        our_module.kr_proj.bias.data.copy_(deepseek_kr_bias_single.repeat(num_heads))
    
    # Copy kv norm params
    our_module.kv_norm.weight.data.copy_(deepseek_module.kv_a_layernorm.weight.data)
    
    # Split DeepSeek's kv_b_proj into key and value parts
    nope_dim_total = our_module.num_heads * our_module.qk_nope_head_dim
    v_dim_total = our_module.num_heads * our_module.v_head_dim
    
    deepseek_uk_weight = deepseek_module.kv_b_proj.weight.data[:nope_dim_total]
    deepseek_uv_weight = deepseek_module.kv_b_proj.weight.data[nope_dim_total:]
    
    # Copy key and value projections
    our_module.uk_proj.weight.data.copy_(deepseek_uk_weight)
    our_module.uv_proj.weight.data.copy_(deepseek_uv_weight)
    
    if hasattr(deepseek_module.kv_b_proj, 'bias') and deepseek_module.kv_b_proj.bias is not None:
        if hasattr(our_module.uk_proj, 'bias') and our_module.uk_proj.bias is not None:
            our_module.uk_proj.bias.data.copy_(deepseek_module.kv_b_proj.bias.data[:nope_dim_total])
        if hasattr(our_module.uv_proj, 'bias') and our_module.uv_proj.bias is not None:
            our_module.uv_proj.bias.data.copy_(deepseek_module.kv_b_proj.bias.data[nope_dim_total:])
    
    # 3. Copy o_proj weights
    our_module.o_proj.weight.data.copy_(deepseek_module.o_proj.weight.data)
    if hasattr(our_module.o_proj, 'bias') and our_module.o_proj.bias is not None:
        our_module.o_proj.bias.data.copy_(deepseek_module.o_proj.bias.data)
    
    print("Weight copying completed")
    return our_module


def test_latent_attention():
    setup_dist()
    args = parse_args()
    
    import importlib.util
    import sys
    
    seed_everything(42)
    our_module = make_latent_attention(args).half()
    #our_module = make_packed_attention(args).half()
    hidden_states, attention_mask, position_ids = make_input(args)
    
    seed_everything(42)
    deepseek_module = make_deepseek_attention(args).half()

    # our_module = copy_weights_from_deepseek_to_our(deepseek_module, our_module)
    print('=== started first test ===')
    our_output, _, _ = our_module(
        hidden_states=hidden_states.clone(),
        attention_mask=None, #attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    print(f'allocated: {torch.cuda.max_memory_allocated() / (10 ** 9)} Gb')
    print('=== outputs (sample) ===')
    print(our_output[0, :4, :4])
    
    deepseek_output, _, _ = deepseek_module(
        hidden_states=hidden_states.clone(),
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    
    print(f"Our output shape: {our_output.shape}")
    print(f"DeepSeek output shape: {deepseek_output.shape}")
    assert our_output.shape == hidden_states.shape, f"Our output shape {our_output.shape} doesn't match input shape {hidden_states.shape}"
    assert deepseek_output.shape == hidden_states.shape, f"DeepSeek output shape {deepseek_output.shape} doesn't match input shape {hidden_states.shape}"
    
    print_diffs(our_output, deepseek_output, "our_vs_deepseek")
    
    our_grad_output = torch.randn_like(our_output)
    our_output.backward(our_grad_output)
    
    assert hidden_states.grad is not None, "Gradients not computed for hidden_states"
    
    torch.cuda.synchronize()
    with torch.no_grad():
        output1, _, _ = our_module(
            hidden_states=hidden_states.detach().clone(),
            attention_mask=None, #attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        output2, _, _ = our_module(
            hidden_states=hidden_states.detach().clone(),
            attention_mask=None, #attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    
    is_close = torch.allclose(output1, output2)
    print(f"Our outputs are deterministic: {is_close}")
    assert is_close, "Our outputs are not deterministic"
    
    # benchmark our module
    print("Benchmarking our LlamaPackedLatentAttention implementation:")
    run_benchmark(our_module, hidden_states.clone(), attention_mask, position_ids, args, "our_mla")
    
    # benchmark DeepSeek module
    print("Benchmarking DeepSeek MLA implementation:")
    run_benchmark(deepseek_module, hidden_states.clone(), attention_mask, position_ids, args, "deepseek_mla")

def run_benchmark(module, hidden_states, attention_mask, position_ids, args, prefix="latent_attention"):
    torch.cuda.synchronize()
    
    def forward_func(inputs):
        return module(
            hidden_states=inputs[0],
            attention_mask=None,#inputs[1],
            position_ids=inputs[2],
            use_cache=False,
        )[0]
    
    # Warm up
    for _ in range(args.warmup_iters):
        _ = forward_func((hidden_states, attention_mask, position_ids))
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(args.iters):
        _ = forward_func((hidden_states, attention_mask, position_ids))
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_iter = total_time / args.iters
    print(f"Average time per iteration: {time_per_iter*1000:.2f} ms")
    
    try:
        # TODO: 
        
        # If benchmark_module is available, use it for more detailed profiling
        perf_result = benchmark_module(
            (hidden_states, attention_mask, position_ids),
            forward_func,
            args.warmup_iters,
            args.iters,
            trace=args.trace,
            profile_memory=args.profile_memory,
        )
        
        torch.distributed.barrier()
        perf_result.print_table()
        
        if args.trace:
            perf_result.profile.export_chrome_trace(
                f"/home/hovyan/abeletsky/dump/{prefix}_trace_rank_{get_tp_group_rank()}.json.gz"
            )

        if args.profile_memory:
            with open(
                f"/home/hovyan/abeletsky/dump/{prefix}_stacktrace_rank{get_tp_group_rank()}.pickle", "wb"
            ) as f:
                pickle.dump(perf_result.memory_snapshot.snapshot, f)
    except (NameError, AttributeError):
        print("benchmark_module not available for detailed profiling")

if __name__ == "__main__":
    test_latent_attention()
