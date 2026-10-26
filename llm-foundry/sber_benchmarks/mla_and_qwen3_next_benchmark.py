import gc
import os
import argparse
import sys
import numpy as np
import torch
torch.set_warn_always(True)
import time
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from composer.utils.dist import initialize_dist
from composer.utils.device import get_device


def get_tp_group_rank():
    return 0

from benchmark import benchmark_module

from llmfoundry.models.layers.attention import (
    LlamaLatentAttention,
    Qwen3NextGatedDeltaNet,
)

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig

# ENVS
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('benchmark_name', help='')
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--seq_len", type=int, default=16384, required=False)
    parser.add_argument("--hidden_size", type=int, default=2048, required=False)
    parser.add_argument("--num_attention_heads", type=int, default=16, required=False)
    parser.add_argument("--num_key_value_heads", type=int, default=16, required=False)
    parser.add_argument("--qk_nope_head_dim", type=int, default=64, required=False)
    parser.add_argument("--qk_rope_head_dim", type=int, default=64, required=False)
    parser.add_argument("--v_head_dim", type=int, default=128, required=False)
    parser.add_argument("--kv_lora_rank", type=int, default=1024, required=False)
    parser.add_argument("--q_lora_rank", type=int, default=0, required=False)
    parser.add_argument("--num_krope_heads", type=int, default=None, required=False)
    parser.add_argument("--apply_qk_norm", type=bool, default=False, required=False)
    parser.add_argument(
        "--max_position_embeddings", type=int, default=8192, required=False
    )
    parser.add_argument("--rope_theta", type=float, default=10000.0, required=False)
    parser.add_argument("--attention_bias", type=bool, default=False, required=False)
    parser.add_argument("--norm_type", type=str, default="rmsnorm", required=False)
    parser.add_argument(
        "--apply_torch_compile_to_projections", type=bool, default=False, required=False
    )
    parser.add_argument(
        "--use_custom_rotary_kernel", type=bool, default=False, required=False
    )

    parser.add_argument("--warmup_iters", type=int, default=10, required=False)
    parser.add_argument("--iters", type=int, default=50, required=False)
    parser.add_argument("--trace", action="store_true", default=True)
    parser.add_argument("--profile_memory", action="store_true", default=True)
    
    parser.add_argument("--trace_output_path", type=str, default=None, required=False,
                        help="Path to save trace output. If not provided, uses default path with prefix and rank.")
    parser.add_argument("--memory_profiler_output_path", type=str, default=None, required=False,
                        help="Path to save memory profiler stats. If not provided, uses default path with prefix and rank.")
    
    # Qwen3Next Gated Delta Net arguments
    parser.add_argument("--linear_key_head_dim", type=int, default=128, required=False)
    parser.add_argument("--linear_value_head_dim", type=int, default=128, required=False)
    parser.add_argument("--linear_conv_kernel_dim", type=int, default=4, required=False)
    parser.add_argument("--linear_num_key_heads", type=int, default=16, required=False)
    parser.add_argument("--linear_num_value_heads", type=int, default=32, required=False)
    parser.add_argument("--linear_use_short_conv", type=bool, default=False, required=False)

    args = parser.parse_args()

    tp_size = 1
    assert args.seq_len % tp_size == 0
    assert args.hidden_size % tp_size == 0
    assert args.num_attention_heads % tp_size == 0
    assert args.num_key_value_heads % tp_size == 0

    return args


def make_config(args: argparse.Namespace):
    config_dict = {
        "attention_type": "LlamaLatentAttention",
        "hidden_size": args.hidden_size,
        "attention_hidden_size": None,  # Use hidden_size
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "qk_nope_head_dim": args.qk_nope_head_dim,
        "qk_rope_head_dim": args.qk_rope_head_dim,
        "v_head_dim": args.v_head_dim,
        "kv_lora_rank": args.kv_lora_rank,
        "q_lora_rank": args.q_lora_rank,
        "max_position_embeddings": args.max_position_embeddings,
        "rope_theta": args.rope_theta,
        "attention_bias": args.attention_bias,
        "norm_type": args.norm_type,
        "deterministic_attention": True,
        "init_device": "cpu",
        "apply_torch_compile_to_projections": args.apply_torch_compile_to_projections,
        "use_custom_rotary_kernel": args.use_custom_rotary_kernel,
        "apply_qk_norm": args.apply_qk_norm,
        "num_krope_heads": args.num_krope_heads,
        # Qwen3Next Gated Delta Net parameters
        "linear_key_head_dim": args.linear_key_head_dim,
        "linear_value_head_dim": args.linear_value_head_dim,
        "linear_conv_kernel_dim": args.linear_conv_kernel_dim,
        "linear_num_key_heads": args.linear_num_key_heads,
        "linear_num_value_heads": args.linear_num_value_heads,
        "linear_use_short_conv": args.linear_use_short_conv,
    }
    config = GigaMixConfig(**config_dict)
    return config


def make_input(args: argparse.Namespace):
    hidden_states = torch.randn(
        args.batch_size,
        args.seq_len,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        requires_grad=True,
    )

    position_ids = (
        torch.arange(
            0, args.seq_len, dtype=torch.long, device=torch.cuda.current_device()
        )
        .unsqueeze(0)
        .expand(args.batch_size, -1)
    )

    attention_mask = torch.ones_like(position_ids, device=torch.cuda.current_device())

    return hidden_states, attention_mask, position_ids


def _instantiate_attention(attn_cls, args: argparse.Namespace):
    config = make_config(args)
    return attn_cls(config).cuda()


def make_attention(kind: str, args: argparse.Namespace):
    factories = {
        "latent": LlamaLatentAttention,
        "qwen3-next": Qwen3NextGatedDeltaNet 
    }

    key = kind.lower()
    if key not in factories:
        raise ValueError(f"Unknown attention kind: {kind}")
    return _instantiate_attention(factories[key], args)


def run_benchmark(
    module, hidden_states, attention_mask, position_ids, args, prefix="latent_attention"
):
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    torch.cuda.memory.reset_peak_memory_stats(device)
    
    def forward_func(inputs):
        return module(
            hidden_states=inputs[0],
            attention_mask=None,  # inputs[1],
            position_ids=inputs[2],
            use_cache=False,
        )[0]

    # Warm up forward pass
    for _ in range(args.warmup_iters):
        _ = forward_func((hidden_states, attention_mask, position_ids))

    torch.cuda.synchronize()
    # Benchmark forward pass
    forward_times = []
    for _ in range(args.iters):
        start_iter = time.time()
        _ = forward_func((hidden_states, attention_mask, position_ids))
        torch.cuda.synchronize()
        end_iter = time.time()
        forward_times.append(end_iter - start_iter)

    forward_total_time = sum(forward_times)
    forward_time_per_iter = forward_total_time / args.iters
    forward_time_std = np.std(forward_times)
    print(f"Average forward time per iteration: {forward_time_per_iter*1000:.2f} ms")
    print(f"Forward standard deviation: {forward_time_std*1000:.2f} ms")

    # Warm up backward pass
    for _ in range(args.warmup_iters):
        # Clear gradients before warmup
        module.zero_grad()
        output = forward_func((hidden_states, attention_mask, position_ids))
        loss = output.sum()
        loss.backward()

    torch.cuda.synchronize()
    # Benchmark backward pass separately
    backward_times = []
    for _ in range(args.iters):
        # Clear gradients before each backward measurement
        module.zero_grad()
        torch.cuda.synchronize()
        
        # Forward pass to get output
        output = forward_func((hidden_states, attention_mask, position_ids))
        loss = output.sum()
        
        # Measure only backward pass time
        torch.cuda.synchronize()
        start_backward = time.time()
        loss.backward()
        torch.cuda.synchronize()
        end_backward = time.time()
        
        backward_times.append(end_backward - start_backward)

    backward_total_time = sum(backward_times)
    backward_time_per_iter = backward_total_time / args.iters
    backward_time_std = np.std(backward_times)
    print(f"Average backward time per iteration: {backward_time_per_iter*1000:.2f} ms")
    print(f"Backward standard deviation: {backward_time_std*1000:.2f} ms")

    # Get memory stats
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB

    try:
        perf_result = benchmark_module(
            hidden_states,
            # (hidden_states, attention_mask, position_ids),
            module,
            args.warmup_iters,
            args.iters,
            trace=args.trace,
            profile_memory=args.profile_memory,
        )

        # torch.distributed.barrier()
        perf_result.print_table()

        if args.trace:
            if args.trace_output_path:
                trace_path = args.trace_output_path
            else:
                trace_path = f"/home/jovyan/dtitov/dump/{prefix}_trace_rank_{get_tp_group_rank()}.json.gz"
            perf_result.profile.export_chrome_trace(trace_path)

        if args.profile_memory:
            if args.memory_profiler_output_path:
                memory_path = args.memory_profiler_output_path
            else:
                memory_path = f"/home/jovyan/dtitov/dump/{prefix}_stacktrace_rank{get_tp_group_rank()}.pickle"
            with open(memory_path, "wb") as f:
                pickle.dump(perf_result.memory_snapshot.snapshot, f)

    except (NameError, AttributeError) as e:
        print(e)
        print("benchmark_module not available for detailed profiling")

    return {
        "forward_time_avg": forward_time_per_iter *1000,
        "forward_time_std": forward_time_std *1000,
        "backward_time_avg": backward_time_per_iter *1000,
        "backward_time_std": backward_time_std *1000,
        "max_reserved_memory": max_memory
    }

def save_results_to_markdown(results, output_path="benchmark_results.md"):
    """
    Save benchmark results to a markdown file with a table.

    Args:
        results (list): List of dictionaries containing benchmark results for each attention type.
        output_path (str): Path to the output markdown file.
    """
    # Create the markdown table header
    markdown_content = """# Benchmark Results

| attn_variant | forward time avg (ms) | forward time std (ms) | backward time avg (ms) | backward time std (ms) | max reserved memory (GB) |
|--------------|-----------------------|-----------------------|------------------------|------------------------|--------------------------|
"""

    # Add each result as a row in the table
    for result in results:
        markdown_content += f"| {result['attn_variant']} | {result['forward_time_avg']:.6f} | {result['forward_time_std']:.6f} | {result['backward_time_avg']:.6f} | {result['backward_time_std']:.6f} | {result['max_reserved_memory']:.2f} |\n"

    # Write the markdown content to the file
    with open(output_path, "w") as f:
        f.write(markdown_content)

    print(f"Benchmark results saved to {output_path}")


def test_qwen3next_gdn_and_mla_comparison():
    seed_everything(42)
    args = parse_args()

    hidden_states, attention_mask, position_ids = make_input(args)

    latent_attention_module = make_attention(kind="latent", args=args).to(torch.bfloat16)
    gated_delta_net_module = make_attention(kind="qwen3-next", args=args).to(torch.bfloat16)

    # Calculate and print number of parameters for LlamaLatentAttention
    latent_params = sum(p.numel() for p in latent_attention_module.parameters() if p.requires_grad)
    print(f"Number of parameters in LlamaLatentAttention: {latent_params:,}")

    print("Benchmarking LlamaLatentAttention")
    torch.cuda.empty_cache()
    gc.collect()
    latent_results = run_benchmark(
        latent_attention_module,
        hidden_states.clone(),
        attention_mask,
        position_ids,
        args,
        "latent_attention",
    )
    latent_results["attn_variant"] = "LlamaLatentAttention"

    torch.cuda.memory.reset_accumulated_memory_stats(0)

    # Calculate and print number of parameters for Qwen3NextGatedDeltaNet
    gdn_params = sum(p.numel() for p in gated_delta_net_module.parameters() if p.requires_grad)
    print(f"Number of parameters in Qwen3NextGatedDeltaNet: {gdn_params:,}")

    print("Benchmarking Qwen3NextGatedDeltaNet")
    torch.cuda.empty_cache()
    gc.collect()
    gdn_results = run_benchmark(
        gated_delta_net_module,
        hidden_states.clone(),
        attention_mask,
        position_ids,
        args,
        "gated_delta_net_attention",
    )
    gdn_results["attn_variant"] = "Qwen3NextGatedDeltaNet"

    # Save results to markdown file
    results = [latent_results, gdn_results]
    save_results_to_markdown(results, f"dump/benchmark_results_{args.benchmark_name}.md")


if __name__ == "__main__":
    test_qwen3next_gdn_and_mla_comparison()