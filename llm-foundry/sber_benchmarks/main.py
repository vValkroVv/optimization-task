import argparse
import os
import pickle

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from composer.utils.dist import (
    initialize_dist,
    get_tp_group,
    get_tp_group_size,
    get_tp_group_rank,
)
from composer.utils.device import get_device

from llmfoundry.models.gigar import GigarConfig
from llmfoundry.models.layers.fc import ColumnParallelLinear
from llmfoundry.models.layers.ffn import ParallelMLP

from benchmark import benchmark_module


RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE


def setup_dist():
    initialize_dist(get_device(None), tp_size=2, sp_size=1, timeout=999999)
    torch.distributed.barrier()
    torch.cuda.set_device(get_tp_group_rank())
    torch.cuda.synchronize()
    enable_symm_mem_for_group(get_tp_group().group_name)
    torch.manual_seed(42 + RANK)


def print_diffs(lhs: torch.Tensor, rhs: torch.Tensor, name: str = "") -> None:
    assert lhs.shape == rhs.shape
    mean_diff = (lhs - rhs).abs().mean().item()
    max_diff = (lhs - rhs).abs().max().item()
    print(f"{name}: mean_diff={'{:.8f}'.format(mean_diff)}, max_diff={'{:.8f}'.format(max_diff)}")


def make_input(args: argparse.Namespace, baseline: bool | None = None) -> torch.Tensor:
    baseline = args.baseline if baseline is None else baseline
    
    batch_size: int = args.batch_size
    seq_len: int = args.seq_len
    tp_size: int = get_tp_group_size()
    assert batch_size % tp_size == 0
    
    batch_size = batch_size // tp_size if not baseline else batch_size
    return torch.randn(
        batch_size,
        seq_len,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        requires_grad=True,
    )


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_config(args: argparse.Namespace, baseline: bool | None = None) -> GigarConfig:
    baseline = baseline if baseline is not None else args.baseline
    return GigarConfig(
        num_hidden_layers=3,
        deterministic_attention=True,
        attention_type="LlamaPackedAttention",
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        fused_mlp=False,
        init_device="cpu",
        tp_size=get_tp_group_size(),
        num_attention_heads=64,
        num_key_value_heads=8,
        dtype=torch.bfloat16,
        async_tensor_paralellism=not baseline,
    )


def make_parallel_mlp(
    args: argparse.Namespace, baseline: bool | None = None
) -> ParallelMLP:
    config = make_config(args, baseline)
    return ParallelMLP(config).cuda()


def unshard_tp_tensor_like(
    reference: torch.Tensor,
    shard: torch.Tensor,
) -> torch.Tensor:
    full = torch.empty_like(reference)
    torch.distributed.all_gather_into_tensor(full, shard, get_tp_group())
    return full


def test_column_parallel():
    setup_dist()
    args = parse_args()

    tp_size = get_tp_group_size()

    async_tp_input = torch.randn(
        args.batch_size // tp_size,
        args.seq_len,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        requires_grad=True,
    )

    baseline_input = torch.empty(
        args.batch_size,
        args.seq_len,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        requires_grad=True,
    )
    torch.distributed.all_gather_into_tensor(
        baseline_input, async_tp_input, get_tp_group()
    )

    seed_everything(42)
    baseline_module = make_parallel_mlp(args, baseline=True)
    seed_everything(42)
    async_tp_module = make_parallel_mlp(args, baseline=False)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        baseline_out = baseline_module(baseline_input)
        async_tp_out = async_tp_module(async_tp_input)

        grad_output_async_tp = torch.randn_like(async_tp_out)
        baseline_grad_output = unshard_tp_tensor_like(baseline_out, grad_output_async_tp)

        baseline_out.backward(baseline_grad_output)
        async_tp_out.backward(grad_output_async_tp)

    full_async_tp_out = unshard_tp_tensor_like(baseline_out, async_tp_out)
    assert torch.allclose(baseline_out, full_async_tp_out)

    async_tp_grad_input = unshard_tp_tensor_like(baseline_input.grad, async_tp_input.grad)
    assert torch.allclose(baseline_input.grad, async_tp_grad_input)

    for (baseline_name, baseline_param), (flux_name, flux_param) in zip(
        baseline_module.named_parameters(),
        async_tp_module.named_parameters()
    ):
        assert baseline_name == flux_name
        torch.allclose(baseline_param.grad, flux_param.grad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # (8192, 8192, 32768)  90B ours
    # (4096 12288 49152)

    parser.add_argument("--batch_size", type=int, default=4, required=False)
    parser.add_argument("--seq_len", type=int, default=8192, required=False)
    parser.add_argument("--hidden_size", type=int, default=8192, required=False)
    parser.add_argument(
        "--intermediate_size", type=int, default=16384, required=False
    )
    parser.add_argument("--warmup_iters", type=int, default=10, required=False)
    parser.add_argument("--iters", type=int, default=100, required=False)
    parser.add_argument("--baseline", action="store_true", default=True)
    parser.add_argument("--trace", action="store_true", default=False)
    parser.add_argument("--profile_memory", action="store_true", default=False)

    args = parser.parse_args()

    tp_size = get_tp_group_size()
    assert args.seq_len % tp_size == 0
    assert args.hidden_size % tp_size == 0
    assert args.intermediate_size % tp_size == 0

    return args


def run_benchmark():
    setup_dist()

    args = parse_args()
    baseline: bool = args.baseline
    trace: bool = args.trace
    profile_memory: bool = args.profile_memory

    module = make_parallel_mlp(args)
    input_ = make_input(args)

    torch.distributed.barrier()

    perf_result = benchmark_module(
        input_,
        module,
        args.warmup_iters,
        args.iters,
        trace=True,
        profile_memory=profile_memory,
    )

    torch.distributed.barrier()
    perf_result.print_table()

    if trace:
        prefix = "baseline" if baseline else "async_tp"
        perf_result.profile.export_chrome_trace(
            f"dump/{prefix}_trace_rank_{get_tp_group_rank()}.json.gz"
        )

    if profile_memory:
        prefix = "baseline" if baseline else "async_tp"
        with open(
            f"dump/{prefix}_stacktrace_rank{get_tp_group_rank()}.pickle", "wb"
        ) as f:
            pickle.dump(perf_result.memory_snapshot.snapshot, f)


if __name__ == "__main__":
    run_benchmark()
