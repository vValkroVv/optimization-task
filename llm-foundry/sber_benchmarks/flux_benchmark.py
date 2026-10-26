import pickle
import argparse
import os
from functools import partial
from omegaconf import OmegaConf

import torch
from torch import nn
import flux
import numpy as np
import torch.distributed
from transformers import PreTrainedTokenizer

from llmfoundry.models.gigar import GigarConfig
from llmfoundry.models.layers.fc import ColumnParallelLinear, RowParallelLinear
from llmfoundry.models.layers.ffn import ParallelMLP
from llmfoundry.models.layers.blocks import LlamaDecoderLayer
from llmfoundry.models.gigar import ComposerGigarCausalLM


from composer.utils.dist import (
    initialize_dist,
    get_tp_group,
    get_tp_group_size,
    get_tp_group_rank,
)
from composer.utils.device import get_device

from benchmark import benchmark_module


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
    initialize_dist(get_device(None), tp_size=2, sp_size=1, timeout=999999)
    torch.distributed.barrier()
    torch.cuda.set_device(get_tp_group_rank())
    flux.init_flux_shm(get_tp_group())
    torch.cuda.synchronize()


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


def print_diffs(lhs: torch.Tensor, rhs: torch.Tensor, name: str = "") -> None:
    assert lhs.shape == rhs.shape
    mean_diff = (lhs - rhs).abs().mean().item()
    max_diff = (lhs - rhs).abs().max().item()
    print(f"{name}: mean_diff={'{:.8f}'.format(mean_diff)}, max_diff={'{:.8f}'.format(max_diff)}")


def unshard_tp_tensor_like(
    reference: torch.Tensor,
    shard: torch.Tensor,
) -> torch.Tensor:
    full = torch.empty_like(reference)
    torch.distributed.all_gather_into_tensor(full, shard, get_tp_group())
    return full


def make_config(args: argparse.Namespace, baseline: int | None = None) -> GigarConfig:
    baseline = baseline if baseline is not None else args.baseline

    tp_size = get_tp_group_size()

    if baseline:
        flux_tp_split_dim = None
    else:
        flux_tp_split_dim = 0 if args.batch_size >= tp_size else 1

    print(f"{flux_tp_split_dim=}")

    return GigarConfig(
        num_hidden_layers=3,
        deterministic_attention=True,
        attention_type="LlamaPackedAttention",
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        use_flux=not baseline,
        flux_tp_split_dim=flux_tp_split_dim,
        dtype=torch.bfloat16,
        fused_mlp=False,
        init_device="cpu",
        tp_size=get_tp_group_size(),
        num_attention_heads=64,
        num_key_value_heads=8,
    )


def make_parallel_mlp(
    args: argparse.Namespace, baseline: bool | None = None
) -> ParallelMLP:
    config = make_config(args, baseline)
    return ParallelMLP(config).cuda()


def make_llama_decoder_layer(args: argparse.Namespace, baseline: bool | None = None) -> LlamaDecoderLayer:
    config = make_config(args, baseline=baseline)
    return LlamaDecoderLayer(config).cuda()


def make_input(args: argparse.Namespace, baseline: bool | None = None) -> torch.Tensor:
    batch_size: int = args.batch_size
    seq_len: int = args.seq_len
    tp_size: int = get_tp_group_size()

    baseline = baseline if baseline is not None else args.baseline

    if not baseline:
        if batch_size < tp_size:
            batch_size = 1
            seq_len = seq_len // tp_size
        else:
            batch_size = batch_size // tp_size
            seq_len = seq_len

    return torch.randn(
        batch_size,
        seq_len,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        requires_grad=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # (8192, 8192, 32768)  90B ours
    # (4096 12288 49152)

    parser.add_argument("--batch_size", type=int, default=2, required=False)
    parser.add_argument("--seq_len", type=int, default=8192, required=False)
    parser.add_argument("--hidden_size", type=int, default=8192, required=False)
    parser.add_argument(
        "--intermediate_size", type=int, default=32768, required=False
    )
    parser.add_argument("--warmup_iters", type=int, default=10, required=False)
    parser.add_argument("--iters", type=int, default=10, required=False)
    parser.add_argument("--baseline", action="store_true", default=True)
    parser.add_argument("--trace", action="store_true", default=True)
    parser.add_argument("--profile_memory", action="store_true", default=True)

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

    module = make_llama_decoder_layer(args)
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
        prefix = "baseline" if baseline else "flux"
        perf_result.profile.export_chrome_trace(
            f"dump/{prefix}_trace_rank_{get_tp_group_rank()}.json.gz"
        )

    if profile_memory:
        prefix = "baseline" if baseline else "flux"
        with open(
            f"dump/{prefix}_stacktrace_rank{get_tp_group_rank()}.pickle", "wb"
        ) as f:
            pickle.dump(perf_result.memory_snapshot.snapshot, f)


def test_parallel_mlp_flux():
    setup_dist()

    args = parse_args()

    baseline_input = make_input(args, baseline=True)
    flux_input = make_input(args, baseline=False)

    torch.distributed.all_gather_into_tensor(
        baseline_input, flux_input, group=get_tp_group()
    )

    seed_everything(0)
    baseline_model = nn.Sequential(
        make_parallel_mlp(args, baseline=True),
        nn.ReLU(),
        make_parallel_mlp(args, baseline=True),
        nn.ReLU(),
        make_parallel_mlp(args, baseline=True),
    )
    seed_everything(0)
    flux_model = nn.Sequential(
        make_parallel_mlp(args, baseline=False),
        nn.ReLU(),
        make_parallel_mlp(args, baseline=False),
        nn.ReLU(),
        make_parallel_mlp(args, baseline=False),
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        baseline_out = baseline_model(baseline_input)
        flux_out = flux_model(flux_input)

        grad_output_flux = torch.randn_like(flux_out)
        baseline_grad_output = unshard_tp_tensor_like(baseline_out, grad_output_flux)

        baseline_out.backward(baseline_grad_output)
        flux_out.backward(grad_output_flux)

    full_flux_output = unshard_tp_tensor_like(baseline_out, flux_out)
    print_diffs(full_flux_output, baseline_out, "outputs")
    assert torch.allclose(full_flux_output, baseline_out)

    flux_grad_input = unshard_tp_tensor_like(baseline_input.grad, flux_input.grad)
    print_diffs(flux_grad_input, baseline_input.grad, "grad input")
    assert torch.allclose(baseline_input.grad, flux_grad_input)

    for (baseline_name, baseline_param), (flux_name, flux_param) in zip(
        baseline_model.named_parameters(), flux_model.named_parameters()
    ):
        assert baseline_name == flux_name
        print_diffs(baseline_param.grad, flux_param.grad, f"grad {baseline_name}")
        torch.allclose(baseline_param.grad, flux_param.grad)


def column_parallel_fuck_around():
    setup_dist()

    args = parse_args()

    baseline_input = make_input(args, baseline=True)
    flux_input = make_input(args, baseline=False)

    torch.distributed.all_gather_into_tensor(
        baseline_input, flux_input, group=get_tp_group()
    )

    config = make_config(args, baseline=False)
    module = ColumnParallelLinear(
        config.hidden_size,
        config.intermediate_size,
        config=config,
        bias=False,
        skip_bias_add=True,
    ).cuda()

    input_1 = make_input(args, baseline=True) * 10
    input_2 = make_input(args, baseline=True)

    torch.distributed.barrier()
    if get_tp_group_rank() == 0: print(".max after both forward")

    torch.distributed.barrier()
    out_1, _ = module(input_1)
    out_2, _ = module(input_2)

    print(out_1.max().item())
    print(out_2.max().item())

    torch.distributed.barrier()

    if get_tp_group_rank() == 0: print(".max interleaved")

    torch.distributed.barrier()
    out_1, _ = module(input_1)
    print(out_1.max().item())
    out_2, _ = module(input_2)
    print(out_2.max().item())

    assert out_1.data_ptr() != out_2.data_ptr()


def row_parallel_fuck_around():
    setup_dist()

    args = parse_args()

    baseline_input = make_input(args, baseline=True)
    flux_input = make_input(args, baseline=False)

    torch.distributed.all_gather_into_tensor(
        baseline_input, flux_input, group=get_tp_group()
    )

    tp_size: int = get_tp_group_size()
    if args.batch_size >= tp_size:
        batch_size = args.batch_size // tp_size
        seq_len = args.seq_len
    else:
        batch_size = args.batch_size
        seq_len = args.seq_len // tp_size

    input_ = torch.randn(
        (
            batch_size,
            seq_len,
            args.intermediate_size // get_tp_group_size(),
        ),
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
    )

    config = make_config(args, baseline=False)
    module = RowParallelLinear(
        config.intermediate_size,
        config.hidden_size,
        config=config,
        input_is_parallel=True,
        bias=False,
        skip_bias_add=True,
    ).cuda()

    out_1, _ = module(input_)
    out_2, _ = module(input_)

    assert out_1.data_ptr() != out_2.data_ptr()


if __name__ == "__main__":
    run_benchmark()
