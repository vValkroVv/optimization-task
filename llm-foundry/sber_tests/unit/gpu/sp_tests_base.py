"""
Base utilities for linear attention tests with Sequence Parallelism support.

This module provides common functions and context managers for testing
linear attention implementations with SP (Sequence Parallelism).
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
import composer.utils.dist as composer_dist
from transformers import PretrainedConfig
from torch._C._distributed_c10d import ProcessGroup


ATOL = 1e-4


@dataclass
class TensorDiffStats:
    close: bool
    sim_diff: float
    max_abs_diff: float
    max_rel_diff: float


@dataclass
class ParamGradDiffStats:
    close: bool
    worst_param_name: Optional[str]
    worst_sim_diff: float
    worst_max_abs_diff: float
    worst_max_rel_diff: float
    failure_reason: Optional[str] = None


@dataclass
class CorrectnessResult:
    close: bool
    output: TensorDiffStats
    param_grads: Optional[ParamGradDiffStats] = None


class SPPatches:
    """Context manager that patches composer.utils.dist SP functions."""

    def __init__(self, sp_group: ProcessGroup, rank: int, world_size: int):
        self.sp_group = sp_group
        self.rank = rank
        self.world_size = world_size
        self._patchers = []

    def __enter__(self):
        self._patchers = [
            patch("composer.utils.dist.get_sp_group", return_value=self.sp_group),
            patch("composer.utils.dist.get_sp_group_rank", return_value=self.rank),
            patch("composer.utils.dist.get_sp_group_size", return_value=self.world_size),
        ]
        for p in self._patchers:
            p.__enter__()
        return self

    def __exit__(self, *args: Any):
        for p in reversed(self._patchers):
            p.__exit__(*args)


def setup_distributed(rank: int, world_size: int, port: str = "12357"):
    """Initialize distributed process group for testing."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    try:
        composer_dist.initialize_dist(device="gpu", sp_size=world_size)
    except RuntimeError:
        pass


def cleanup_distributed():
    """Destroy distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def tensor_diff_stats(
    x: torch.Tensor,
    y: torch.Tensor,
    atol: float,
    chunk_size: int = 2_000_000,
) -> TensorDiffStats:
    """Compute similarity-based difference statistics between two tensors."""
    x_flat = x.detach().reshape(-1)
    y_flat = y.detach().reshape(-1)

    numerator = 0.0
    denominator = 0.0
    max_abs_diff = 0.0
    max_rel_diff = 0.0

    for start in range(0, x_flat.numel(), chunk_size):
        end = min(start + chunk_size, x_flat.numel())
        x_chunk = x_flat[start:end].float()
        y_chunk = y_flat[start:end].float()
        diff = (x_chunk - y_chunk).abs()

        max_abs_diff = max(max_abs_diff, diff.max().item())
        denom_chunk = torch.maximum(x_chunk.abs(), y_chunk.abs()).clamp_min_(1e-8)
        max_rel_diff = max(max_rel_diff, (diff / denom_chunk).max().item())

        x_chunk64 = x_chunk.double()
        y_chunk64 = y_chunk.double()
        numerator += (2 * (x_chunk64 * y_chunk64).sum()).item()
        denominator += ((x_chunk64 * x_chunk64).sum() + (y_chunk64 * y_chunk64).sum()).item()

    sim_diff = 0.0 if denominator == 0.0 else 1 - numerator / denominator
    return TensorDiffStats(
        close=sim_diff < atol,
        sim_diff=sim_diff,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
    )


def compare_parameter_grads(
    baseline: torch.nn.Module,
    improved: torch.nn.Module,
    atol: float,
    world_size: int,
) -> ParamGradDiffStats:
    """Compare gradients between two models."""
    baseline_params = {
        name: param for name, param in baseline.named_parameters() if param.requires_grad
    }
    improved_params = {
        name: param for name, param in improved.named_parameters() if param.requires_grad
    }

    baseline_names = set(baseline_params)
    improved_names = set(improved_params)
    if baseline_names != improved_names:
        missing_in_improved = sorted(baseline_names - improved_names)
        missing_in_baseline = sorted(improved_names - baseline_names)
        return ParamGradDiffStats(
            close=False,
            worst_param_name=None,
            worst_sim_diff=float("inf"),
            worst_max_abs_diff=float("inf"),
            worst_max_rel_diff=float("inf"),
            failure_reason=(
                f"parameter mismatch: missing_in_improved={missing_in_improved}, "
                f"missing_in_baseline={missing_in_baseline}"
            ),
        )

    close = True
    worst_param_name = None
    worst_sim_diff = -1.0
    worst_max_abs_diff = 0.0
    worst_max_rel_diff = 0.0

    for name in sorted(baseline_params):
        baseline_grad = baseline_params[name].grad
        improved_grad = improved_params[name].grad

        if baseline_grad is None or improved_grad is None:
            return ParamGradDiffStats(
                close=False,
                worst_param_name=name,
                worst_sim_diff=float("inf"),
                worst_max_abs_diff=float("inf"),
                worst_max_rel_diff=float("inf"),
                failure_reason=(
                    f"missing gradient for `{name}`: "
                    f"baseline_grad_is_none={baseline_grad is None}, "
                    f"improved_grad_is_none={improved_grad is None}"
                ),
            )

        if world_size > 1:
            dist.all_reduce(baseline_grad)
            dist.all_reduce(improved_grad)

        grad_stats = tensor_diff_stats(baseline_grad, improved_grad, atol=atol)
        if not grad_stats.close:
            close = False
        if grad_stats.sim_diff > worst_sim_diff:
            worst_param_name = name
            worst_sim_diff = grad_stats.sim_diff
            worst_max_abs_diff = grad_stats.max_abs_diff
            worst_max_rel_diff = grad_stats.max_rel_diff

    return ParamGradDiffStats(
        close=close,
        worst_param_name=worst_param_name,
        worst_sim_diff=worst_sim_diff,
        worst_max_abs_diff=worst_max_abs_diff,
        worst_max_rel_diff=worst_max_rel_diff,
    )


def generate_test_data(batch_size: int, seq_len: int, hidden_size: int, device: str = "cuda"):
    """Generate test data for linear attention tests."""
    torch.manual_seed(42)
    full_hidden_states = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    torch.manual_seed(43)
    full_target = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16
    )

    return full_hidden_states, full_target


def generate_test_cu_seqlens(
    batch_size: int,
    seq_len: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate fixed-length cumulative sequence lengths for varlen kernels."""
    return torch.arange(
        batch_size + 1,
        device=device,
        dtype=torch.int32,
    ) * seq_len


def compute_test_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    global_numel: int,
) -> torch.Tensor:
    """Compute a globally normalized loss so SP and non-SP gradients are comparable."""
    return (output * target).sum() / global_numel


def check_forward_outputs(output_sp1: torch.Tensor, output_sp2: torch.Tensor, atol: float = ATOL):
    """Compare forward outputs between SP=1 and SP=N using similarity-based check."""
    stats = tensor_diff_stats(output_sp1, output_sp2, atol=atol)

    if not stats.close:
        raise AssertionError(
            (
                f"Outputs mismatch! sim_diff={stats.sim_diff:.6e} >= atol={atol} "
                f"(max_abs_diff={stats.max_abs_diff:.6e}, max_rel_diff={stats.max_rel_diff:.6e})"
            )
        )


def check_input_grads(grad_input_sp1: torch.Tensor, grad_input_sp2: torch.Tensor, atol: float = ATOL):
    """Compare input gradients between SP=1 and SP=N using similarity-based check."""
    stats = tensor_diff_stats(grad_input_sp1, grad_input_sp2, atol=atol)

    if not stats.close:
        raise AssertionError(
            (
                f"Input gradients mismatch! sim_diff={stats.sim_diff:.6e} >= atol={atol} "
                f"(max_abs_diff={stats.max_abs_diff:.6e}, max_rel_diff={stats.max_rel_diff:.6e})"
            )
        )


def check_weight_grads(model_sp2: torch.nn.Module, grad_weights_sp1: dict, atol: float = ATOL):
    """Compare weight gradients between SP=1 and SP=N using similarity-based check."""
    worst_param_name = None
    worst_sim_diff = -1.0
    worst_max_abs_diff = 0.0
    worst_max_rel_diff = 0.0
    all_weights_close = True

    for name, param in model_sp2.named_parameters():
        if name in grad_weights_sp1:
            grad_sp1 = grad_weights_sp1[name]
            grad_sp2 = param.grad

            if grad_sp2 is None:
                continue

            stats = tensor_diff_stats(grad_sp1, grad_sp2, atol=atol)
            if not stats.close:
                all_weights_close = False
                if stats.sim_diff > worst_sim_diff:
                    worst_param_name = name
                    worst_sim_diff = stats.sim_diff
                    worst_max_abs_diff = stats.max_abs_diff
                    worst_max_rel_diff = stats.max_rel_diff

    if not all_weights_close:
        raise AssertionError(
            (
                f"Weight gradients mismatch! worst_param='{worst_param_name}', "
                f"sim_diff={worst_sim_diff:.6e} >= atol={atol} "
                f"(max_abs_diff={worst_max_abs_diff:.6e}, max_rel_diff={worst_max_rel_diff:.6e})"
            )
        )


def run_sp_test_impl(
    attn_class: Any,
    rank: int,
    world_size: int,
    batch_size: int,
    seq_len: int,
    config: PretrainedConfig,
    use_cu_seqlens: bool = False,
):
    """Run SP test implementation comparing SP=1 vs SP=N."""
    setup_distributed(rank, world_size)

    try:
        full_hidden_states, full_target = generate_test_data(
            batch_size, seq_len, config.hidden_size
        )
        full_cu_seqlens = (
            generate_test_cu_seqlens(batch_size, seq_len)
            if use_cu_seqlens
            else None
        )

        grad_input_sp1 = None
        grad_weights_sp1 = {}
        output_sp1 = None
        state_dict = None

        if rank == 0:
            with patch("composer.utils.dist.get_sp_group_size", return_value=1):
                model_sp1 = attn_class(config).to("cuda").to(torch.bfloat16)
                model_sp1.train()

                input_sp1 = full_hidden_states.clone().detach().requires_grad_(True)
                output_sp1, _, _ = model_sp1(
                    input_sp1,
                    cu_seqlens=full_cu_seqlens,
                )

                loss_sp1 = compute_test_loss(
                    output_sp1,
                    full_target,
                    global_numel=full_target.numel(),
                )
                loss_sp1.backward()

                grad_input_sp1 = input_sp1.grad
                for name, param in model_sp1.named_parameters():
                    if param.grad is not None:
                        grad_weights_sp1[name] = param.grad.clone()

                state_dict = model_sp1.state_dict()

        sp_group = dist.new_group(ranks=list(range(world_size)))

        with SPPatches(sp_group, rank, world_size):
            model_sp2 = attn_class(config).to("cuda").to(torch.bfloat16)

        if rank == 0 and state_dict is not None:
            model_sp2.load_state_dict(state_dict)

        for param in model_sp2.parameters():
            dist.broadcast(param.data, src=0)

        model_sp2.train()

        local_hidden_states = (
            torch.tensor_split(full_hidden_states, world_size, dim=1)[rank].clone().detach()
        )
        local_hidden_states.requires_grad = True
        local_target = (
            torch.tensor_split(full_target, world_size, dim=1)[rank].clone().detach()
        )

        with SPPatches(sp_group, rank, world_size):
            local_output_sp2, _, _ = model_sp2(
                local_hidden_states,
                cu_seqlens=full_cu_seqlens,
            )

            loss_sp2 = compute_test_loss(
                local_output_sp2,
                local_target,
                global_numel=full_target.numel(),
            )
            loss_sp2.backward()

        # Replicated parameters need a gradient reduction to match the SP=1 baseline.
        for param in model_sp2.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

        # Local inputs can influence outputs on every SP rank, so each rank only
        # holds a partial input gradient until we sum across the SP group.
        dist.all_reduce(local_hidden_states.grad)

        gathered_grad_input = [
            torch.zeros_like(local_hidden_states.grad) for _ in range(world_size)
        ]
        dist.all_gather(gathered_grad_input, local_hidden_states.grad)
        grad_input_sp2 = torch.cat(gathered_grad_input, dim=1)

        gathered_output = [torch.zeros_like(local_output_sp2) for _ in range(world_size)]
        dist.all_gather(gathered_output, local_output_sp2)
        output_sp2 = torch.cat(gathered_output, dim=1)

        if rank == 0:
            check_forward_outputs(output_sp1, output_sp2)
            check_input_grads(grad_input_sp1, grad_input_sp2)
            check_weight_grads(model_sp2, grad_weights_sp1)

    finally:
        cleanup_distributed()


def run_single_gpu_test_impl(
    attn_class: Any,
    batch_size: int,
    seq_len: int,
    config: PretrainedConfig,
    reference_class: Any = None,
    reference_config: PretrainedConfig | None = None,
    copy_weights_fn: Callable | None = None,
):
    """Run single-GPU test comparing implementation against reference."""
    device = "cuda"
    dtype = torch.bfloat16

    model = attn_class(config).to(device).to(dtype)
    model.reset_parameters()
    model.train()

    hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size, device=device, dtype=dtype
    )

    output, _, _ = model(hidden_states)

    assert not torch.isnan(output).any(), "Output contains NaNs"
    assert not torch.isinf(output).any(), "Output contains Infs"

    loss = output.mean()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"{name}.grad contains NaNs"
            assert not torch.isinf(param.grad).any(), f"{name}.grad contains Infs"

    if reference_class is not None:
        ref_config = reference_config if reference_config is not None else config
        ref_model = reference_class(ref_config).to(device).to(dtype)

        if copy_weights_fn is not None:
            copy_weights_fn(model, ref_model)

        ref_output = ref_model(hidden_states)

        assert torch.allclose(output, ref_output, rtol=1e-3, atol=1e-3), \
            f"Output mismatch with reference: max_diff={torch.abs(output - ref_output).max()}"

        ref_loss = ref_output.mean()
        ref_loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and hasattr(ref_model, name):
                ref_param = getattr(ref_model, name)
                if ref_param.grad is not None:
                    assert torch.allclose(param.grad, ref_param.grad, rtol=1e-2, atol=1e-2), \
                        f"Gradient mismatch for {name}"
