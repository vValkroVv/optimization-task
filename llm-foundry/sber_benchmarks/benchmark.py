from typing import Any

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from composer.utils import dist

from contextlib import nullcontext


@dataclass
class SnapshotHandle:
    snapshot: Any
    max_reserved_memory: int


@contextmanager
def record_memory_stacks(max_entries: int = 100_000):
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    handle = SnapshotHandle(None, 0)
    yield handle
    handle.snapshot = torch.cuda.memory._snapshot()
    handle.max_reserved_memory = torch.cuda.max_memory_reserved()
    torch.cuda.memory._record_memory_history(enabled=None)


@dataclass
class PerfResult:
    forward_time: float
    backward_time: float
    profile: torch.profiler.profile
    memory_snapshot: Any

    def print_table(self):
        if (dist.get_tp_group_rank() == 0) or (dist.get_tp_group_rank() is None):
            header = f"{'Rank'} | {'Forward Time (ms)':<18} | {'Backward Time (ms)':<19} | {'Max Reserved Memory (GB)':<24}"
            print(header)
            print('-' * len(header))
        
        forward_ms = self.forward_time
        backward_ms = self.backward_time

        if self.memory_snapshot:
            max_mem_gb = self.memory_snapshot.max_reserved_memory / (1 << 30)
            mem_str = f"{max_mem_gb:.2f}"
        else:
            mem_str = "N/A"

        stats = f"{dist.get_tp_group_rank()} | {forward_ms:<18.2f} | {backward_ms:<19.2f} | {mem_str:<24}"
        print(stats, flush=True)

def make_profiler(profile_memory: bool) -> torch.profiler.profile:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=profile_memory,
        with_stack=profile_memory,
    )


def benchmark_module(
    input_: torch.Tensor,
    module: nn.Module,
    warmup_iters: int,
    iters: int,
    trace: bool = False,
    profile_memory: bool = False,
) -> PerfResult:
    forward_start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(iters)
    ]
    forward_end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(iters)
    ]
    backward_start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(iters)
    ]
    backward_end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(iters)
    ]

    prof_ctx = make_profiler(profile_memory) if trace else nullcontext()
    mem_ctx = record_memory_stacks()

    if input_.is_floating_point():
        inputs = [
            torch.randn_like(input_, requires_grad=input_.requires_grad)
            for _ in range(iters)
        ]
    else:
        inputs = [
            torch.randint_like(input_, torch.max(input_))
            for _ in range(iters)
        ]

    for _ in range(warmup_iters):
        x = (
            torch.randn_like(input_, requires_grad=input_.requires_grad)
            if input_.is_floating_point()
            else torch.randint_like(input_, torch.max(input_))
        )
        out, *_ = module(x)
        grad_output = torch.zeros_like(out)
        out.backward(grad_output)

    with prof_ctx, mem_ctx as mem_snapshot:
        for i in range(iters):
            x = inputs[i]
            forward_start_events[i].record()
            out, *_ = module(x)
            forward_end_events[i].record()
            grad_output = torch.randn_like(out)
            backward_start_events[i].record()
            out.backward(grad_output)
            backward_end_events[i].record()

    forward_times = list()
    backward_times = list()
    for i in range(iters):
        forward_end_events[i].synchronize()
        backward_end_events[i].synchronize()
        forward_times.append(
            forward_start_events[i].elapsed_time(forward_end_events[i])
        )
        backward_times.append(
            backward_start_events[i].elapsed_time(backward_end_events[i])
        )

    return PerfResult(
        forward_time=sum(forward_times) / iters,
        backward_time=sum(backward_times) / iters,
        profile=prof_ctx,
        memory_snapshot=mem_snapshot,
    )
