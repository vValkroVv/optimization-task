"""Benchmark the attention row-statistics records against the flash-attn forward.

The kernel recomputes QK^T per tile from the log-partition of a flash-attn call,
so the cost that matters is its time relative to the forward that produced the
log-partition, and, per training step, the share of the step. Forward only, bf16,
causal, MLA geometry (q/k head_dim 192 = 128 nope + 64 rope, v head_dim 128
where flash-attn supports it):

  flash-attn fwd   llmfoundry's flash_attn_import path (FA3 on Hopper), the call
                   the statistics sit next to in training
  row kernel       the program that reduces its rows to partial records
  merge            the merge kernel and the zero-filled buffers around it
  quantiles        exact_quantiles of entn (metric_sampler's prebuilt extension
                   when it is built, its Triton fallback otherwise)
  records          attention_row_stats: the row kernel, merge and quantiles

Reports median times, own FLOP/s (QK^T and PV for flash-attn, QK^T for the
kernels), the share of the flash-attn forward, the autotuned config and every
config's time, and the entropy error against a plain fp32 evaluation.

A call of the records is a handful of launches, and at a small shape the host
needs about as long to issue them as the device needs to run them, so three
costs per call are reported for the model that has the geometry, each with its
share of a training step: eager (one call at a time, as do_bench times it, which
includes launch latency the device waits for), queued (many calls back to back,
the host ahead of the device, as in a training step) and graph replay (one
cudaGraphLaunch of the captured call, the device time alone; the metric-sampler
callback graphs its other computations with GraphedComputation). Under
"gpu kernels" every kernel of one call is listed with its device time.

    python sber_benchmarks/attn_row_stats_benchmark.py
"""

import time

import torch
import triton
import triton.testing
from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.kernels.triton_launch import jit_launches
from metric_sampler.capture.stats.helpers.quantiles import exact_quantiles

from llmfoundry.models.layers import attn_row_stats
from llmfoundry.models.layers.attn_row_stats import (
    RowField,
    attention_row_stats,
    attention_row_values,
)
from llmfoundry.models.utils import flash_attn_import as lf_attn

QK_HEAD_DIM = 192
# FA2 has no unequal q/k and v head dims.
V_HEAD_DIM = 128 if lf_attn.HAS_FA3 else QK_HEAD_DIM
# (name, batch, seq_len, heads, MLA calls per step, seconds per step)
GEOMETRIES = (
    ("L8: B=12 S=4096 H=8", 12, 4096, 8, 6, 1.19),
    ("37B: B=1 S=4096 H=32", 1, 4096, 32, 288, 39.4),
)


def _median_ms(fn):
    fn()  # compile / autotune outside the measured window
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")


def _queued_ms(fn, calls=20):
    """Mean time of `calls` calls issued back to back."""
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(calls):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / calls


def _graphed_ms(fn):
    """Median time of the replay of one captured call, or None when the call is not capturable."""
    if activation_ext.prebuilt() is None:
        # exact_quantiles' Triton fallback copies a host list to the device.
        return None
    fn()  # compile and tune before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return _median_ms(graph.replay)


def _gpu_kernels(fn):
    """(name, device ms) of every kernel one call runs."""
    fn()
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        fn()
        torch.cuda.synchronize()
    return [
        (event.key, event.device_time_total / 1e3 / event.count)
        for event in profile.key_averages()
        if event.device_time_total
    ]


def _issue_ms(fn, calls=200):
    """Host time per call of `fn` issued back to back, which is what the training thread pays."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(calls):
        fn()
    issued = time.perf_counter() - started
    torch.cuda.synchronize()
    return issued / calls * 1e3


def _entropy_error(q, k, lse, rows, scale):
    """Max |ent - plain fp32 evaluation| over the first sequence of head 0."""
    seq_len = q.shape[1]
    s = (q[0, :, 0].float() @ k[0, :, 0].float().T) * scale
    allowed = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
    x = (s - lse[0, 0][:, None]).masked_fill(~allowed, 0.0)
    ent = -(torch.where(allowed, x.exp(), 0.0) * x).sum(-1)
    return (rows[RowField.ENT, 0, :seq_len] - ent).abs().max().item()


def _print_config_timings():
    timings = attn_row_stats._row_stats_kernel.configs_timings
    ranked = sorted(timings.items(), key=lambda item: float(item[1][0]))
    for config, times in ranked:
        print(f"      {float(times[0]):8.3f} ms  {config}")


def run_case(name, bsz, seq_len, heads, calls_per_step, step_seconds):
    torch.manual_seed(0)
    q = (torch.randn(bsz, seq_len, heads, QK_HEAD_DIM, device="cuda") * 2).to(
        torch.bfloat16
    )
    k = (torch.randn(bsz, seq_len, heads, QK_HEAD_DIM, device="cuda") * 2).to(
        torch.bfloat16
    )
    v = torch.randn(
        bsz, seq_len, heads, V_HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    scale = QK_HEAD_DIM**-0.5

    def flash():
        return lf_attn._flash_attn_forward(
            q, k, v, 0.0, scale, True, -1, -1, 0.0, None, False
        )

    lse = flash()[1]

    def kernel():
        return attn_row_stats._launch(q, k, lse, scale, None, None, False)

    launch = kernel()

    def quantiles():
        return exact_quantiles(launch.entn, launch.valid.view(torch.bool))

    def records():
        return attention_row_stats(q, k, lse, scale)

    flash_ms = _median_ms(flash)
    kernel_ms = _median_ms(kernel)
    quantiles_ms = _median_ms(quantiles)
    records_ms = _median_ms(records)
    queued_ms = _queued_ms(records)
    graphed_ms = _graphed_ms(records)
    pairs = bsz * heads * seq_len * (seq_len + 1) / 2
    qk_flops = 2 * pairs * QK_HEAD_DIM

    print(f"\n== {name}: q/k head_dim {QK_HEAD_DIM}, v head_dim {V_HEAD_DIM}")
    print(f"   {'method':<18}{'ms':>9}{'own TFLOP/s':>13}{'vs flash fwd':>14}")
    for label, ms, flops in (
        ("flash-attn fwd", flash_ms, 2 * pairs * (QK_HEAD_DIM + V_HEAD_DIM)),
        ("row kernel", kernel_ms, qk_flops),
        ("merge + quantiles", records_ms - kernel_ms, None),
        ("  quantiles alone", quantiles_ms, None),
        ("records, eager", records_ms, qk_flops),
        ("records, queued", queued_ms, qk_flops),
        ("records, graph", graphed_ms, qk_flops),
    ):
        if ms is None:
            print(f"   {label:<18}{'(needs the extension)':>9}")
            continue
        own = "" if flops is None else f"{flops / ms / 1e9:.1f}"
        print(f"   {label:<18}{ms:9.3f}{own:>13}{ms / flash_ms:14.2f}")
    for label, ms in (
        ("eager", records_ms),
        ("queued", queued_ms),
        ("graph", graphed_ms),
    ):
        if ms is not None:
            per_step_ms = calls_per_step * ms
            share = per_step_ms / (step_seconds * 1e3)
            print(
                f"   per step, {label}: {calls_per_step} calls x {ms:.3f} ms = "
                f"{per_step_ms:.1f} ms = {share:.2%} of {step_seconds} s"
            )
    kernels = _gpu_kernels(records)
    print(f"   gpu kernels of one call ({sum(ms for _, ms in kernels):.3f} ms busy):")
    for kernel_name, ms in sorted(kernels, key=lambda item: -item[1]):
        print(f"      {ms:8.3f} ms  {kernel_name[:70]}")
    direct_ms = _issue_ms(records)
    with jit_launches():
        jit_ms = _issue_ms(records)
    print(
        f"   host time to issue a call: {direct_ms:.3f} ms directly launched, "
        f"{jit_ms:.3f} ms through the JIT call"
    )
    best = attn_row_stats._row_stats_kernel.best_config
    print(f"   autotuned config: {best}")
    _print_config_timings()
    rows = attention_row_values(q, k, lse, scale)
    error = _entropy_error(q, k, lse, rows, scale)
    print(f"   max |ent error| vs fp32 torch (head 0): {error:.2e} nats")


def main():
    print(
        torch.cuda.get_device_name(),
        "torch",
        torch.__version__,
        "triton",
        triton.__version__,
    )
    print(
        "flash-attn:",
        "FA3 (llmfoundry flash_attn_import)" if lf_attn.HAS_FA3 else "FA2",
    )
    print("exact_quantiles extension built:", activation_ext.prebuilt() is not None)
    for geometry in GEOMETRIES:
        run_case(*geometry)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
