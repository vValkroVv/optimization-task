from __future__ import annotations

import argparse
import contextlib
import itertools
import sys
import time
from dataclasses import dataclass

import torch
import triton

from metric_sampler.capture.kernels import activation_ext, triton_launch
from metric_sampler.capture.kernels.attention_kernel import (
    launch_gate_stats_kernel,
    launch_linear_stats_kernel,
    launch_softmax_stats_kernel,
)
from metric_sampler.capture.recorder import ResolvedSlot
from metric_sampler.capture.stats import attention
from metric_sampler.capture.stats.attention import AttentionStats
from metric_sampler.contract.metrics import ValueType

_SLOT_NAMES = (
    "attn/softmax/lse_stats", "attn/softmax/qkv_stats", "attn/softmax/gate_stats",
    "attn/linear/gate_stats", "attn/linear/norm_stats",
)
_TIMING_REPEATS = 100
_TIMING_WARMUP = 10
_RTOL = 5e-6
_SWEEP_TILES = (16, 32, 64)
_SWEEP_WARPS = (4, 8)
_SWEEP_PROGRAMS = (528, 1056, 2048)
# Unused words between the slots of the fake recorder: a write past a slot shows in them.
_SLOT_GAP_WORDS = 4
# GPU ms per observation the 8k rows x 16 heads shape is aimed at.
_TARGET_SHAPE = "MLA 8k rows, 16 heads"
_TARGET_MS = {"softmax q/k/v/lse": 0.080, "softmax gate": 0.040, "linear beta/g/v/o": 0.080}


@dataclass
class _Shape:
    name: str
    rows: int
    heads: int
    nope: int
    rope: int
    value_dim: int
    dtype: torch.dtype


_SHAPES = (
    _Shape("odd dims, fp32", 37, 3, 5, 2, 4, torch.float32),
    _Shape("one row", 1, 4, 8, 8, 8, torch.float32),
    _Shape("tile boundary, bf16", 64, 8, 32, 16, 32, torch.bfloat16),
    _Shape("MLA 192/128, fp32, many blocks", 2050, 6, 128, 64, 128, torch.float32),
    _Shape("MLA 192/128, bf16, many blocks", 8192, 16, 128, 64, 128, torch.bfloat16),
)
_PRODUCTION_SHAPES = (
    _Shape(_TARGET_SHAPE, 8192, 16, 128, 64, 128, torch.bfloat16),
    _Shape("MLA 16k rows, 64 heads", 16384, 64, 128, 64, 128, torch.bfloat16),
    _Shape("MLA 1k rows, 8 heads", 1024, 8, 128, 64, 128, torch.bfloat16),
)


class _FakeRecorder:
    """The three recorder methods `AttentionStats` uses, over one float32 buffer."""

    def __init__(self, heads: int, device: torch.device) -> None:
        self.layout: dict[tuple[str, int, int], ResolvedSlot] = {}
        widths = {"attn/softmax/lse_stats": 1 + 3 * heads, "attn/softmax/qkv_stats": 1 + 8 * heads,
                  "attn/softmax/gate_stats": 1 + 3 * heads, "attn/linear/gate_stats": 1 + 7 * heads,
                  "attn/linear/norm_stats": 1 + 3 * heads, "attn/linear/decay_params": 2 * heads}
        offset = 0
        for microbatch in (0, 1):
            for name in _SLOT_NAMES:
                self.layout[name, 0, microbatch] = ResolvedSlot(
                    offset, widths[name], len(self.layout), False, ValueType.FLOAT32
                )
                offset += widths[name] + _SLOT_GAP_WORDS
        self.layout["attn/linear/decay_params", 0, -1] = ResolvedSlot(
            offset, widths["attn/linear/decay_params"], len(self.layout), False, ValueType.FLOAT32)
        self.buffer = torch.zeros(offset + 2 * heads + _SLOT_GAP_WORDS, device=device)

    def resolve(self, name: str, layer: int, microbatch: int) -> ResolvedSlot:
        return self.layout[name, layer, microbatch]

    def mark_recorded(self, indices) -> None:
        pass

    def float_buffer(self) -> torch.Tensor:
        return self.buffer

    def view(self, slot: ResolvedSlot) -> torch.Tensor:
        return self.buffer[slot.offset : slot.offset + slot.width]


def _softmax_inputs(shape: _Shape, device: torch.device, seed: int, nonfinite: bool):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, heads = shape.rows, shape.heads
    query, key = (torch.randn(rows, heads, shape.nope + shape.rope, generator=generator) * 3 for _ in range(2))
    value = torch.randn(rows, heads, shape.value_dim, generator=generator)
    lse = torch.randn(heads, rows, generator=generator) * 2 + 8
    gate = torch.randn(rows, heads, shape.value_dim, generator=generator)
    if nonfinite:
        query[rows // 2, heads - 1, shape.nope] = float("nan")
        key[0, 0, 0] = float("inf")
        value[rows - 1, 0, shape.value_dim - 1] = float("-inf")
        lse[heads - 1, rows // 3] = float("-inf")
        gate[rows // 2, 0, 0] = float("nan")
    beta = torch.rand(rows, heads, generator=generator)
    log_decay = torch.rand(rows, heads, generator=generator) * 4 - 3
    output = torch.randn(rows, heads, shape.value_dim, generator=generator)
    if nonfinite:
        beta[rows // 2, 0] = float("nan")
        log_decay[0, heads - 1] = float("-inf")
        output[rows - 1, heads - 1, 0] = float("inf")
    tensors = [tensor.to(shape.dtype) if tensor is not lse else tensor
               for tensor in (query, key, value, lse, gate, beta, log_decay, value.clone(), output)]
    return [tensor.to(device) for tensor in tensors]


def _observe(
    stats: AttentionStats, inputs, shape: _Shape, microbatch: int, dense_batch: int | None, a_log, dt_bias,
) -> None:
    query, key, value, lse, gate, beta, log_decay, linear_value, output = inputs
    if dense_batch is not None:
        sequence_length = shape.rows // dense_batch

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(dense_batch, sequence_length, *tensor.shape[1:])

        flash_lse = lse.reshape(shape.heads, dense_batch, sequence_length).transpose(0, 1).contiguous()
        query, key, value, gate, lse = split(query), split(key), split(value), split(gate), flash_lse.transpose(0, 1)
        beta, log_decay, linear_value, output = split(beta), split(log_decay), split(linear_value), split(output)
    stats.observe_softmax(0, microbatch, query, key, value, lse, shape.nope)
    stats.observe_softmax_gate(0, microbatch, gate)
    stats.observe_linear(0, microbatch, beta, log_decay, linear_value, output, a_log, dt_bias)


def _compare(label: str, reference: _FakeRecorder, got: _FakeRecorder) -> list[str]:
    problems = []
    got_buffer, reference_buffer = got.buffer.cpu(), reference.buffer
    covered = torch.zeros(reference_buffer.numel(), dtype=torch.bool)
    for (name, _, microbatch), slot in reference.layout.items():
        span = slice(slot.offset, slot.offset + slot.width)
        covered[span] = True
        want, have = reference_buffer[span], got_buffer[span]
        if name != "attn/linear/decay_params" and want[0] != have[0]:
            problems.append(f"{label} {name} mb{microbatch}: rows {have[0].item()} != {want[0].item()}")
        if torch.isnan(want).tolist() != torch.isnan(have).tolist():
            problems.append(f"{label} {name} mb{microbatch}: NaN positions differ")
            continue
        try:
            torch.testing.assert_close(have, want, rtol=_RTOL, atol=0, equal_nan=True)
        except AssertionError as error:
            problems.append(f"{label} {name} mb{microbatch}: "
                            f"{str(error).splitlines()[0]} {str(error).splitlines()[-2:]}")
    if got_buffer[~covered].any():
        problems.append(f"{label}: wrote outside its slots")
    return problems


def _check_correctness(device: torch.device) -> list[str]:
    problems = []
    cpu = torch.device("cpu")
    for shape in _SHAPES:
        a_log, dt_bias = torch.randn(shape.heads), torch.linspace(-7, 25, shape.heads)
        for nonfinite in (False, True):
            for dense_batch in (None, 1, 2):
                if dense_batch is not None and shape.rows % dense_batch:
                    continue
                label = f"{shape.name} nonfinite={nonfinite} dense_batch={dense_batch}"
                reference, got = _FakeRecorder(shape.heads, cpu), _FakeRecorder(shape.heads, device)
                for stats, recorder, where in ((AttentionStats(reference), reference, cpu),
                                               (AttentionStats(got), got, device)):
                    for step, seed in enumerate((1, 2)):
                        if step:
                            recorder.buffer.zero_()
                            stats.begin_step(False)
                        inputs = _softmax_inputs(shape, where, seed, nonfinite)
                        for microbatch in (0, 1):
                            # Twice the same slot in one step: the second observation merges into the first.
                            for _ in range(2):
                                _observe(stats, inputs, shape, microbatch, dense_batch,
                                         a_log.to(where), dt_bias.to(where))
                problems.extend(_compare(label, reference, got))
    return problems


def _launch_path(direct: bool):
    """Launches through the compiled kernels (when the Triton version allows it) or through the JIT call."""
    return contextlib.nullcontext() if direct else triton_launch.jit_launches()


def _launchers(shape: _Shape, inputs, recorder: _FakeRecorder, a_log: torch.Tensor, dt_bias: torch.Tensor):
    """(label, bytes read, launch(**tuning)) of the three kernels on `inputs`, writing the slots of layer 0,
    microbatch 0 of `recorder`; `tuning` is any of tile_rows, num_warps and target_programs."""
    query, key, value, lse, gate, beta, log_decay, linear_value, output = inputs
    layout, buffer = recorder.layout, recorder.buffer
    element = query.element_size()
    rows_heads = shape.rows * shape.heads
    return (
        ("softmax q/k/v/lse", rows_heads * ((2 * (shape.nope + shape.rope) + shape.value_dim) * element + 4),
         lambda **tuning: launch_softmax_stats_kernel(
             query, key, value, lse, buffer, layout[_SLOT_NAMES[0], 0, 0].offset, layout[_SLOT_NAMES[1], 0, 0].offset,
             shape.nope, 0, **tuning)),
        ("softmax gate", rows_heads * shape.value_dim * element,
         lambda **tuning: launch_gate_stats_kernel(gate, buffer, layout[_SLOT_NAMES[2], 0, 0].offset, 0, **tuning)),
        ("linear beta/g/v/o", rows_heads * (2 * shape.value_dim * element + 2 * element),
         lambda **tuning: launch_linear_stats_kernel(
             beta, log_decay, linear_value, output, a_log, dt_bias, buffer, layout[_SLOT_NAMES[3], 0, 0].offset,
             layout[_SLOT_NAMES[4], 0, 0].offset, layout["attn/linear/decay_params", 0, -1].offset, 0, **tuning)),
    )


def _check_determinism(device: torch.device) -> list[str]:
    """The words of the same observation do not change from one run to the next, down to the last bit."""
    shape = _SHAPES[-1]
    a_log, dt_bias = torch.randn(shape.heads, device=device), torch.linspace(-7, 25, shape.heads, device=device)
    inputs = _softmax_inputs(shape, device, 3, True)
    runs = []
    for _ in range(8):
        recorder = _FakeRecorder(shape.heads, device)
        _observe(AttentionStats(recorder), inputs, shape, 0, None, a_log, dt_bias)
        runs.append(recorder.buffer.cpu().view(torch.int32))
    if all(torch.equal(runs[0], run) for run in runs[1:]):
        return []
    return ["the same observation recorded different words in different runs"]


def _check_launch_paths(device: torch.device) -> list[str]:
    """The same observations, three of each so that the later ones are direct launches, record the same
    words, bit for bit, whichever way they are launched."""
    problems = []
    for shape in _SHAPES:
        a_log, dt_bias = torch.randn(shape.heads, device=device), torch.linspace(-7, 25, shape.heads, device=device)
        inputs = _softmax_inputs(shape, device, 7, True)
        for dense_batch in (None, 1):
            words = {}
            for direct in (False, True):
                with _launch_path(direct):
                    recorder = _FakeRecorder(shape.heads, device)
                    stats = AttentionStats(recorder)
                    for microbatch in (0, 1):
                        for _ in range(3):
                            _observe(stats, inputs, shape, microbatch, dense_batch, a_log, dt_bias)
                words[direct] = recorder.buffer.cpu().view(torch.int32)
            if not torch.equal(words[False], words[True]):
                problems.append(f"{shape.name} dense_batch={dense_batch}: "
                                "direct launches recorded other words than the JIT call")
    return problems


def _check_linear_state(device: torch.device) -> list[str]:
    """The extension's observe_linear_state against the equivalent torch operations, on the same device: a
    first observation of a slot and a second one merged into it must leave the same words to the last bit."""
    problems = []
    extension = activation_ext.prebuilt() or activation_ext.build()
    generator = torch.Generator(device=device).manual_seed(181)
    for documents, heads, key_dim, value_dim, dtype, nonfinite in (
            (1, 32, 128, 128, torch.float32, False), (5, 32, 128, 128, torch.bfloat16, False),
            (3, 4, 16, 8, torch.float32, True)):
        got = torch.zeros(1 + 2 * heads, device=device)
        want = torch.zeros(1 + 2 * heads, device=device)
        for call in range(2):
            state = (torch.randn(documents, heads, key_dim, value_dim, device=device, generator=generator)
                     * (call + 1)).to(dtype)
            if nonfinite and call:
                state[0, 1, 2, 3] = float("inf")
            extension.linear_state_stats(state, got, call > 0)
            norms = torch.linalg.vector_norm(state, dim=(-2, -1), dtype=torch.float32)
            values = torch.cat((torch.full((1,), documents, dtype=torch.float64, device=device),
                                norms.sum(dim=0, dtype=torch.float64), norms.amax(dim=0)))
            if call:
                earlier = want.double()
                attention._merge(earlier, values, attention._LINEAR_STATE_FIELDS)
                values = earlier
            want.copy_(values)
        if not torch.equal(got.view(torch.int32), want.view(torch.int32)):
            problems.append(f"linear state {documents}x{heads}x{key_dim}x{value_dim} {dtype}: words differ")
    return problems


def _check_tuning(device: torch.device) -> list[str]:
    """Every configuration the sweep tries gives the words of the CPU reference."""
    problems = []
    cpu = torch.device("cpu")
    for shape in _SHAPES[-2:]:
        a_log, dt_bias = torch.randn(shape.heads), torch.linspace(-7, 25, shape.heads)
        reference = _FakeRecorder(shape.heads, cpu)
        _observe(AttentionStats(reference), _softmax_inputs(shape, cpu, 5, True), shape, 0, None, a_log, dt_bias)
        inputs = _softmax_inputs(shape, device, 5, True)
        for tile_rows, num_warps, target_programs in itertools.product(_SWEEP_TILES, _SWEEP_WARPS, _SWEEP_PROGRAMS):
            got = _FakeRecorder(shape.heads, device)
            for _, _, launch in _launchers(shape, inputs, got, a_log.to(device), dt_bias.to(device)):
                launch(tile_rows=tile_rows, num_warps=num_warps, target_programs=target_programs)
            label = f"{shape.name} tile_rows={tile_rows} num_warps={num_warps} target_programs={target_programs}"
            problems.extend(_compare(label, reference, got))
    return problems


def _time_cuda(function, repeats: int = _TIMING_REPEATS) -> tuple[float, float]:
    """(GPU ms, CPU issue ms) per call."""
    for _ in range(_TIMING_WARMUP):
        function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start.record()
    for _ in range(repeats):
        function()
    issued = time.perf_counter() - started
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats, issued / repeats * 1000


def _time_observations(device: torch.device) -> None:
    """The observations as the callback makes them, in the default configuration, against the target of the
    8k rows x 16 heads shape; the CPU issue time with the direct launch and with the JIT call."""
    print(f"\n{'observation':44s} {'GPU ms':>9s} {'CPU ms direct':>14s} {'CPU ms JIT':>11s} "
          f"{'GB/s read':>10s} {'target ms':>10s}")
    for shape in _PRODUCTION_SHAPES:
        a_log, dt_bias = torch.randn(shape.heads, device=device), torch.randn(shape.heads, device=device)
        inputs = _softmax_inputs(shape, device, 0, False)
        query, key, value, lse, gate, beta, log_decay, linear_value, output = inputs
        stats = AttentionStats(_FakeRecorder(shape.heads, device))
        reads = {label: read
                 for label, read, _ in _launchers(shape, inputs, _FakeRecorder(shape.heads, device), a_log, dt_bias)}
        for label, function in (
            ("softmax q/k/v/lse", lambda: stats.observe_softmax(0, 1, query, key, value, lse, shape.nope)),
            ("softmax gate", lambda: stats.observe_softmax_gate(0, 1, gate)),
            ("linear beta/g/v/o",
             lambda: stats.observe_linear(0, 0, beta, log_decay, linear_value, output, a_log, dt_bias)),
        ):
            def call():
                stats.begin_step(False)
                function()
            timings = {}
            for direct in (True, False):
                with _launch_path(direct):
                    timings[direct] = _time_cuda(call)
            gpu_ms, direct_ms = timings[True]
            target = _TARGET_MS[label] if shape.name == _TARGET_SHAPE else None
            verdict = "" if target is None else f"{target:>10.3f} {'ok' if gpu_ms <= target else 'OVER'}"
            print(f"{shape.name + ' ' + label:44s} {gpu_ms:>9.4f} {direct_ms:>14.4f} {timings[False][1]:>11.4f} "
                  f"{reads[label] / gpu_ms / 1e6:>10.0f} {verdict}")


def _time_sweep(device: torch.device) -> None:
    """The best few of tile rows x warps x target programs for every kernel and production shape, with the
    default's time, so the defaults in attention_kernel.py can be set from this table."""
    for shape in _PRODUCTION_SHAPES:
        a_log, dt_bias = torch.randn(shape.heads, device=device), torch.randn(shape.heads, device=device)
        inputs = _softmax_inputs(shape, device, 0, False)
        for label, read, launch in _launchers(shape, inputs, _FakeRecorder(shape.heads, device), a_log, dt_bias):
            default_ms, _ = _time_cuda(launch)
            timings = sorted(
                (_time_cuda(lambda: launch(tile_rows=tile, num_warps=warps, target_programs=programs))[0],
                 tile, warps, programs)
                for tile, warps, programs in itertools.product(_SWEEP_TILES, _SWEEP_WARPS, _SWEEP_PROGRAMS)
            )
            print(f"\n{shape.name} {label}: default {default_ms:.4f} ms ({read / default_ms / 1e6:.0f} GB/s)")
            for gpu_ms, tile, warps, programs in timings[:5]:
                print(f"    tile rows {tile:>3d}, {warps} warps, {programs:>4d} programs: "
                      f"{gpu_ms:.4f} ms ({read / gpu_ms / 1e6:.0f} GB/s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-only", action="store_true")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        print("this check needs a CUDA device", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")

    problems = (_check_correctness(device) + _check_determinism(device) + _check_launch_paths(device)
                + _check_tuning(device) + _check_linear_state(device))
    if problems:
        print(f"{len(problems)} mismatch(es):")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print("row counts match exactly; every field matches the CPU reference within float32 rounding, in every "
          "tuning configuration, and the words are the same in every run and in both launch paths "
          f"(direct launches {'on' if triton_launch._DIRECT_LAUNCH else 'unavailable: Triton ' + triton.__version__}, "
          f"{sum(native is not None for _, _, native in triton_launch._compiled.values())} of "
          f"{len(triton_launch._compiled)} variants launched by the extension)")
    if arguments.correctness_only:
        return
    _time_observations(device)
    _time_sweep(device)


if __name__ == "__main__":
    main()
