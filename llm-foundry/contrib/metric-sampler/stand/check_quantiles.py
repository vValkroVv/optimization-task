from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import RankRecorder
from metric_sampler.capture.ring import ShmRing
from metric_sampler.capture.stats.helpers import quantiles as implementation
from metric_sampler.capture.stats.helpers.quantiles import (
    QUANTILE_PERCENTILES,
    exact_quantiles,
    exact_quantiles_aten,
    order_statistics,
    row_top_values,
)
from metric_sampler.capture.stats.quantiles import ExactQuantileStats
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import QUANTILE_COUNTS_WIDTH, QUANTILE_LEVELS

_PERCENT = 100
_MILLISECONDS_PER_SECOND = 1000
_RING_NAME_SUFFIX_LENGTH = 16
_WARMUP_CALLS = 3
# A value is masked out when its uniform draw is at most the threshold.
_BENCHMARK_MASK_THRESHOLD = 0.1
_CASE_MASK_THRESHOLD = 0.5
# Activation populations of one microbatch: tokens by width of a narrow layer and of a wide one.
_ACTIVATION_POPULATION_SHAPES = ((49152, 896), (24576, 7168))


def reference(values: torch.Tensor, valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    flat = values.detach().cpu().to(torch.float64).numpy().reshape(-1)
    mask = np.ones(flat.shape, dtype=bool) if valid is None else valid.cpu().numpy().reshape(-1)
    finite = flat[mask & np.isfinite(flat)]
    counts = np.array([finite.size, np.count_nonzero(mask & np.isnan(flat)),
                       np.count_nonzero(mask & np.isposinf(flat)),
                       np.count_nonzero(mask & np.isneginf(flat)), np.count_nonzero(~mask)], dtype=np.int32)
    selected = (np.quantile(finite, np.array(QUANTILE_PERCENTILES) / _PERCENT, method='inverted_cdf')
                if finite.size else np.full(len(QUANTILE_LEVELS), np.nan))
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    return torch.from_numpy(selected).to(dtype), torch.from_numpy(counts)


def check_case(name: str, values: torch.Tensor, valid: torch.Tensor | None = None) -> dict:
    expected, expected_counts = reference(values, valid)
    actual, counts = exact_quantiles(values, valid)
    legacy, legacy_counts = exact_quantiles_aten(values, valid)
    actual, counts = actual.cpu(), counts.cpu()
    legacy, legacy_counts = legacy.cpu(), legacy_counts.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(actual, legacy, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(counts, legacy_counts, rtol=0, atol=0)
    if values.dtype == torch.float64:
        finite = torch.isfinite(actual)
        assert torch.equal(actual[finite].view(torch.int64), legacy[finite].view(torch.int64))
    assert int(counts.sum()) == values.numel()
    return {'case': name, 'dtype': str(values.dtype), 'shape': list(values.shape),
            'contiguous': values.is_contiguous(), 'counts': counts.tolist(), 'passed': True,
            'numpy_and_legacy_exact': True}


def check_capture() -> dict:
    requests = [(metric_by_name(name), 0, 0, width) for name, width in [
        ('act/residual/quantiles', len(QUANTILE_LEVELS)), ('act/residual/quantile_counts', QUANTILE_COUNTS_WIDTH),
        ('router/entropy_quantiles', 2 * len(QUANTILE_LEVELS)),
        ('router/entropy_quantile_counts', QUANTILE_COUNTS_WIDTH),
    ]]
    layout = Layout(requests)
    ring = ShmRing.create('ms_quantiles_' + uuid.uuid4().hex[:_RING_NAME_SUFFIX_LENGTH], layout, 1)
    recorder = RankRecorder(layout, ring, torch.device('cuda'))
    try:
        recorder.begin_step(False)
        stats = ExactQuantileStats(recorder)
        population = torch.tensor([7., 3., float('nan'), 2.], device='cuda')
        valid = torch.tensor([True, True, False, True], device='cuda')
        expected, counts = reference(population, valid)
        stats.observe('act/residual/quantiles', 'act/residual/quantile_counts', 0, 0, population, valid)
        population.fill_(999)
        valid.fill_(False)
        try:
            stats.observe('act/residual/quantiles', 'act/residual/quantile_counts', 0, 0, population)
        except ValueError:
            pass
        else:
            raise AssertionError('capture accepted a second population for one key')
        entropy = torch.tensor([1 + 2**-40, 1 + 2**-42, 0.5, float('nan')],
                               dtype=torch.float64, device='cuda')
        expected_entropy, entropy_counts = reference(entropy, None)
        stats.observe('router/entropy_quantiles', 'router/entropy_quantile_counts', 0, 0,
                      entropy, pack_float64=True)
        stats.flush()
        for name, expected_value in [
            ('act/residual/quantiles', expected), ('act/residual/quantile_counts', counts),
            ('router/entropy_quantiles', expected_entropy.view(torch.int32)),
            ('router/entropy_quantile_counts', entropy_counts),
        ]:
            slot = layout.slot(name, 0, 0)
            buffer = recorder.float_buffer() if name == 'act/residual/quantiles' else recorder.buffer
            actual = buffer[slot.offset:slot.offset + slot.width].cpu()
            torch.testing.assert_close(actual, expected_value, rtol=0, atol=0, equal_nan=True)
        return {'passed': True, 'computed_at_observation': True, 'repeated_key_rejected': True,
                'float64_bits_equal': True}
    finally:
        torch.cuda.synchronize()
        recorder.close()
        ring.unlink()


def check_order_statistics(generator: torch.Generator) -> list[dict]:
    """Selected ranks equal torch.sort's documented order (every NaN last) on ties, signed zeros,
    non-finite values and starts that defeat 16-byte loads; row top values equal torch.topk's."""
    cases = []
    populations = {
        'normal': torch.randn(1_000_003, generator=generator),
        'ties': (torch.randn(300_001, generator=generator) * 3).round(),
        'signed_zeros_subnormals': torch.tensor([-0.0, 0.0, 1e-45, -1e-45, 1e-40]).repeat(4001),
        'nonfinite': torch.cat((torch.randn(50_000, generator=generator),
                                torch.tensor([float('nan')] * 7 + [float('inf')] * 5 + [-float('inf')] * 3))),
        'single': torch.tensor([2.5]),
    }
    for name, values in populations.items():
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for offset in (0, 1):
                gpu_values = values.to(dtype).to('cuda')[offset:]
                if not gpu_values.numel():
                    continue
                size = gpu_values.numel()
                widened = gpu_values.float()
                # CUDA's radix sort keeps a negative NaN (which a 16-bit conversion can produce) first.
                ordered = torch.sort(
                    torch.where(widened.isnan(), torch.full_like(widened, float('nan')), widened)
                ).values
                ranks = sorted({0, size // 7, size // 3, (size - 1) // 2, (3 * size) // 4, size - 1})
                actual = order_statistics(gpu_values, ranks).cpu()
                expected = torch.stack([ordered[rank] for rank in ranks]).cpu()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                cases.append({'case': f'order_statistics_{name}', 'dtype': str(dtype), 'offset': offset,
                              'passed': True})
    for rows, columns, top_k in [(1, 1, 1), (7, 5, 3), (49152, 256, 12), (333, 1024, 32), (2000, 160, 9)]:
        for quantized in (False, True):
            matrix = torch.randn(rows, columns, generator=generator)
            matrix = ((matrix * 2).round() if quantized else matrix).to('cuda')
            torch.testing.assert_close(
                row_top_values(matrix, top_k), matrix.topk(top_k, dim=-1).values, rtol=0, atol=0
            )
            cases.append({'case': 'row_top_values', 'shape': [rows, columns], 'k': top_k, 'ties': quantized,
                          'passed': True})
    return cases


def benchmark(elements: int, dtype: torch.dtype, population: str, repeats: int, width: int = 1) -> dict:
    generator = torch.Generator(device='cuda').manual_seed(74821)
    values = (torch.linspace(-5, 5, elements, device='cuda', dtype=dtype) if population == 'ordered'
              else torch.randn(elements, device='cuda', dtype=dtype, generator=generator))
    values = values.reshape(-1, width)
    valid = (
        (torch.rand(elements, device='cuda', generator=generator) > _BENCHMARK_MASK_THRESHOLD).reshape(values.shape)
        if population == 'random_masked' else None
    )
    actual = exact_quantiles(values, valid)
    legacy = exact_quantiles_aten(values, valid)
    torch.testing.assert_close(actual[0], legacy[0], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(actual[1], legacy[1], rtol=0, atol=0)
    del actual, legacy
    functions = [('aten', exact_quantiles_aten), ('radix', exact_quantiles)]
    for _, function in functions:
        for _ in range(_WARMUP_CALLS):
            function(values, valid)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    timings = {name: {'gpu_ms': [], 'wall_ms': [], 'peak_bytes': []} for name, _ in functions}
    for repeat in range(repeats):
        for name, function in functions[::1 if repeat % 2 == 0 else -1]:
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            function(values, valid)
            end.record()
            end.synchronize()
            timings[name]['gpu_ms'].append(start.elapsed_time(end))
            timings[name]['wall_ms'].append((time.perf_counter() - wall_start) * _MILLISECONDS_PER_SECOND)
            timings[name]['peak_bytes'].append(torch.cuda.max_memory_allocated() - baseline)
    return {'elements': elements, 'dtype': str(dtype), 'population': population, 'repeats': repeats,
            'same_input_legacy_exact': True,
            'backends': {name: {'gpu_median_ms': statistics.median(samples['gpu_ms']),
                                'wall_median_ms': statistics.median(samples['wall_ms']),
                                'peak_allocated_above_input_bytes': max(samples['peak_bytes'])}
                         for name, samples in timings.items()},
            'scope': 'Complete quantile computation only; excludes capture cloning, training and delivery.'}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict CUDA checks and measured cost of complete rank-local quantile populations."
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    arguments = parser.parse_args()
    if arguments.repeats <= 0:
        parser.error('--repeats must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU execution is not a passing GPU check')
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(73421)
    cases = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        values = torch.randn(33, 97, generator=generator).to(dtype)
        values[0, :3] = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=dtype)
        gpu = values.T.to('cuda')
        cases.append(check_case('noncontiguous', gpu))
        mask = (torch.arange(gpu.numel()).reshape(gpu.shape) % 7 != 0).to('cuda')
        cases.append(check_case('label_mask', gpu, mask))
        cases.append(check_case('empty', gpu[:0]))
        cases.append(check_case('all_masked', gpu, torch.zeros_like(gpu, dtype=torch.bool)))
        cases.append(check_case('all_nonfinite', values[0, :3].to('cuda')))
    cases.append(check_case('rank_boundaries', torch.arange(10001, dtype=torch.float32, device='cuda')))
    for elements in (1, 1023, 1024, 1025, 262143, 262144, 262145):
        values = torch.randn(elements, generator=generator, dtype=torch.float64).to('cuda')
        valid = torch.arange(elements, device='cuda') % 11 != 0
        cases.append(check_case('tile_and_grid_boundary', values, valid))
    cases.append(check_case('float64_precision', torch.tensor(
        [1 + 2**-40, 1 + 2**-42, -1e100, 1e100, -0.0, 0.0], dtype=torch.float64, device='cuda')))
    large = torch.ones(2**24 + 1, dtype=torch.bfloat16, device='cuda')
    values, counts = exact_quantiles(large)
    assert values.cpu().tolist() == [1.] * 9
    assert counts.cpu().tolist() == [2**24 + 1, 0, 0, 0, 0]
    legacy_values, legacy_counts = exact_quantiles_aten(large)
    torch.testing.assert_close(values, legacy_values, rtol=0, atol=0)
    torch.testing.assert_close(counts, legacy_counts, rtol=0, atol=0)
    cases.append({'case': 'count_above_float32_integer_precision', 'elements': large.numel(), 'passed': True})
    del large
    # Radix selection: ties, signed zeros, subnormals, extreme exponents and block/grid boundaries.
    tiny = torch.finfo(torch.float32).tiny
    hard = {
        'constant': torch.full((5000,), -2.5),
        'two_values': torch.tensor([1., 2.]).repeat(4097),
        'signed_zeros': torch.tensor([-0.0, 0.0, -0.0, 1e-45, -1e-45]).repeat(999),
        'subnormals': torch.arange(-3000, 3000, dtype=torch.float32) * tiny / 1024,
        'extremes': torch.tensor([3.4e38, -3.4e38, 1e-38, -1e-38, 0.0, 1.0]).repeat(777),
        'wide_range': (
            torch.exp(torch.empty(300001).uniform_(-80, 80, generator=generator))
            * torch.sign(torch.randn(300001, generator=generator))
        ),
        'last_mantissa_bits': 1 + torch.arange(70001, dtype=torch.float32) * torch.finfo(torch.float32).eps,
        'heavy_tail': torch.randn(500003, generator=generator) ** 7,
        'descending': torch.arange(65537, 0, -1, dtype=torch.float32),
    }
    for name, values in hard.items():
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            converted = values.to(dtype).to('cuda')
            cases.append(check_case(name, converted))
            cases.append(check_case(
                name + '_masked', converted,
                torch.rand(converted.shape, generator=generator).to('cuda') > _CASE_MASK_THRESHOLD
            ))
    for elements in (2, 3, 7, 4095, 4096, 4097, 4096 * 264 - 1, 4096 * 264, 4096 * 264 + 1, 4096 * 264 * 3 + 5):
        cases.append(check_case(
            'radix_block_and_grid_boundary', torch.randn(elements, generator=generator).to(torch.bfloat16).to('cuda')
        ))
    cases += check_order_statistics(generator)
    capture = check_capture()
    timings = [benchmark(elements, dtype, population, arguments.repeats)
               for elements in (4096, 262144, 4194304)
               for dtype in (torch.bfloat16, torch.float32, torch.float64)
               for population in ('ordered', 'random', 'random_masked')]
    timings += [benchmark(tokens * width, torch.bfloat16, 'random', arguments.repeats, width)
                for tokens, width in _ACTIVATION_POPULATION_SHAPES]
    result = {'passed': True, 'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'cuda': torch.version.cuda,
              'source_sha256': hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
              'kernel_sha256': hashlib.sha256(
                  Path(importlib.util.find_spec('metric_sampler.capture.kernels.quantiles_kernel').origin).read_bytes()
              ).hexdigest(),
              'quantile_percentiles': list(QUANTILE_PERCENTILES), 'cases': cases, 'capture': capture,
              'timings': timings, 'elapsed_seconds': time.perf_counter() - started,
              'training_overhead_measured': False, 'global_optimality_claim': False}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(
        {'passed': True, 'cases': len(cases), 'capture': capture['passed'], 'output': str(arguments.output)}
    ))


if __name__ == '__main__':
    main()
