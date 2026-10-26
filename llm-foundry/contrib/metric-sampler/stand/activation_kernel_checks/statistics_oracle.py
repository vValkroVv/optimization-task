from __future__ import annotations

import math

import torch

from metric_sampler.capture.stats.activations import ActivationStats
from stand.activation_kernel_checks.fake_recorder import FakeRecorder, make_recorder

# Constants of the bound on the rounding error of a float32 sum: the Triton kernel's tile sizes and tiles per
# program, and the CUDA extension's block of threads, vector width, warp reduction stages and blocks per
# multiprocessor.
TRITON_TILE_ELEMENTS = (2048, 4096, 8192, 16384)
TRITON_MAX_TILES_PER_PROGRAM = 1024
CUDA_BLOCK_THREADS = 256
CUDA_VECTOR_ELEMENTS = 8
CUDA_WARP_REDUCTION_ADDITIONS = 2 * 5
CUDA_BLOCKS_PER_MULTIPROCESSOR = 8
CPU_MULTIPROCESSORS = 1
MAX_REPORTED_WORDS = 8
NAN_PLACEHOLDER = 7.0


def sum_rounding_factor(rows: int, width: int, multiprocessors: int) -> float:
    # Maximum additions on any input's path: pairwise row/tile reduction, local loop,
    # then serial atomic additions. Include one rounding for x*x.
    block_width = 1 << (width - 1).bit_length()
    depths = []
    for tile_elements in TRITON_TILE_ELEMENTS:
        tile_rows = max(1, tile_elements // block_width)
        tiles = (rows + tile_rows - 1) // tile_rows
        per_program = max(1, (tiles + TRITON_MAX_TILES_PER_PROGRAM - 1) // TRITON_MAX_TILES_PER_PROGRAM)
        programs = (tiles + per_program - 1) // per_program
        depths.append(block_width.bit_length() - 1 + tile_rows.bit_length() - 1 + per_program + programs + 1)
    blocks = min(rows, multiprocessors * CUDA_BLOCKS_PER_MULTIPROCESSOR)
    depths.append(
        (width + CUDA_BLOCK_THREADS - 1) // CUDA_BLOCK_THREADS + CUDA_VECTOR_ELEMENTS
        + CUDA_WARP_REDUCTION_ADDITIONS + (rows + blocks - 1) // blocks + blocks + 1
    )
    error32 = max(depths) * (torch.finfo(torch.float32).eps / 2)
    error64 = rows * width * (torch.finfo(torch.float64).eps / 2)
    if max(error32, error64) >= 1:
        raise ValueError("Reduction exceeds the rounding-bound domain")
    return error32 / (1 - error32) + error64 / (1 - error64)


def run_reference(tensor: torch.Tensor, detailed: bool) -> FakeRecorder:
    recorder = make_recorder(tensor.device)
    stats = ActivationStats(recorder)
    stats.begin_step(detailed)
    stats._observe_cpu(stats._resolve_point("point", 0), tensor.reshape(-1, tensor.shape[-1]))
    values = tensor.detach().to(torch.float64)
    values = torch.where(torch.isfinite(values), values, 0)
    float64_sum = values.sum().item()
    float64_sum_of_squares = (values * values).sum().item()
    absolute_sum = values.abs().sum().item()
    width = tensor.shape[-1]
    multiprocessors = (
        torch.cuda.get_device_properties(tensor.device).multi_processor_count
        if tensor.is_cuda else CPU_MULTIPROCESSORS
    )
    factor = sum_rounding_factor(tensor.numel() // width, width, multiprocessors)
    # Absolute underflow allowance also covers flush-to-zero of FP32 intermediates.
    underflow = tensor.numel() * torch.finfo(torch.float32).tiny
    recorder.sum_reference = (float64_sum, float64_sum_of_squares)
    recorder.sum_bounds = (factor * absolute_sum + underflow, factor * float64_sum_of_squares + underflow)
    # A cube or fourth power is one more rounding than a square; its bound scales with |x|^3, x^4.
    cubes, fourths = values ** 3, values ** 4
    recorder.power_reference = (cubes.sum().item(), fourths.sum().item())
    recorder.power_bounds = (2 * factor * cubes.abs().sum().item() + underflow,
                             2 * factor * fourths.sum().item() + underflow)
    largest = torch.finfo(torch.float32).max
    recorder.power_overflow = (bool((cubes.abs() > largest).any()), bool((fourths > largest).any()))
    return recorder


def equal_with_nans(first: torch.Tensor, second: torch.Tensor) -> bool:
    return torch.equal(first.nan_to_num(NAN_PLACEHOLDER), second.nan_to_num(NAN_PLACEHOLDER))


def compare_histogram(name: str, histogram_name: str, reference: torch.Tensor, got: torch.Tensor) -> list[str]:
    if reference.is_floating_point() or got.is_floating_point():
        raise TypeError("Histogram references and observations must retain integer counts")
    reference = reference.to(torch.int64)
    got = got.to(torch.int64)
    if torch.equal(reference, got):
        return []
    bad_bins = (reference != got).nonzero().flatten().tolist()
    return [f"{name}: {histogram_name} differs at bins {bad_bins}: "
            f"expected {reference[bad_bins].tolist()}, got {got[bad_bins].tolist()}"]


def compare(name: str, reference: FakeRecorder, got: FakeRecorder) -> list[str]:
    problems = []
    problems.extend(compare_histogram(name, "exp_hist", reference.tensor("exp_hist"), got.tensor("exp_hist")))
    problems.extend(
        compare_histogram(name, "token_rms_hist", reference.tensor("token_rms_hist"), got.tensor("token_rms_hist"))
    )
    problems.extend(_compare_moment_sums(name, reference, got))

    reference_nonfinite = reference.tensor("counts")[0].item()
    got_nonfinite = got.tensor("counts")[0].item()
    if reference_nonfinite != got_nonfinite:
        problems.append(f"{name}: nonfinite count {reference_nonfinite} vs {got_nonfinite}")

    reference_finite = reference.tensor("counts")[1].item()
    got_finite = got.tensor("counts")[1].item()
    if reference_finite != got_finite:
        problems.append(f"{name}: finite count {reference_finite} vs {got_finite}")

    if torch.count_nonzero(got.tensor("moments")[2:]):
        problems.append(f"{name}: unused float32 count words must remain zero")

    problems.extend(_compare_power_sums(name, reference, got))

    reference_amax = reference.tensor("amax").item()
    got_amax = got.tensor("amax").item()
    if reference_amax != got_amax:
        problems.append(f"{name}: amax {reference_amax} vs {got_amax}")

    return problems


def _compare_moment_sums(name: str, reference: FakeRecorder, got: FakeRecorder) -> list[str]:
    problems = []
    assert reference.sum_reference is not None and reference.sum_bounds is not None
    for field, expected, bound, actual in zip(
        ("sum", "sumsq"), reference.sum_reference, reference.sum_bounds, got.tensor("moments")[:2].tolist()
    ):
        if abs(expected) > torch.finfo(torch.float32).max:
            matches = math.isinf(actual) and math.copysign(1, actual) == math.copysign(1, expected)
        else:
            matches = math.isfinite(actual) and abs(expected - actual) <= bound
        if not matches:
            problems.append(f"{name}: {field} FP64={expected}, got={actual}, absolute rounding bound={bound}")
    return problems


def _compare_power_sums(name: str, reference: FakeRecorder, got: FakeRecorder) -> list[str]:
    problems = []
    assert reference.power_reference is not None and reference.power_bounds is not None
    for field, expected, bound, overflow, actual in zip(
        ("sum_cube", "sum_fourth"), reference.power_reference, reference.power_bounds, reference.power_overflow,
        got.tensor("power_sums").tolist(),
    ):
        if overflow or abs(expected) > torch.finfo(torch.float32).max:
            matches = not math.isfinite(actual)
        else:
            matches = math.isfinite(actual) and abs(expected - actual) <= bound
        if not matches:
            problems.append(f"{name}: {field} FP64={expected}, got={actual}, absolute rounding bound={bound}")
    return problems
