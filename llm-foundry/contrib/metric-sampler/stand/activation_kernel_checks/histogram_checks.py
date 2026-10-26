from __future__ import annotations

import torch

from metric_sampler.capture.stats.activations import ActivationStats
from stand.activation_kernel_checks.fake_recorder import (
    BUFFER_WORDS,
    FLOAT_SUM_WORDS,
    HISTOGRAM_BINS,
    QUANTILE_COUNTS_OFFSET,
    QUANTILE_WORDS,
    QUANTILES_OFFSET,
    WORDS_WITH_QUANTILES,
    FakeRecorder,
    make_recorder,
)
from stand.activation_kernel_checks.kernel_calls import (
    launch_statistics_kernel,
    observe_statistics,
    observe_with_quantiles,
)
from stand.activation_kernel_checks.populations import ACTIVATION_SCALE, make_tensors
from stand.activation_kernel_checks.statistics_oracle import compare, compare_histogram, run_reference

SECOND_MICROBATCH_SCALE = 0.5
PADDED_ROWS = 65536
PADDED_WIDTH = 7168
PADDED_ZERO_ROW_STRIDE = 256
PADDED_ZERO_COUNT = PADDED_ROWS // PADDED_ZERO_ROW_STRIDE
PADDED_TILE_CONFIGURATIONS = ((4096, 8), (16384, 2))
PADDED_REPEATS = 5
LARGE_COUNT_SHAPES = ((5592406, 3), (40560, 1536), (66048, 1536))
LARGE_COUNT_CALLS = (1, 2)
UNIT_EXPONENT_BIN = 40
FLOAT_SUM_TOLERANCE = 1e-5
MAGNITUDE_POWERS = (1, 2, 3, 4)


def check_tensor_statistics(device: torch.device, extension) -> list[str]:
    problems = []
    for name, tensor in make_tensors(device):
        for detailed in (False, True):
            label = f"{name} ({'detailed' if detailed else 'non-detailed'})"
            reference = run_reference(tensor, detailed)
            problems.extend(compare(label, reference, _run_kernel(tensor)))
            problems.extend(f"{label} [observer Triton] {problem}"
                            for problem in check_microbatches(tensor, detailed, False))
            if extension is not None:
                problems.extend(compare(f"{label} [ext]", reference, _run_extension(tensor, extension)))
                problems.extend(compare(f"{label} [ext with quantiles]", reference,
                                        _run_extension_with_quantiles(tensor, extension)))
                problems.extend(f"{label} [observer extension] {problem}"
                                for problem in check_microbatches(tensor, detailed, extension))
    return problems


def _run_kernel(tensor: torch.Tensor) -> FakeRecorder:
    recorder = make_recorder(tensor.device)
    matrix = tensor.reshape(-1, tensor.shape[-1])
    launch_statistics_kernel(matrix, recorder.float_words)
    torch.cuda.synchronize()
    return recorder


def _run_extension(tensor: torch.Tensor, extension) -> FakeRecorder:
    recorder = make_recorder(tensor.device)
    matrix = tensor.reshape(-1, tensor.shape[-1])
    observe_statistics(extension, matrix, recorder.float_words, False)
    torch.cuda.synchronize()
    return recorder


def _run_extension_with_quantiles(tensor: torch.Tensor, extension) -> FakeRecorder:
    """The statistics of `tensor` as ActivationStats takes them with exact quantiles: one call, which also
    writes the quantiles and their counts past the slots."""
    words = torch.zeros(WORDS_WITH_QUANTILES, device=tensor.device)
    observe_with_quantiles(extension, tensor.reshape(-1, tensor.shape[-1]), words)
    torch.cuda.synchronize()
    return FakeRecorder(words[:BUFFER_WORDS])


def check_microbatches(tensor: torch.Tensor, detailed: bool, extension) -> list[str]:
    recorder = make_recorder(tensor.device)
    stats = ActivationStats(recorder)
    stats._extension = extension
    stats.begin_step(detailed)
    microbatches = (tensor, tensor[:max(1, tensor.shape[0] // 2)] * SECOND_MICROBATCH_SCALE)
    for microbatch, value in enumerate(microbatches):
        stats.begin_microbatch(microbatch)
        stats.observe("point", 0, value)
    torch.cuda.synchronize()
    problems = []
    for microbatch, value in enumerate(microbatches):
        reference = run_reference(value, detailed)
        got = FakeRecorder(recorder.float_words[microbatch * BUFFER_WORDS:(microbatch + 1) * BUFFER_WORDS])
        problems.extend(compare(f"microbatch={microbatch}", reference, got))
    return problems


def check_padded_histogram_cancellation(device: torch.device) -> list[str]:
    # BLOCK_H=8192 creates 67,108,864 excluded elements; only 256 real elements belong to bin 0.
    activations = torch.ones(PADDED_ROWS, PADDED_WIDTH, dtype=torch.bfloat16, device=device)
    activations[::PADDED_ZERO_ROW_STRIDE, 0] = 0
    problems = []
    for tile_elements, num_warps in PADDED_TILE_CONFIGURATIONS:
        for repeat in range(PADDED_REPEATS):
            recorder = make_recorder(device)
            launch_statistics_kernel(
                activations, recorder.float_words, tile_elements=tile_elements, num_warps=num_warps,
            )
            torch.cuda.synchronize()
            count = recorder.tensor("exp_hist")[0].item()
            if count != PADDED_ZERO_COUNT:
                problems.append(
                    f"padded histogram tile={tile_elements} warps={num_warps} repeat={repeat}: "
                    f"bin 0 expected {PADDED_ZERO_COUNT}, got {count}"
                )
    return problems


def check_large_integer_counts(device: torch.device, extension) -> list[str]:
    problems = []
    for rows, width in LARGE_COUNT_SHAPES:
        activations = torch.ones(rows, width, dtype=torch.bfloat16, device=device)
        activations[0, 0] = 2
        expected_histogram = torch.zeros(HISTOGRAM_BINS, dtype=torch.int64)
        expected_histogram[UNIT_EXPONENT_BIN] = rows * width - 1
        expected_histogram[UNIT_EXPONENT_BIN + 1] = 1
        expected_token_histogram = torch.zeros(HISTOGRAM_BINS, dtype=torch.int64)
        expected_token_histogram[UNIT_EXPONENT_BIN] = rows
        for backend, backend_extension in (("triton", False), ("extension", extension)):
            recorder = make_recorder(device)
            stats = ActivationStats(recorder)
            stats._extension = backend_extension
            stats.begin_step(False)
            for call in LARGE_COUNT_CALLS:
                stats.observe("point", 0, activations)
                torch.cuda.synchronize()
                label = f"large [{rows}, {width}] {backend} calls={call}"
                problems.extend(_compare_large_histograms(
                    label, recorder, expected_histogram * call, expected_token_histogram * call))
                expected_counts = [0, rows * width * call]
                actual_counts = recorder.tensor("counts")[:2].to(torch.int64).cpu().tolist()
                if actual_counts != expected_counts:
                    problems.append(f"{label}: expected counts {expected_counts}, got {actual_counts}")
                if recorder.tensor("moments")[2:].count_nonzero().item() != 0:
                    problems.append(f"{label}: unused float32 count words changed")
        # The counts of the raw values of a tensor of one value, given away by their counters over and over,
        # and the quantiles selected from them, call after call into one set of words.
        words = torch.zeros(WORDS_WITH_QUANTILES, device=device)
        recorded = FakeRecorder(words[:BUFFER_WORDS])
        for call in LARGE_COUNT_CALLS:
            observe_with_quantiles(extension, activations, words)
            torch.cuda.synchronize()
            label = f"large [{rows}, {width}] quantiles calls={call}"
            problems.extend(_compare_large_histograms(
                label, recorded, expected_histogram * call, expected_token_histogram * call))
            quantiles = words[QUANTILES_OFFSET:QUANTILE_COUNTS_OFFSET].tolist()
            quantile_counts = words[QUANTILE_COUNTS_OFFSET:].view(torch.int32).tolist()
            if quantiles != [1.0] * (QUANTILE_WORDS - 1) + [2.0] or quantile_counts != [rows * width, 0, 0, 0, 0]:
                problems.append(f"{label}: quantiles {quantiles} with counts {quantile_counts}")
    return problems


def _compare_large_histograms(
    label: str, recorder: FakeRecorder, expected_histogram: torch.Tensor, expected_token_histogram: torch.Tensor
) -> list[str]:
    problems = []
    problems.extend(compare_histogram(label, "exp_hist", expected_histogram, recorder.tensor("exp_hist").cpu()))
    problems.extend(
        compare_histogram(label, "token_rms_hist", expected_token_histogram, recorder.tensor("token_rms_hist").cpu())
    )
    return problems


def check_observed_layouts(device: torch.device, extension) -> list[str]:
    """The extension reads an observed tensor of any layout as the rows of its last axis: counts, histograms,
    amax and quantiles of the raw tensor equal, bit for bit, those of its reshape to rows. The fp32 sums of powers
    are added by blocks in the order they finish, so they differ by float32 rounding between any two calls, even on
    one tensor."""
    layouts = _observed_layouts(device)
    problems = []
    for name, raw in layouts:
        rows = raw.reshape(-1, raw.shape[-1])
        results = []
        for tensor in (raw, rows):
            words = torch.zeros(WORDS_WITH_QUANTILES, device=device)
            observe_with_quantiles(extension, tensor, words)
            results.append(words)
        float_sums = list(FLOAT_SUM_WORDS)
        exact = torch.ones(WORDS_WITH_QUANTILES, dtype=torch.bool, device=device)
        exact[float_sums] = False
        if not torch.equal(results[0].view(torch.int32)[exact], results[1].view(torch.int32)[exact]):
            problems.append(f"observe_with_quantiles of a {name} tensor differs from that of its reshape to rows")
        magnitude = raw.double().abs()
        scale = torch.stack([magnitude.pow(power).sum() for power in MAGNITUDE_POWERS])
        difference = (results[0][float_sums].double() - results[1][float_sums].double()).abs()
        if (difference > FLOAT_SUM_TOLERANCE * scale).any():
            problems.append(f"sums of powers of a {name} tensor differ from those of its reshape to rows "
                            "beyond fp32 rounding")
    return problems


def _observed_layouts(device: torch.device) -> list[tuple[str, torch.Tensor]]:
    return [
        ("contiguous 3-D", (torch.randn(8, 512, 896, device=device) * ACTIVATION_SCALE).to(torch.bfloat16)),
        ("contiguous 4-D", torch.randn(2, 3, 700, 96, device=device)),
        ("contiguous 1-D", torch.randn(5000, device=device)),
        ("one row", torch.randn(1, 300, device=device).half()),
        ("last axis of one", torch.randn(4096, 1, device=device)),
        ("sliced columns", torch.randn(3000, 128, device=device).to(torch.bfloat16)[:, 3:99]),
        ("sliced 3-D", torch.randn(4, 600, 64, device=device)[:, ::2]),
        ("transposed", torch.randn(896, 4096, device=device).to(torch.bfloat16).t()),
    ]
