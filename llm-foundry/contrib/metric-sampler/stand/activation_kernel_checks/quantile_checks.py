from __future__ import annotations

import torch

from metric_sampler.capture.stats.helpers.quantiles import exact_quantiles, exact_quantiles_aten, order_statistics
from metric_sampler.contract.metrics import QUANTILE_COUNTS_WIDTH
from stand.activation_kernel_checks.fake_recorder import (
    BUFFER_WORDS,
    QUANTILE_COUNTS_OFFSET,
    QUANTILE_WORDS,
    QUANTILES_OFFSET,
    WORDS_WITH_QUANTILES,
    FakeRecorder,
    make_recorder,
)
from stand.activation_kernel_checks.kernel_calls import observe_statistics, observe_with_quantiles
from stand.activation_kernel_checks.key_reference import (
    KEY_SCRATCH_WORDS,
    ONE_BLOCK_SELECTION,
    PAIR_COUNT_WORDS,
    SELECTION_DTYPES,
    group_totals,
    key_histogram,
    raw_pair_counts,
    select_from_key_histogram,
    selected_bits,
)
from stand.activation_kernel_checks.populations import (
    SPECIAL_POPULATION_SIZE,
    SPECIAL_ROW_WIDTH,
    SPECIAL_VALUE_SCALE,
    selection_populations,
    special_population,
)
from stand.activation_kernel_checks.statistics_oracle import (
    MAX_REPORTED_WORDS,
    compare,
    equal_with_nans,
    run_reference,
)

SELECTION_AND_FLOAT32_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
SIGNED_ZERO_ROWS = 300
SIGNED_ZERO_WIDTH = 300
SIGNED_ZERO_REPEATS = 500
MICROBATCH_ROWS = 2048
MICROBATCH_WIDTH = 896
MICROBATCH_SCALE = 3
HALVED_MICROBATCH_ROWS = 1500
FLIPPED_MICROBATCH_ROWS = 1024
HALVING_FACTOR = 0.5
BOUNDARY_SIZES = (ONE_BLOCK_SELECTION - 1, ONE_BLOCK_SELECTION, ONE_BLOCK_SELECTION + 1)
BOUNDARY_NAN_STRIDE = 997
BOUNDARY_TAIL_OFFSET = 70
MASK_VALID_PROBABILITY = 0.5
SPECIAL_INFINITIES_PER_SIGN = 25
SPECIAL_LARGEST_FINITES_PER_SIGN = 50
SPECIAL_NANS = 50


def check_raw_counts(populations: list[tuple[str, torch.Tensor]], extension) -> list[str]:
    """The scratch of a bf16/fp16 population has the CPU reference's counts of its raw values exactly; a
    smaller one has the first digit's scratch; the refining path's scratch is that of the first digit
    for every size."""
    problems = []
    for name, tensor in populations:
        if tensor.dtype not in SELECTION_DTYPES:
            continue
        matrix = tensor.reshape(-1, tensor.shape[-1])
        words = make_recorder(tensor.device).float_words
        scratch = observe_statistics(extension, matrix, words, True)
        refining = observe_statistics(extension, matrix, words, True, False)
        if refining.dtype != torch.int32:
            problems.append(f"raw counts {name}: the refining path's scratch is of {refining.dtype}, "
                            "not the first digit's")
        if tensor.numel() <= ONE_BLOCK_SELECTION:
            if scratch.numel() != refining.numel():
                problems.append(f"raw counts {name}: a one-block selection has a scratch of {scratch.numel()} words")
            continue
        if scratch.dtype != torch.int64 or scratch.numel() != KEY_SCRATCH_WORDS:
            problems.append(f"raw counts {name}: the scratch has {scratch.numel()} words of {scratch.dtype}, "
                            f"expected {KEY_SCRATCH_WORDS} of int64")
            continue
        histogram = scratch[:PAIR_COUNT_WORDS].cpu()
        expected = raw_pair_counts(tensor.cpu())
        if not torch.equal(histogram, expected):
            bad = (histogram != expected).nonzero().flatten().tolist()
            problems.append(
                f"raw counts {name}: differ at words {bad[:MAX_REPORTED_WORDS]}: "
                f"expected {expected[bad[:MAX_REPORTED_WORDS]].tolist()}, "
                f"got {histogram[bad[:MAX_REPORTED_WORDS]].tolist()}"
            )
        totals = scratch[PAIR_COUNT_WORDS:].cpu().view(torch.int32)
        expected_totals = group_totals(tensor.cpu())
        if not torch.equal(totals, expected_totals):
            bad = (totals != expected_totals).nonzero().flatten().tolist()
            problems.append(
                f"raw counts {name}: the totals of the groups of 32 words differ at {bad[:MAX_REPORTED_WORDS]}: "
                f"expected {expected_totals[bad[:MAX_REPORTED_WORDS]].tolist()}, "
                f"got {totals[bad[:MAX_REPORTED_WORDS]].tolist()}"
            )
    return problems


def check_quantile_first_pass(tensors: list[tuple[str, torch.Tensor]], extension) -> list[str]:
    """Exact quantiles selected from the first pass the statistics kernel counted equal the
    separate selection, the refining path (bit for bit) and the sorted reference, signed zeros,
    subnormals and non-finite values included. A bf16/fp16 population's equal the CPU reference's
    selection from its keys' histogram bit for bit."""
    device = tensors[0][1].device
    extra = _special_value_populations(device)
    problems = []
    for name, tensor in tensors + extra + selection_populations(device):
        problems.extend(_first_pass_problems(device, name, tensor, extension))
    return problems


def _special_value_populations(device: torch.device) -> list[tuple[str, torch.Tensor]]:
    special = torch.tensor([-0.0, 0.0, 1e-40, -1e-40, 3e-39, 1e-45, float("inf"), -float("inf"), float("nan")])
    extra = []
    for dtype in SELECTION_AND_FLOAT32_DTYPES:
        population = torch.cat(
            (torch.randn(SIGNED_ZERO_ROWS * SIGNED_ZERO_WIDTH) * SPECIAL_VALUE_SCALE,
             special.repeat(SIGNED_ZERO_REPEATS))
        ).to(dtype)
        extra.append((f"signed zeros and subnormals {dtype}", population.reshape(-1, SIGNED_ZERO_WIDTH).to(device)))
        special_values = special_population(SPECIAL_POPULATION_SIZE, dtype, device)
        extra.append((f"largest finite beside non-finite {dtype}", special_values.reshape(-1, SPECIAL_ROW_WIDTH)))
    return extra


def _first_pass_problems(device: torch.device, name: str, tensor: torch.Tensor, extension) -> list[str]:
    problems = []
    matrix = tensor.reshape(-1, tensor.shape[-1])
    first_pass = observe_statistics(extension, matrix, make_recorder(device).float_words, True)
    flat = tensor.reshape(-1).contiguous()
    fused = (torch.empty(QUANTILE_WORDS, device=device),
             torch.empty(QUANTILE_COUNTS_WIDTH, dtype=torch.int32, device=device))
    separate = (torch.empty(QUANTILE_WORDS, device=device),
                torch.empty(QUANTILE_COUNTS_WIDTH, dtype=torch.int32, device=device))
    extension.exact_quantiles(tensor, None, *fused, first_pass)
    extension.exact_quantiles(flat, None, *separate, None)
    # The same selection inside the statistics call, written into the recorder's words, with the
    # keys' histogram and without it (the first digit counted, then refined).
    in_call, refined = [], []
    for with_key_histogram, selections in ((True, in_call), (False, refined)):
        words = torch.zeros(WORDS_WITH_QUANTILES, device=device)
        observe_with_quantiles(extension, tensor, words, with_key_histogram)
        selections += [words[QUANTILES_OFFSET:QUANTILE_COUNTS_OFFSET],
                       words[QUANTILE_COUNTS_OFFSET:].view(torch.int32)]
    expected, expected_counts = exact_quantiles_aten(flat)
    for label, (quantiles, counts) in (("first pass", fused), ("separate", separate),
                                       ("in the statistics call", in_call), ("refined", refined)):
        if not (equal_with_nans(quantiles, expected) and torch.equal(counts, expected_counts)):
            problems.append(f"quantiles {label} {name}: {quantiles.tolist()} {counts.tolist()} vs "
                            f"{expected.tolist()} {expected_counts.tolist()}")
    for label, (quantiles, counts) in (("first pass", fused), ("in the statistics call", in_call)):
        if not (torch.equal(selected_bits(quantiles), selected_bits(refined[0]))
                and torch.equal(counts, refined[1])):
            problems.append(f"quantiles {label} {name} differ in bits from the refining path: {quantiles.tolist()} "
                            f"vs {refined[0].tolist()}")
        if tensor.dtype in SELECTION_DTYPES:
            model = select_from_key_histogram(key_histogram(flat.cpu()), tensor.dtype)
            if not torch.equal(selected_bits(quantiles.cpu()), selected_bits(model)):
                problems.append(f"quantiles {label} {name} differ in bits from the CPU reference: "
                                f"{quantiles.tolist()} vs {model.tolist()}")
    return problems


def check_quantile_microbatches(device: torch.device, extension) -> list[str]:
    """Consecutive microbatches observed with their quantiles into one word buffer, as the recorder's
    slots, each record the statistics and the exact quantiles of its own tensor: the scratch of one
    observation never reaches the next."""
    problems = []
    stride = WORDS_WITH_QUANTILES
    for dtype in SELECTION_DTYPES:
        base = (torch.randn(MICROBATCH_ROWS, MICROBATCH_WIDTH, device=device) * MICROBATCH_SCALE).to(dtype)
        microbatches = (
            base, base[:HALVED_MICROBATCH_ROWS] * HALVING_FACTOR, base[:FLIPPED_MICROBATCH_ROWS].flip(0) + 1, base[::2]
        )
        words = torch.zeros(len(microbatches) * stride, device=device)
        for microbatch, value in enumerate(microbatches):
            observe_with_quantiles(extension, value, words, offset=microbatch * stride)
        torch.cuda.synchronize()
        for microbatch, value in enumerate(microbatches):
            offset = microbatch * stride
            label = f"quantile microbatch={microbatch} {dtype}"
            problems.extend(compare(label, run_reference(value, False),
                                    FakeRecorder(words[offset:offset + BUFFER_WORDS])))
            expected, expected_counts = exact_quantiles_aten(value.reshape(-1))
            quantiles = words[offset + QUANTILES_OFFSET:offset + QUANTILE_COUNTS_OFFSET]
            counts = words[offset + QUANTILE_COUNTS_OFFSET:offset + stride].view(torch.int32)
            if not (equal_with_nans(quantiles, expected) and torch.equal(counts, expected_counts)):
                problems.append(f"{label}: {quantiles.tolist()} {counts.tolist()} vs "
                                f"{expected.tolist()} {expected_counts.tolist()}")
    return problems


def check_selection_boundaries(device: torch.device) -> list[str]:
    """Exact quantiles and order statistics of populations around the size where selection
    changes kernels equal the sorted reference, NaN last."""
    problems = []
    for size in BOUNDARY_SIZES:
        for dtype in SELECTION_AND_FLOAT32_DTYPES:
            values = torch.randn(size, device=device).to(dtype)
            values[::BOUNDARY_NAN_STRIDE] = float("nan")
            quantiles, counts = exact_quantiles(values)
            expected, expected_counts = exact_quantiles_aten(values)
            if not (equal_with_nans(quantiles, expected) and torch.equal(counts, expected_counts)):
                problems.append(f"quantiles of {size} {dtype}: {quantiles.tolist()} vs {expected.tolist()}")
            ranks = [0, size // 3, size // 2, size - BOUNDARY_TAIL_OFFSET, size - 1]
            selected = order_statistics(values, ranks)
            reference = torch.sort(values).values[ranks].float()
            if not equal_with_nans(selected, reference):
                problems.append(f"order statistics of {size} {dtype}: {selected.tolist()} vs {reference.tolist()}")
    return problems


def check_selection_edges(device: torch.device) -> list[str]:
    """Masked quantiles, order statistics among infinities and NaN of either sign (NaN last, like
    torch.sort), and quantiles of a population without a finite value equal the sorted reference
    for populations selected by several blocks."""
    problems = []
    size = SPECIAL_POPULATION_SIZE
    for dtype in SELECTION_AND_FLOAT32_DTYPES:
        values = special_population(size, dtype, device)
        valid = torch.rand(size, device=device) < MASK_VALID_PROBABILITY
        nonfinite = torch.where(values.isfinite(), torch.full_like(values, float("nan")), values)
        for label, population, mask in (("masked", values, valid), ("non-finite only", nonfinite, None)):
            quantiles, counts = exact_quantiles(population, mask)
            expected, expected_counts = exact_quantiles_aten(population, mask)
            if not (equal_with_nans(quantiles, expected) and torch.equal(counts, expected_counts)):
                problems.append(f"quantiles {label} {dtype}: {quantiles.tolist()} {counts.tolist()} vs "
                                f"{expected.tolist()} {expected_counts.tolist()}")
        # Sorted: 25 -inf, 50 -max, the normal values, 50 +max, 25 +inf, 50 NaN. torch.sort puts a
        # negative NaN first, hence the reference sorts NaN of either sign as +NaN.
        ranks = [
            0,
            SPECIAL_INFINITIES_PER_SIGN,
            size // 2,
            size - (SPECIAL_LARGEST_FINITES_PER_SIGN + SPECIAL_INFINITIES_PER_SIGN + SPECIAL_NANS),
            size - (SPECIAL_INFINITIES_PER_SIGN + SPECIAL_NANS),
            size - SPECIAL_NANS,
            size - 1,
        ]
        selected = order_statistics(values, ranks)
        canonical = torch.where(values.isnan(), torch.full_like(values, float("nan")), values)
        reference = torch.sort(canonical).values[ranks].float()
        if not equal_with_nans(selected, reference):
            problems.append(f"order statistics among non-finite {dtype}: {selected.tolist()} vs {reference.tolist()}")
    return problems
