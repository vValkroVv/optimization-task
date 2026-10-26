from __future__ import annotations

import math

import torch

from metric_sampler.capture.stats.helpers.norms import (
    accumulate_row_norm_stats,
    accumulate_row_norms,
    accumulate_row_norms_batch,
    amax_and_row_norm_sum,
)

ROW_NORM_SUM_TOLERANCE = 1e-6
ACCUMULATION_SLOTS = 3
ACCUMULATION_SCALES = (1.0, 3.0, 0.5)
UNKEPT_ROUNDS = 8
BATCH_LAUNCH_TENSORS = 32
TENSORS_BEYOND_ONE_LAUNCH = 70
SLOTS_BEYOND_ONE_LAUNCH = 60
MULTI_SLOT_TENSORS = 3
MULTI_SLOT_PERIOD = 5
MULTI_SLOT_REMAINDER = 2
BATCHED_TENSOR_COUNT = 75
BATCHED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
BATCHED_SHAPES = (
    (1, 13), (7, 64), (130, 96), (1000, 128), (4096, 896), (3000, 2304), (20000, 64), (9, 7168), (77, 301),
)
BATCHED_MISALIGNED_PERIOD = 4
BATCHED_INITIAL_SLOT_VALUE = 1.5
BATCHED_ROUNDS = 2
REPORTED_BATCH_WORDS = 4


def empty_accumulator(slots: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([[-math.inf, 0.0]] * slots, dtype=torch.float64, device=device)


def check_layout_row_norms(label: str, activations: torch.Tensor, rows: int, device: torch.device) -> list[str]:
    problems = []
    amax, norm_sum, counted = amax_and_row_norm_sum(activations)
    expected_sum = torch.linalg.vector_norm(activations, dim=-1, dtype=torch.float32).double().sum()
    if amax.item() != activations.float().abs().max().item() or counted != rows:
        problems.append(f"row norms {label}: amax or row count differs")
    if abs(norm_sum.item() - expected_sum.item()) > ROW_NORM_SUM_TOLERANCE * expected_sum.item():
        problems.append(f"row norms {label}: sum {norm_sum.item()} vs {expected_sum.item()}")
    accumulator = empty_accumulator(1, device)
    accumulate_row_norms(activations, accumulator, 0)
    if accumulator[0].tolist() != [amax.item(), norm_sum.item()]:
        problems.append(f"accumulated row norms {label}: {accumulator[0].tolist()}")
    return problems


def check_nan_propagation(device: torch.device) -> list[str]:
    problems = []
    activations = torch.randn(64, 32, device=device)
    activations[5, 3] = float("nan")
    amax, norm_sum, _ = amax_and_row_norm_sum(activations)
    if not (amax.isnan() and norm_sum.isnan()):
        problems.append("row norms: NaN did not propagate")
    return problems


def check_accumulation(device: torch.device) -> list[str]:
    """Accumulated statistics equal the observations' own: maximum of float32 maxima, sum of the
    row-norm sums with each observation added first, NaN propagating."""
    accumulator = empty_accumulator(ACCUMULATION_SLOTS, device)
    expected = [[-math.inf, 0.0] for _ in range(ACCUMULATION_SLOTS)]
    observations = [
        torch.randn(4096, 896, device=device).to(torch.bfloat16) * scale for scale in ACCUMULATION_SCALES
    ]
    observations.append(torch.randn(77, 13, device=device, dtype=torch.float16))
    with_nan = torch.randn(64, 32, device=device)
    with_nan[3, 4] = float("nan")
    for slot, value in [(0, observations[0]), (0, observations[1]), (1, observations[2]),
                        (0, observations[3]), (2, with_nan)]:
        stats = accumulate_row_norms(value, accumulator, slot)
        accumulate_row_norm_stats(stats, accumulator, 1)
        amax, norm_sum, _ = amax_and_row_norm_sum(value)
        for target in (slot, 1):
            previous_amax, previous_sum = expected[target]
            expected[target] = [max(previous_amax, amax.item()) if not math.isnan(amax.item()) else math.nan,
                                norm_sum.item() + previous_sum]
    actual = accumulator.tolist()
    for slot in range(ACCUMULATION_SLOTS):
        same = all(
            actual_value == expected_value or (math.isnan(actual_value) and math.isnan(expected_value))
            for actual_value, expected_value in zip(actual[slot], expected[slot])
        )
        if not same:
            return [f"accumulated row norms slot {slot}: {actual[slot]} vs {expected[slot]}"]
    return []


def check_unkept_accumulation(device: torch.device) -> list[str]:
    """Statistics a launch does not keep are accumulated bit for bit like those it keeps, launch after
    launch on one scratch and on another stream's, with NaN, strided rows and every dtype between them."""
    values = [torch.randn(4096, 896, device=device).to(torch.bfloat16), torch.randn(77, 13, device=device).half(),
              torch.randn(1000, 2304, device=device), torch.randn(3000, 96, device=device)[:, 1:90],
              torch.randn(9, 7168, device=device).to(torch.bfloat16), torch.randn(1, 1, device=device)]
    values[3][5, 7] = float("nan")
    kept = empty_accumulator(1, device)
    unkept = empty_accumulator(2, device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    for value in values * UNKEPT_ROUNDS:
        accumulate_row_norms(value, kept, 0)
        accumulate_row_norms(value, unkept, 0, keep_stats=False)
        with torch.cuda.stream(side):
            accumulate_row_norms(value, unkept, 1, keep_stats=False)
    torch.cuda.current_stream().wait_stream(side)
    return [f"unkept row norms slot {slot}: {unkept[slot].tolist()} vs {kept[0].tolist()}"
            for slot in range(2) if not torch.equal(unkept[slot].view(torch.int64), kept[0].view(torch.int64))]


def check_batched_accumulation(device: torch.device) -> list[str]:
    """`accumulate_row_norms_batch` takes the statistics of every tensor into its slots bit for bit as a
    launch per tensor and one per further slot do: with every dtype, NaN, strided and misaligned rows, tensors
    with one slot and with several, slots that already hold observations, the second round merging into the
    first, and lists that need several launches per dtype, by their number of tensors and by their slots."""
    cases = _batched_cases(device)
    problems = []
    for label, case_values, counts in cases:
        for length in sorted({1, 2, BATCH_LAUNCH_TENSORS - 1, BATCH_LAUNCH_TENSORS, BATCH_LAUNCH_TENSORS + 1,
                              len(case_values)}):
            slots, next_slot = [], 0
            for count in counts[:length]:
                slots.append(list(range(next_slot, next_slot + count)))
                next_slot += count
            single = empty_accumulator(next_slot, device)
            single[0] = BATCHED_INITIAL_SLOT_VALUE
            batched = single.clone()
            for _ in range(BATCHED_ROUNDS):
                for value, value_slots in zip(case_values[:length], slots):
                    stats = accumulate_row_norms(value, single, value_slots[0])
                    for slot in value_slots[1:]:
                        accumulate_row_norm_stats(stats, single, slot)
                accumulate_row_norms_batch(case_values[:length], batched, slots)
            torch.cuda.synchronize()
            if not torch.equal(single.view(torch.int64), batched.view(torch.int64)):
                bad = (single.view(torch.int64) != batched.view(torch.int64)).nonzero().flatten()[
                    :REPORTED_BATCH_WORDS].tolist()
                problems.append(f"batched row norms, {label}, {length} tensors, "
                                f"differ from the single launches at words {bad}")
    return problems


def _batched_cases(device: torch.device) -> list[tuple[str, list[torch.Tensor], list[int]]]:
    values = []
    for index in range(BATCHED_TENSOR_COUNT):
        rows, width = BATCHED_SHAPES[index % len(BATCHED_SHAPES)]
        dtype = BATCHED_DTYPES[index % len(BATCHED_DTYPES)]
        value = (torch.randn(rows, width + 1, device=device) * (1.0 + index)).to(dtype)
        values.append(value[:, 1:] if index % BATCHED_MISALIGNED_PERIOD == 0 else value[:, :width])
    values[10][3, 5] = float("nan")
    values[23][0, 0] = float("inf")
    values[41][-1, -1] = float("-inf")
    values[52] = values[52].t().contiguous().t()
    slot_counts = [
        MULTI_SLOT_TENSORS if index % MULTI_SLOT_PERIOD == MULTI_SLOT_REMAINDER else 1 for index in range(len(values))
    ]
    cases = [("mixed dtypes", values, slot_counts)]
    many = [
        (torch.randn(130 + index, 96, device=device) * (1.0 + index)).to(torch.bfloat16)
        for index in range(TENSORS_BEYOND_ONE_LAUNCH)
    ]
    cases.append((f"{TENSORS_BEYOND_ONE_LAUNCH} bfloat16 tensors", many, [1] * len(many)))
    cases.append((f"tensors with {SLOTS_BEYOND_ONE_LAUNCH} slots each", many[:5], [SLOTS_BEYOND_ONE_LAUNCH] * 5))
    return cases
