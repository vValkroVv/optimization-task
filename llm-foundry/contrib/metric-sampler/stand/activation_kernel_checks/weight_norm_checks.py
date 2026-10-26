from __future__ import annotations

import torch

from metric_sampler.capture.stats.helpers.norms import weight_norm_rows

BFLOAT16_NORM_TOLERANCE = 2 ** -7
FLOAT16_NORM_TOLERANCE = 2 ** -10
FLOAT32_NORM_TOLERANCE = 1e-6
SMALL_WEIGHTS = 70
NORM_WORDS = 3
EXPERTS = 8
EXPERT_ROWS = 64
EXPERT_COLUMNS = 96


def check_weight_norms(device: torch.device) -> list[str]:
    """Weight norms equal torch's in the weight's dtype: the absolute max exactly, the L2 norm
    within one rounding of that dtype."""
    weights = [torch.randn(896, 2304, device=device).to(torch.bfloat16), torch.randn(896, device=device) * 0.1,
               torch.randn(7, device=device).half(), torch.randn(8192, 896, device=device).to(torch.bfloat16)]
    weights += [torch.randn(13, device=device) for _ in range(SMALL_WEIGHTS)]
    rows = weight_norm_rows(weights)
    amax = torch.stack([weight.abs().max().float() for weight in weights])
    l2_norms = torch.stack([torch.linalg.vector_norm(weight).float() for weight in weights])
    tolerance = torch.tensor(
        [BFLOAT16_NORM_TOLERANCE if weight.dtype == torch.bfloat16
         else (FLOAT16_NORM_TOLERANCE if weight.dtype == torch.float16 else FLOAT32_NORM_TOLERANCE)
         for weight in weights], device=device)
    problems = []
    if not torch.equal(rows[:, 0], amax) or not torch.equal(rows[:, 2], torch.ones_like(amax)):
        problems.append("weight norms: absolute max or presence differs")
    if ((rows[:, 1] - l2_norms).abs() > tolerance * l2_norms).any():
        problems.append(
            f"weight norms: L2 beyond one rounding of the dtype: {(rows[:, 1] - l2_norms).abs().max().item()}"
        )
    nan_weight = torch.randn(100, device=device)
    nan_weight[7] = float("nan")
    if not weight_norm_rows([nan_weight])[0, :2].isnan().all():
        problems.append("weight norms: NaN did not propagate")
    return problems


def check_weight_norm_batches(device: torch.device, extension) -> list[str]:
    """Weights of several batches recorded in one call equal each batch recorded alone, bit for bit.
    Constant fp32 data makes the number of blocks a weight is reduced by show in its norm."""
    batches = [
        [torch.full((5_000_000,), 0.1, device=device), torch.full((100_000,), 0.1, device=device)],
        [torch.full((300_000,), 0.7, device=device), torch.full((30_000,), 0.7, device=device)],
        [torch.randn(896, 2304, device=device).to(torch.bfloat16), torch.randn(896, device=device) * 0.1],
        [torch.randn(7, device=device).half()],
        [torch.randn(50_000, device=device) for _ in range(100)],
    ]
    weights = [weight for batch in batches for weight in batch]
    launch_numels = [max(weight.numel() for weight in batch) for batch in batches for _ in batch]
    grouped = torch.zeros(NORM_WORDS * len(weights), device=device)
    extension.weight_norms_into(
        weights, [0] * len(weights), grouped, list(range(0, NORM_WORDS * len(weights), NORM_WORDS)), launch_numels
    )
    alone = torch.cat([extension.weight_norms(batch) for batch in batches])
    different = (
        (grouped.view(-1, NORM_WORDS).view(torch.int32) != alone.view(torch.int32)).any(dim=1).nonzero().flatten()
    )
    problems = [f"weight norms of batch member {index} differ between a grouped call and its batch alone"
                for index in different.tolist()]
    own_sizes = torch.zeros(NORM_WORDS * len(weights), device=device)
    extension.weight_norms_into(
        weights, [0] * len(weights), own_sizes, list(range(0, NORM_WORDS * len(weights), NORM_WORDS)),
        [single_weight.numel() for single_weight in weights])
    if torch.equal(own_sizes, grouped):
        problems.append("weight norms: sizing every weight alone did not change a norm, "
                        "so this check cannot tell block counts apart")
    problems.extend(_check_expert_stacks(device, extension))
    return problems


def _check_expert_stacks(device: torch.device, extension) -> list[str]:
    # Expert stacks -- contiguous, strided along the experts, and one whose view as [experts, -1] is a copy -- give
    # each expert the words of its row recorded as a weight of its own.
    problems = []
    stacks = [torch.randn(EXPERTS, EXPERT_ROWS, EXPERT_COLUMNS, device=device).to(torch.bfloat16),
              torch.randn(2 * EXPERTS, EXPERT_ROWS * EXPERT_COLUMNS, device=device).to(torch.bfloat16)[::2],
              torch.randn(EXPERT_COLUMNS, EXPERTS * EXPERT_ROWS, device=device).to(torch.bfloat16).t()]
    experts = [row for stack in stacks for row in stack.reshape(EXPERTS, -1).unbind(0)]
    offsets = list(range(0, NORM_WORDS * len(experts), NORM_WORDS))
    stacked, unstacked = torch.zeros(2, NORM_WORDS * len(experts), device=device)
    expert_numels = [EXPERT_ROWS * EXPERT_COLUMNS] * len(experts)
    took = (extension.weight_norms_into(stacks, [EXPERTS] * len(stacks), stacked, offsets, expert_numels),
            extension.weight_norms_into(experts, [0] * len(experts), unstacked, offsets, expert_numels))
    if took != (True, True) or not torch.equal(stacked.view(torch.int32), unstacked.view(torch.int32)):
        problems.append("weight norms: experts of a stack differ from the same rows recorded as weights")
    untouched = torch.zeros(NORM_WORDS, device=device)
    if extension.weight_norms_into(
        [stacks[0][0].t()], [0], untouched, [0], [EXPERT_ROWS * EXPERT_COLUMNS]
    ) or untouched.any():
        problems.append("weight norms: a non-contiguous weight was taken")
    return problems
