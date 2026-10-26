"""Foundry capacity selection over token/expert pairs, before communication."""

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from torchtitan.config.moe import TokenDropConfig


def _branchless_capacity_mask(
    weights: torch.Tensor,
    buckets: torch.Tensor,
    valid: torch.Tensor,
    counts: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Rank routed pairs by expert and descending score, with stable token ties."""
    if weights.dtype == torch.float64:
        # Packing float32 bits would turn distinct float64 scores into ties.
        by_score = torch.argsort(weights, descending=True, stable=True)
        order = by_score[torch.argsort(buckets[by_score], stable=True)]
    else:
        # Float16/bfloat16 promote exactly to float32. Map signed IEEE bits to
        # increasing unsigned order; each expert occupies the upper 32 bits.
        bits = weights.float().contiguous().view(torch.int32).to(torch.int64)
        ordered_bits = torch.where(bits < 0, ~bits, bits ^ 0x80000000)
        key = (buckets << 32) | (0xFFFFFFFF - ordered_bits)
        order = torch.argsort(key, stable=True)
    starts = counts.cumsum(0) - counts
    positions = torch.arange(weights.numel(), device=weights.device)
    keep_sorted = (positions - starts[buckets[order]] < capacity) & valid[order]
    keep = torch.empty_like(keep_sorted)
    keep.scatter_(0, order, keep_sorted)
    return keep


def _sorted_capacity_mask(
    scores: torch.Tensor,
    buckets: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_experts: int,
    capacity: int,
) -> torch.Tensor:
    """Stable per-expert selection, returning a mask in compact router layout."""
    # The last column receives inactive slots, so they cannot overwrite a live
    # route. -inf also keeps absent routes behind any negative routed scores.
    probs = scores.new_full((scores.shape[0], num_experts + 1), -torch.inf)
    probs.scatter_(1, buckets.view_as(scores), scores)
    top_indices = torch.argsort(
        probs[:, :num_experts], dim=0, descending=True, stable=True
    )[:capacity]
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[:, :num_experts].scatter_(0, top_indices, True)
    return mask.gather(1, buckets.view_as(scores)).reshape(-1) & valid


@torch.library.custom_op(
    "torchtitan::conditional_capacity_mask",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _conditional_capacity_mask(
    scores: torch.Tensor,
    buckets: torch.Tensor,
    valid: torch.Tensor,
    needs_drop: torch.Tensor,
    num_experts: int,
    capacity: int,
) -> torch.Tensor:
    """Keep the CPU overload check opaque to fullgraph tracing and checkpointing.

    torch.cond is unsupported by the tested PyTorch checkpoint dispatch mode.
    This operator keeps selection on device and synchronizes only the predicate;
    its boolean output is nondifferentiable. The host check prevents CUDA graph
    capture, so the operator is tagged accordingly.
    """
    if needs_drop:
        return _sorted_capacity_mask(
            scores, buckets, valid, num_experts=num_experts, capacity=capacity
        )
    # Custom operator outputs must not alias their inputs.
    return valid.clone()


@_conditional_capacity_mask.register_fake
def _conditional_capacity_mask_fake(
    scores: torch.Tensor,
    buckets: torch.Tensor,
    valid: torch.Tensor,
    needs_drop: torch.Tensor,
    num_experts: int,
    capacity: int,
) -> torch.Tensor:
    return torch.empty_like(valid)


@torch.no_grad()
def select_capacity_routes(
    scores: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    num_experts: int,
    policy: TokenDropConfig,
    ep_group: dist.ProcessGroup | None = None,
    max_num_tokens: int | None = None,
) -> torch.Tensor:
    """Keep highest-scoring routes within source-local expert quotas.

    Zero scores and negative IDs are inactive. Conditional selection checks
    overload on the host; branchless selection stays on device. Group gating
    requires both local expert and destination group overload.
    """
    if scores.ndim != 2 or expert_ids.shape != scores.shape:
        raise ValueError("Capacity routing expects matching [tokens, top_k] tensors")
    ep_size = ep_group.size() if ep_group is not None else 1
    capacity = policy.source_capacity(
        num_tokens=scores.shape[0],
        top_k=scores.shape[1],
        num_experts=num_experts,
        ep_size=ep_size,
        max_num_tokens=max_num_tokens,
    )
    ids = expert_ids.reshape(-1).long()
    weights = scores.detach().reshape(-1)
    valid = (ids >= 0) & (weights != 0)
    buckets = torch.where(valid, ids, num_experts)
    counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=ids.device)
    counts.scatter_add_(0, buckets, torch.ones_like(buckets))
    needs_drop = (counts[:num_experts] > capacity).any()

    if policy.drop_by_group_capacity:
        # Match Foundry: reduce loads, but keep this source's capacity. Empty
        # sources also enter the collective before either path can return.
        totals = counts[:num_experts]
        if ep_group is not None:
            # Functional collectives are saved by SAC and remain visible to
            # compile even on an empty source whose mask does not use the gate.
            totals = funcol.all_reduce(totals, "sum", ep_group)
        loads = totals.view(ep_size, num_experts // ep_size).sum(1)
        budget = capacity * num_experts
        needs_drop = needs_drop & (loads > budget).any()

    if policy.branchless_capacity_drop:
        # Keep this after the group collective: peers may still need to drop.
        num_routes = weights.numel()
        if num_routes == 0 or capacity >= num_routes:
            return valid.view_as(expert_ids)
        if weights.is_cuda and weights.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            from torchtitan.ops.token_drop import MAX_RADIX_ROUTES, capacity_mask

            if num_routes <= MAX_RADIX_ROUTES:
                return capacity_mask(
                    weights, buckets, valid, counts, needs_drop, capacity
                ).view_as(expert_ids)
        keep = _branchless_capacity_mask(weights, buckets, valid, counts, capacity)
        keep = torch.where(needs_drop, keep, valid)
    elif torch.compiler.is_compiling():
        keep = _conditional_capacity_mask(
            scores, buckets, valid, needs_drop, num_experts, capacity
        )
    elif needs_drop:
        keep = _sorted_capacity_mask(
            scores, buckets, valid, num_experts=num_experts, capacity=capacity
        )
    else:
        keep = valid
    return keep.view_as(expert_ids)
