"""Host-side bounds for HybridEP's permuted receive layout.

These helpers use only tensor shapes and configuration, never device counts.
They can validate a model before loading the optional HybridEP library.
"""


def non_blocking_buffer_capacity(
    num_tokens: int,
    ep_size: int,
    num_local_experts: int,
    top_k: int,
    capacity_factor: float,
    pad_multiple: int | None = None,
) -> int:
    """Match HybridEP's allocated row count, including its final alignment."""
    rows = int(num_tokens * ep_size * min(num_local_experts, top_k) * capacity_factor)
    alignment = 1 if pad_multiple is None else pad_multiple
    return ((rows + alignment - 1) // alignment) * alignment


def retained_receive_capacity(
    *,
    num_tokens: int,
    source_capacity: int,
    ep_size: int,
    num_local_experts: int,
    top_k: int,
    drop_by_group_capacity: bool,
    pad_multiple: int | None = None,
) -> int:
    """Bound padded rows at a receiver for equal-sized, equally configured sources.

    Strict dropping bounds each expert. With the group gate, an individual
    expert may exceed that bound, but the destination's total still cannot
    exceed the budget: the gate either admits that total or enables clipping
    on all overloaded sources. This argument requires equal source quotas.
    """
    alignment = 1 if pad_multiple is None else pad_multiple
    per_expert = ep_size * min(num_tokens, source_capacity)
    routes = min(
        ep_size * num_tokens * min(num_local_experts, top_k),
        per_expert * num_local_experts,
    )
    if routes == 0:
        return 0
    if drop_by_group_capacity:
        per_expert = ep_size * num_tokens
    # Distribute at most `routes` assignments over nonempty experts. Opening
    # each expert costs one token and one alignment block; every additional
    # block needs `alignment` tokens. Also respect the per-expert bound.
    nonempty = min(routes, num_local_experts)
    total_bound = alignment * (nonempty + (routes - nonempty) // alignment)
    expert_bound = (
        num_local_experts * ((per_expert + alignment - 1) // alignment) * alignment
    )
    return min(total_bound, expert_bound)
