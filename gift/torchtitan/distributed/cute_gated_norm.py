"""DTensor strategies for the self-gated CuTe norm's opaque operators."""

from torch.distributed.tensor import Partial, Placement, Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding


def register_cute_gated_norm_sharding(forward_op, backward_op):
    """Allow row sharding on every mesh axis, with replicated gate parameters.

    Hidden sharding requires redistribution: each gate sees the whole hidden
    vector. Parameter gradients over sharded rows are Partial sums, just like
    the native linear/reduction path, so DTensor performs the needed reductions.
    """

    replicate: Placement = Replicate()
    partial: Placement = Partial()

    @register_sharding(forward_op)
    def forward_strategy(
        x, weight, gate_up_weight, gate_down_weight, eps, gating_weight
    ):
        strategies: list[tuple[list[Placement], list[Placement | None]]] = [
            ([replicate] * 7, [replicate] * 4 + [None, None])
        ]
        for dim in range(x.ndim - 1):
            shard: Placement = Shard(dim)
            strategies.append(
                (
                    [shard] * 3 + [replicate] * 4,
                    [shard] + [replicate] * 3 + [None, None],
                )
            )
        return strategies

    @register_sharding(backward_op)
    def backward_strategy(
        dy,
        x,
        rho,
        p,
        w_f,
        gd_bf,
        gd_t,
        gu_t,
        gating_weight,
        weight_dtype,
        up_dtype,
        down_dtype,
    ):
        strategies: list[tuple[list[Placement], list[Placement | None]]] = [
            ([replicate] * 4, [replicate] * 8 + [None] * 4)
        ]
        for dim in range(x.ndim - 1):
            shard: Placement = Shard(dim)
            strategies.append(
                (
                    [shard] + [partial] * 3,
                    [shard] * 4 + [replicate] * 4 + [None] * 4,
                )
            )
        return strategies
