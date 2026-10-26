"""Pure parallel-degree geometry shared by pre-submit and runtime meshes."""

from math import prod


def sequence_length_divisor(*, tp: int, cp: int) -> int:
    """Native decoder partitioning: sequence parallel TP and balanced CP halves."""
    return tp * (2 * cp if cp > 1 else 1)


def validate_expert_parallel_degree(
    *, dp_shard: int, cp: int, tp: int, ep: int
) -> None:
    """EP must fit the storage ranks inside one data-parallel replica."""
    for name, degree in (("dp_shard", dp_shard), ("cp", cp), ("tp", tp), ("ep", ep)):
        if degree < 1:
            raise ValueError(f"{name}: parallelism degree must be >= 1, got {degree}")
    storage_ranks = dp_shard * cp * tp
    if storage_ranks % ep:
        raise ValueError(
            f"expert_parallel_degree {ep} must divide the data-sharding storage "
            f"ranks dp_shard * cp * tp ({storage_ranks})"
        )


def resolve_data_parallel_shard_degree(
    *,
    dp_replicate: int,
    dp_shard: int,
    cp: int,
    tp: int,
    pp: int,
    ep: int,
    world_size: int,
) -> int:
    """Validate world-size geometry and resolve ``dp_shard=-1``.

    EP shares the data-sharding ranks and does not multiply the world size.
    Model-specific expert divisibility checks run separately.
    """
    for name, degree in (
        ("dp_replicate", dp_replicate),
        ("cp", cp),
        ("tp", tp),
        ("pp", pp),
        ("ep", ep),
    ):
        if degree < 1:
            raise ValueError(f"{name}: parallelism degree must be >= 1, got {degree}")
    if dp_shard != -1 and dp_shard < 1:
        raise ValueError(f"dp_shard: must be -1 or >= 1, got {dp_shard}")
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    explicit_product = prod((dp_replicate, cp, tp, pp))
    if dp_shard == -1:
        if world_size % explicit_product:
            raise ValueError(
                f"WORLD_SIZE {world_size} is not divisible by the "
                f"product of explicit parallel degrees {explicit_product}"
            )
        dp_shard = world_size // explicit_product
    product = explicit_product * dp_shard
    if product != world_size:
        raise ValueError(
            f"product of parallel degrees {product} != WORLD_SIZE {world_size}"
        )
    validate_expert_parallel_degree(dp_shard=dp_shard, cp=cp, tp=tp, ep=ep)
    return dp_shard
