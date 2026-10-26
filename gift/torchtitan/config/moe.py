"""Lightweight capacity-policy configuration for MoE dispatch."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class TokenDropConfig:
    """Per-source expert quotas for token dropping.

    ``capacity_factor`` scales the observed routes per expert.
    ``peak_capacity_factor`` uses the maximum tokens per forward derived from
    the training layout.
    ``fixed_capacity`` splits explicit ``expert_capacity_tokens`` across EP ranks.
    """

    strategy: Literal["capacity_factor", "fixed_capacity", "peak_capacity_factor"] = (
        "capacity_factor"
    )
    expert_capacity_factor: float | None = None
    expert_capacity_tokens: int | None = None
    drop_by_group_capacity: bool = False
    branchless_capacity_drop: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.branchless_capacity_drop, bool):
            raise ValueError("token_drop.branchless_capacity_drop must be a boolean")
        if self.strategy not in (
            "capacity_factor",
            "fixed_capacity",
            "peak_capacity_factor",
        ):
            raise ValueError(f"Unknown token_drop.strategy: {self.strategy!r}")
        if not isinstance(self.drop_by_group_capacity, bool):
            raise ValueError("token_drop.drop_by_group_capacity must be a boolean")
        if self.strategy == "fixed_capacity":
            if self.expert_capacity_factor is not None:
                raise ValueError(
                    "token_drop.fixed_capacity requires expert_capacity_tokens and "
                    "does not accept expert_capacity_factor. Use "
                    "strategy='peak_capacity_factor' for a quota derived from the "
                    "training layout."
                )
            if (
                type(self.expert_capacity_tokens) is not int
                or self.expert_capacity_tokens <= 0
            ):
                raise ValueError(
                    "token_drop.fixed_capacity requires a positive integer "
                    "expert_capacity_tokens"
                )
        else:
            if self.expert_capacity_tokens is not None:
                raise ValueError("expert_capacity_tokens requires fixed_capacity")
            factor = self.expert_capacity_factor
            if (
                isinstance(factor, bool)
                or not isinstance(factor, (float, int))
                or not math.isfinite(factor)
                or factor <= 0
            ):
                raise ValueError(
                    "token_drop.expert_capacity_factor must be positive and finite"
                )

    def source_capacity(
        self,
        *,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        ep_size: int,
        max_num_tokens: int | None = None,
    ) -> int:
        """Compute a source quota using the dispatcher's derived token bound.

        ``max_num_tokens`` comes from the training layout and is required only
        for peak_capacity_factor. It is not serialized in the config.
        """
        if (
            any(type(v) is not int for v in (num_tokens, top_k, num_experts, ep_size))
            or num_tokens < 0
            or top_k <= 0
            or num_experts <= 0
            or ep_size <= 0
            or top_k > num_experts
        ):
            raise ValueError("Invalid token-drop token, expert, top-k, or EP size")
        if num_experts % ep_size:
            raise ValueError("Token dropping requires num_experts divisible by ep_size")
        if self.strategy == "fixed_capacity":
            assert self.expert_capacity_tokens is not None
            return self.expert_capacity_tokens // ep_size
        assert self.expert_capacity_factor is not None
        if self.strategy == "capacity_factor":
            return int(
                (num_tokens * top_k // num_experts) * self.expert_capacity_factor
            )
        if max_num_tokens is None:
            raise ValueError(
                "peak_capacity_factor requires the training layout to resolve "
                "the token count at the local dispatcher boundary"
            )
        if type(max_num_tokens) is not int or max_num_tokens <= 0:
            raise ValueError(
                "The derived maximum tokens per forward must be a positive integer"
            )
        if num_tokens > max_num_tokens:
            raise ValueError(
                f"Dispatcher received {num_tokens} tokens, exceeding the configured "
                f"peak of {max_num_tokens} tokens per forward"
            )
        # Match Foundry's group-capacity rounding before splitting over sources.
        return (
            int(
                max_num_tokens
                * top_k
                * ep_size
                * self.expert_capacity_factor
                / num_experts
            )
            // ep_size
        )


def dispatcher_peak_tokens(
    *,
    local_batch_size: int,
    seq_len: int,
    pipeline_parallel_degree: int,
    pipeline_parallel_microbatch_size: int,
    tensor_parallel_degree: int,
    context_parallel_degree: int,
    expert_parallel_degree: int,
) -> int:
    """Maximum tokens in one local dispatcher forward under the MoE layout.

    EP enables sequence sharding on TP at the GroupedExperts boundary, even
    when the outer decoder layout is replicated. Without EP, MoE gathers the
    TP sequence; CP still partitions it. PP forwards one pipeline microbatch.
    """
    values = (
        local_batch_size,
        seq_len,
        pipeline_parallel_degree,
        pipeline_parallel_microbatch_size,
        tensor_parallel_degree,
        context_parallel_degree,
        expert_parallel_degree,
    )
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("Token-drop training layout requires positive integer sizes")
    batch = local_batch_size
    if pipeline_parallel_degree > 1:
        batch = pipeline_parallel_microbatch_size
        if local_batch_size % batch:
            raise ValueError(
                "pipeline_parallel_microbatch_size must divide local_batch_size"
            )
    shards = context_parallel_degree
    if expert_parallel_degree > 1:
        shards *= tensor_parallel_degree
    if seq_len % shards:
        raise ValueError(
            f"Token-drop sequence length {seq_len} must divide evenly over "
            f"{shards} dispatcher sequence shards"
        )
    return batch * (seq_len // shards)
