from typing import cast, Self
from torchtitan.config.schema.models.common.moe.dispatch.capacity import (
    non_blocking_buffer_capacity,
    retained_receive_capacity,
)
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.moe.dispatch.base import (
    BaseTokenDispatcherConfig,
)


@config(type="hybridep")
class HybridEPTokenDispatcherConfig(BaseTokenDispatcherConfig):
    """HybridEP token dispatcher.

    Args:
        non_blocking_capacity_factor: ``None`` dispatches blocking: a host sync
            computes the exact permuted token count and nothing is dropped. A
            float in (0, 1] dispatches without a CPU sync and sizes the
            fused-permute output as
            ``num_tokens × ep_size × min(num_local_experts, top_k) × cf``
            (aligned for MXFP8); tokens past it are dropped silently, 1.0 never
            drops. The all-to-all buffer still grows to the full token count.
        num_sms_dispatch: SM budget of the dispatch kernel (``num_sms_combine``
            for combine). 16/16 follows NVIDIA's Hybrid-EP README (16 SMs match
            DeepEP at 32); HybridEP's own default is 24 intranode, 8 multinode.
            With ``fuse_permute`` the dispatch grid also carries
            ``min(108, sm_count - num_sms_dispatch)`` permute blocks.
        pad_multiple: Alignment of the permuted layout in ``dispatch`` and
            ``combine``. ``None`` keeps HybridEP's default of 1 (no padding);
            the fp8 grouped GEMMs need 128.
        comm_stream: Run the all-to-all on a side stream so the expert weight
            quantization overlaps it.
        fuse_permute: Fold permute into ``dispatch`` and unpermute into
            ``combine``.
    """

    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.moe.dispatch.hybridep:HybridEPTokenDispatcher"
    )

    non_blocking_capacity_factor: float | None = None
    pad_multiple: int | None = None
    comm_stream: bool = False
    fuse_permute: bool = True
    num_sms_dispatch: int = 16
    num_sms_combine: int = 16

    def __post_init__(self) -> None:
        BaseTokenDispatcherConfig.__post_init__(self)
        for name in ("num_sms_dispatch", "num_sms_combine"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be a positive SM count, got {value}")
        factor = self.non_blocking_capacity_factor
        if factor is not None and (
            isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not 0 < factor <= 1
        ):
            raise ValueError(
                f"non_blocking_capacity_factor must be in (0, 1], got {factor}"
            )
        if self.pad_multiple is not None and (
            type(self.pad_multiple) is not int or self.pad_multiple <= 0
        ):
            raise ValueError("HybridEP pad_multiple must be a positive integer")
        # TODO: lift this restriction after validating the dependency's
        # fused nonblocking expert-count metadata fix.
        if self.token_drop is not None and factor is not None and self.fuse_permute:
            raise ValueError(
                "HybridEP token_drop cannot be used with "
                "non_blocking_capacity_factor set and fuse_permute=True: "
                "the current HybridEP fused nonblocking metadata kernel has "
                "a known bug that can report incorrect expert counts and cause "
                "NaN gradients. Set fuse_permute=False to keep nonblocking "
                "dispatch, or set non_blocking_capacity_factor=None to use "
                "blocking dispatch."
            )

    def validate_capacity(
        self, *, num_tokens: int, top_k: int, num_experts: int, ep_size: int
    ) -> None:
        policy = self.token_drop
        factor = self.non_blocking_capacity_factor
        if policy is None or factor is None:
            return
        capacity = policy.source_capacity(
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            ep_size=ep_size,
            max_num_tokens=self._capacity_num_tokens,
        )
        local_experts = num_experts // ep_size
        allocated = non_blocking_buffer_capacity(
            num_tokens, ep_size, local_experts, top_k, factor, self.pad_multiple
        )
        required = retained_receive_capacity(
            num_tokens=num_tokens,
            source_capacity=capacity,
            ep_size=ep_size,
            num_local_experts=local_experts,
            top_k=top_k,
            drop_by_group_capacity=policy.drop_by_group_capacity,
            pad_multiple=self.pad_multiple,
        )
        if allocated < required:
            raise ValueError(
                "HybridEP token_drop configuration can cause additional, implicit "
                f"drops: non_blocking_capacity_factor={factor} allocates {allocated} "
                f"rows, but retained routes can require {required} padded rows "
                f"(tokens/source={num_tokens}, EP={ep_size}, top_k={top_k}, "
                f"experts/rank={local_experts}, source_capacity={capacity}, "
                f"pad_multiple={self.pad_multiple or 1}, "
                f"drop_by_group_capacity={policy.drop_by_group_capacity}). "
                "Increase non_blocking_capacity_factor, lower the token-drop "
                "quota or capacity factor, reduce padding, or use blocking dispatch "
                "(non_blocking_capacity_factor=None)."
            )

    def with_capacity_context(
        self, *, num_tokens: int, top_k: int, num_experts: int, ep_size: int
    ) -> Self:
        config = cast(
            Self,
            BaseTokenDispatcherConfig.with_capacity_context(
                self,
                num_tokens=num_tokens,
                top_k=top_k,
                num_experts=num_experts,
                ep_size=ep_size,
            ),
        )
        config.validate_capacity(
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            ep_size=ep_size,
        )
        return config
