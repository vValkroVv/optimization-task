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

    def __post_init__(self):
        if self.non_blocking_capacity_factor is not None and not (
            0 < self.non_blocking_capacity_factor <= 1
        ):
            raise ValueError(
                "non_blocking_capacity_factor must be in (0, 1], got "
                f"{self.non_blocking_capacity_factor}"
            )
        for name in ("num_sms_dispatch", "num_sms_combine"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be a positive SM count, got {value}")
