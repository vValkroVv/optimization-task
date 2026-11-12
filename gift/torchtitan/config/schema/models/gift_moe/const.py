"""GiftMoE init specs; the policies are the ``configs/presets/init/gift_*`` presets."""

from dataclasses import dataclass

from torchtitan.config.configurable import component
from torchtitan.config.schema.protocols.module import InitSpec


__all__ = ["MLAKVAInit", "MLAKVBInit", "RenormKeepStd"]


@component("renorm_keep_std")
@dataclass(frozen=True, slots=True)
class RenormKeepStd(InitSpec):
    """Run ``init``, then scale every row to unit norm and rescale the whole
    tensor back to the std it had after ``init``."""

    init: InitSpec


@component("mla_kv_a")
@dataclass(frozen=True, slots=True)
class MLAKVAInit(InitSpec):
    """Xavier-normal initialization of the separate latent and RoPE matrices."""

    kv_rank: int

    def __post_init__(self) -> None:
        if type(self.kv_rank) is not int or self.kv_rank <= 0:
            raise ValueError("mla_kv_a kv_rank must be a positive integer")


@component("mla_kv_b")
@dataclass(frozen=True, slots=True)
class MLAKVBInit(InitSpec):
    """Xavier-normal initialization of logical K and V, packed per head."""

    num_heads: int
    key_dim: int
    value_dim: int

    def __post_init__(self) -> None:
        for name in ("num_heads", "key_dim", "value_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"mla_kv_b {name} must be a positive integer")
