"""GiftMoE init specs; the policies are the ``configs/presets/init/gift_*`` presets."""

from dataclasses import dataclass

from torchtitan.config.configurable import component
from torchtitan.config.schema.protocols.module import InitSpec


__all__ = ["RenormKeepStd"]


@component("renorm_keep_std")
@dataclass(frozen=True, slots=True)
class RenormKeepStd(InitSpec):
    """Run ``init``, then scale every row to unit norm and rescale the whole
    tensor back to the std it had after ``init``."""

    init: InitSpec
