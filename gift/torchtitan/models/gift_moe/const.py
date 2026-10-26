"""Runtime resolvers for the GiftMoE init specs declared in
``torchtitan.config.schema.models.gift_moe.const``."""

import torch

from torchtitan.config.schema.models.gift_moe.const import RenormKeepStd
from torchtitan.distributed.utils import apply_to_full_tensor
from torchtitan.protocols.module import register_param_init, resolve_param_init


def _renorm_rows_keep_std(weight: torch.Tensor) -> None:
    std = weight.std()
    weight.div_(weight.norm(dim=1, keepdim=True))
    weight.mul_(std / weight.std())


def _renorm_keep_std(weight: torch.Tensor) -> None:
    # Row norms need whole rows, which a gfsdp2 shard does not hold.
    apply_to_full_tensor(weight, fn=_renorm_rows_keep_std)


def _resolve_renorm_keep_std(spec: RenormKeepStd):
    init = resolve_param_init(spec.init)

    def initialize(param: torch.Tensor) -> None:
        init(param)
        _renorm_keep_std(param)

    return initialize


register_param_init(RenormKeepStd)(_resolve_renorm_keep_std)
