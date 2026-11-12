"""Runtime resolvers for the GiftMoE init specs declared in
``torchtitan.config.schema.models.gift_moe.const``."""

from collections.abc import Callable
from math import sqrt

import torch
from torch import nn
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.config.schema.models.gift_moe.const import (
    MLAKVAInit,
    MLAKVBInit,
    RenormKeepStd,
)
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


def _initialize_fused_weight(
    weight: torch.Tensor, initialize: Callable[[torch.Tensor], None]
) -> None:
    from torchtitan.distributed.gfsdp import RaggedShardDTensor

    if isinstance(weight, DTensor) and not isinstance(weight, RaggedShardDTensor):
        # Slicing across shards can redistribute and lose the parameter alias.
        full = weight.full_tensor()
        initialize(full)
        weight.copy_(distribute_tensor(full, weight.device_mesh, weight.placements))
    else:
        apply_to_full_tensor(weight, fn=initialize)


@register_param_init(MLAKVAInit)
def _resolve_mla_kv_a(spec: MLAKVAInit):
    def initialize(full: torch.Tensor) -> None:
        if full.ndim != 2 or not 0 < spec.kv_rank < full.shape[0]:
            raise ValueError(
                "mla_kv_a requires a matrix containing latent and RoPE rows"
            )
        nn.init.xavier_normal_(full[: spec.kv_rank])
        nn.init.xavier_normal_(full[spec.kv_rank :])

    return lambda weight: _initialize_fused_weight(weight, initialize)


@register_param_init(MLAKVBInit)
def _resolve_mla_kv_b(spec: MLAKVBInit):
    def initialize(full: torch.Tensor) -> None:
        rows = spec.num_heads * (spec.key_dim + spec.value_dim)
        if full.ndim != 2 or full.shape[0] != rows:
            raise ValueError(f"mla_kv_b requires a matrix with {rows} packed rows")
        kv_rank = full.shape[1]
        packed = full.view(spec.num_heads, spec.key_dim + spec.value_dim, kv_rank)
        nn.init.normal_(
            packed[:, : spec.key_dim],
            std=sqrt(2 / (kv_rank + spec.num_heads * spec.key_dim)),
        )
        nn.init.normal_(
            packed[:, spec.key_dim :],
            std=sqrt(2 / (kv_rank + spec.num_heads * spec.value_dim)),
        )

    return lambda weight: _initialize_fused_weight(weight, initialize)
