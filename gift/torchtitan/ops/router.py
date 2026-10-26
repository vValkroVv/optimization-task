"""Router selections identified separately from other top-k operations for SAC."""

import torch
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding


@torch.library.custom_op("torchtitan::router_topk_indices", mutates_args=())
def router_topk_indices(
    scores: torch.Tensor, k: int, sorted: bool = False
) -> torch.Tensor:
    """Select expert/group indices; differentiable weights are gathered separately."""
    return torch.topk(scores, k, dim=-1, sorted=sorted).indices


@router_topk_indices.register_fake
def _router_topk_indices_fake(
    scores: torch.Tensor, k: int, sorted: bool = False
) -> torch.Tensor:
    return scores.new_empty((*scores.shape[:-1], k), dtype=torch.int64)


@register_sharding(torch.ops.torchtitan.router_topk_indices.default)
def _router_topk_indices_sharding(scores, k, sorted=False):
    # Selection requires the whole expert/group dimension, but preserves token sharding.
    placements = [Replicate(), *(Shard(dim) for dim in range(scores.ndim - 1))]
    return [([placement], [placement, None, None]) for placement in placements]
