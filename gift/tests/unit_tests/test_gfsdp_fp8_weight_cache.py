from unittest.mock import Mock

import pytest
import torch

from torchtitan.distributed.gfsdp._fp8_expert_weight_cache import (
    ExpertFP8WeightCache,
    ExpertWeightLayout,
)


_TILE = 128 * 128
_PROJECTIONS = (("w13", (3, 128, 128)), ("w2", (1, 128, 128)))


def test_singleton_expert_cache_aliases_local_planes():
    layout = ExpertWeightLayout.for_flat_shards(_PROJECTIONS, 4 * _TILE, 1, 0)
    assert layout is not None
    assert layout.receive_sizes_by_dtype() == {}
    cache = ExpertFP8WeightCache(layout, torch.empty(layout.local_numel), {})
    assert cache.output_data is cache.local_data
    assert cache.output_scales is cache.local_scales
    data, scales = cache.views["w2"]
    assert data.storage_offset() == 3 * _TILE
    assert scales.storage_offset() == 3


def test_rank_major_gather_places_both_projections_and_scale_planes():
    layouts = [
        ExpertWeightLayout.for_flat_shards(_PROJECTIONS, 2 * _TILE, 2, rank)
        for rank in range(2)
    ]
    assert layouts[0] is not None and layouts[1] is not None
    assert [value.projection for value in layouts[0].local_slices] == ["w13"]
    assert [value.projection for value in layouts[1].local_slices] == ["w13", "w2"]

    sizes = layouts[0].receive_sizes_by_dtype()
    receive = {
        dtype: torch.empty(size, dtype=torch.uint8) for dtype, size in sizes.items()
    }
    caches = [
        ExpertFP8WeightCache(layout, torch.empty(layout.local_numel), receive)
        for layout in layouts
    ]
    for rank, cache in enumerate(caches):
        for index, item in enumerate(cache.layout.local_slices):
            data, scales = item.views(cache.local_data, cache.local_scales)
            value = 1 + rank + 4 * index
            data.view(torch.uint8).fill_(value)
            scales.fill_(10 * value)
    torch.cat([cache.local_data for cache in caches], out=caches[0].output_data)
    torch.cat([cache.local_scales for cache in caches], out=caches[0].output_scales)
    w13, w13_scales = caches[0].views["w13"]
    w2, w2_scales = caches[0].views["w2"]
    assert w13.view(torch.uint8)[:, 0, 0].tolist() == [1, 1, 2]
    assert w13_scales.flatten().tolist() == [10, 10, 20]
    assert w2.view(torch.uint8)[0, 0, 0].item() == 6
    assert w2_scales.item() == 60


def test_rejects_partial_block_rows_on_every_rank():
    projections = (("w13", (1, 128, 256)), ("w2", (1, 128, 128)))
    assert ExpertWeightLayout.for_flat_shards(projections, 3 * _TILE // 2, 2, 0) is None
    assert ExpertWeightLayout.for_flat_shards(projections, 3 * _TILE // 2, 2, 1) is None


def test_invalidating_the_cache_rejects_unprepared_and_previous_generations():
    layout = ExpertWeightLayout.for_flat_shards(_PROJECTIONS, 4 * _TILE, 1, 0)
    assert layout is not None
    cache = ExpertFP8WeightCache(layout, torch.empty(layout.local_numel), {})
    cache.prepared_generation = cache.generation
    _, _, used_generation = cache.resolve("w13", None)

    cache.invalidate(Mock())
    with pytest.raises(RuntimeError, match="not prepared"):
        cache.resolve("w13", used_generation)

    cache.prepared_generation = cache.generation
    with pytest.raises(RuntimeError, match="generation changed"):
        cache.resolve("w13", used_generation)
    data, scales, generation = cache.resolve("w13", cache.generation)
    assert data is cache.views["w13"][0] and scales is cache.views["w13"][1]
    assert generation != used_generation
