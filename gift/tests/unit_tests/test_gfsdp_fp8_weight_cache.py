from datetime import timedelta
from math import prod
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

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


@pytest.mark.parametrize(("experts", "shard_count"), [(16, 1), (32, 2), (4, 4)])
def test_crop_expert_shards_keep_rank_major_block_rows(experts, shard_count):
    projections = (
        ("w13", (experts, 3072, 4096)),
        ("w2", (experts, 4096, 1536)),
    )
    numel = sum(prod(shape) for _, shape in projections)
    shard_numel = numel // shard_count
    layouts = [
        ExpertWeightLayout.for_flat_shards(projections, shard_numel, shard_count, rank)
        for rank in range(shard_count)
    ]
    for layout in layouts:
        assert layout is not None
        assert layout.global_numel == numel
        assert sum(item.numel for item in layout.local_slices) == shard_numel
        assert all(
            item.numel == prod(item.scale_shape) * _TILE
            for item in layout.local_slices
        )
    assert layouts[0].global_slices == layouts[-1].global_slices


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


def _run_multirank_fp8_cache(rank: int, rendezvous: str) -> None:
    from torchtitan.ops.float8.triton_kernels.block_quantization import (
        block_quantization_128x128_fn,
    )

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
        device_id=device,
    )
    try:
        projections = (("w13", (2, 256, 256)), ("w2", (2, 256, 128)))
        count = sum(prod(shape) for _, shape in projections)
        split = count // 2
        generator = torch.Generator(device=device).manual_seed(1701)
        full = torch.randn(count, generator=generator, device=device)
        master = full.narrow(0, rank * split, split)
        layout = ExpertWeightLayout.for_flat_shards(projections, split, 2, rank)
        assert layout is not None
        receive = {
            dtype: torch.empty(size, dtype=torch.uint8, device=device)
            for dtype, size in layout.receive_sizes_by_dtype().items()
        }
        cache = ExpertFP8WeightCache(layout, master, receive)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for generation in range(2):
            expected = {
                item.projection: block_quantization_128x128_fn(
                    full.narrow(0, item.offset, item.numel).view(item.shape).bfloat16()
                )
                for item in layout.global_slices
            }
            cache.gather(master, dist.group.WORLD, stream, async_op=False).wait()
            refresh = cache.ready_event
            for async_op in (False, True):
                cache.gather(master, dist.group.WORLD, stream, async_op=async_op).wait()
                assert cache.ready_event is refresh
                for name, (data, scales) in expected.items():
                    actual_data, actual_scales, actual_generation = cache.resolve(
                        name, generation
                    )
                    assert actual_generation == generation
                    assert torch.equal(actual_data.view(torch.uint8), data.view(torch.uint8))
                    assert torch.equal(actual_scales, scales)
                cache.reader_event = torch.cuda.current_stream().record_event()
                stream.wait_event(cache.reader_event)
            cache.invalidate(torch.cuda.current_stream())
            full.mul_(-1.125)
            cache.writer_event = torch.cuda.current_stream().record_event()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="Requires two CUDA devices and NCCL",
)
def test_multirank_fp8_cache_gathers_exact_data_and_scales(tmp_path):
    torch.multiprocessing.spawn(
        _run_multirank_fp8_cache,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
    )
