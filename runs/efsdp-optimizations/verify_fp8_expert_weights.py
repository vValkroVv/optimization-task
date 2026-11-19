"""Verify native singleton GFSDP FP8 expert-weight cache parity.

Run from the task checkout with its baseline CUDA runner environment:

    /opt/torchtitan-venv/bin/python3.12 runs/efsdp-optimizations/verify_fp8_expert_weights.py

The command needs no arguments. It prints a JSON receipt on success and exits
nonzero if Torch/CUDA is unavailable or any cache check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import isfinite, prod
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[2]
GIFT_ROOT = TASK_ROOT / "gift"
SHAPES = (("w13", (16, 3072, 4096)), ("w2", (16, 4096, 1536)))
SEED = 2048
RANDOM_SCALE = 16.0
BLOCK = 128
TILE_NUMEL = BLOCK * BLOCK


def _add_task_source_path() -> None:
    gift_root = str(GIFT_ROOT)
    if gift_root not in sys.path:
        sys.path.insert(0, gift_root)


def _run_probe() -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Torch is unavailable; run this command with the task's baseline CUDA environment"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable; this verifier requires a native CUDA device"
        )

    from torchtitan.distributed.gfsdp._fp8_expert_weight_cache import (
        ExpertFP8WeightCache,
        ExpertWeightLayout,
    )
    from torchtitan.ops.float8.triton_kernels import block_quant_fn

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    total_numel = sum(prod(shape) for _, shape in SHAPES)
    stream = torch.cuda.Stream(device=device)
    torch.cuda.reset_peak_memory_stats(device)

    def projection_view(master: torch.Tensor, name: str) -> torch.Tensor:
        offset = 0
        for projection, shape in SHAPES:
            if projection == name:
                return master.narrow(0, offset, prod(shape)).view(shape)
            offset += prod(shape)
        raise KeyError(name)

    def inject_outliers(master: torch.Tensor, marker: float) -> None:
        # Keep exact FP32-to-BF16 ties and a known scale in one tile per weight.
        for name, _ in SHAPES:
            weight = projection_view(master, name)
            tie_tile = weight[0, :BLOCK, :BLOCK]
            tie_tile[0, 0] = marker
            tie_tile[0, 1] = 1.0625  # Exact E4M3 midpoint between 1.0 and 1.125.
            tie_tile[0, 2] = 1.0 + 2.0**-8  # FP32-to-BF16 tie, rounds to even 1.0.
            tie_tile[0, 3] = (
                1.0 + 3.0 * 2.0**-8
            )  # The next tie rounds to even 1.015625.
            tie_tile[0, 4] = -(1.0 + 2.0**-8)
            tie_tile[0, 5] = -(1.0 + 3.0 * 2.0**-8)
            weight[0, :BLOCK, BLOCK : 2 * BLOCK].zero_()

    def compare_legacy(
        generation: int, master: torch.Tensor, cache
    ) -> dict[str, object]:
        checks: dict[str, object] = {}
        for name, shape in SHAPES:
            source = projection_view(master, name)
            materialized_bf16 = source.to(torch.bfloat16)
            legacy_data, legacy_scales = block_quant_fn(materialized_bf16)
            cache_data, cache_scales = cache.views[name]

            if not torch.equal(
                cache_data.view(torch.uint8), legacy_data.view(torch.uint8)
            ):
                raise AssertionError(
                    f"generation {generation} {name}: FP8 data bytes differ from legacy BF16 path"
                )
            if not torch.equal(
                cache_scales.view(torch.int32), legacy_scales.view(torch.int32)
            ):
                raise AssertionError(
                    f"generation {generation} {name}: FP32 scales differ from legacy BF16 path"
                )

            zero_tile = cache_data[0, :BLOCK, BLOCK : 2 * BLOCK].view(torch.uint8)
            if zero_tile.count_nonzero().item() != 0:
                raise AssertionError(
                    f"generation {generation} {name}: zero tile is not all +0"
                )
            zero_tile_scale = float(cache_scales[0, 0, 1].item())
            if not isfinite(zero_tile_scale) or zero_tile_scale <= 0.0:
                raise AssertionError(
                    f"generation {generation} {name}: zero-tile scale is not finite and positive"
                )

            checks[name] = {
                "fp8_bytes_bitwise_equal": True,
                "fp32_scales_bitwise_equal": True,
                "zero_tile_positive_zero": True,
                "zero_tile_scale": zero_tile_scale,
            }
            # Keep the largest legacy BF16 and FP8 temporaries bounded to one
            # projection at a time; the cache retains only its required planes.
            del materialized_bf16, legacy_data, legacy_scales, source
        return checks

    with torch.cuda.stream(stream), torch.no_grad():
        generator = torch.Generator(device=device)
        generator.manual_seed(SEED)
        master = torch.randn(
            (total_numel,), dtype=torch.float32, device=device, generator=generator
        )
        master.mul_(RANDOM_SCALE)
        inject_outliers(master, marker=448.0)

        layout = ExpertWeightLayout.for_flat_shards(
            SHAPES, shard_numel=total_numel, shard_count=1, shard_rank=0
        )
        if layout is None:
            raise AssertionError(
                "production FP8 cache rejected the singleton expert layout"
            )
        cache = ExpertFP8WeightCache(layout, master, {})

        if cache.output_data is not cache.local_data:
            raise AssertionError(
                "singleton output FP8 data does not alias its local plane"
            )
        if cache.output_scales is not cache.local_scales:
            raise AssertionError(
                "singleton output scales do not alias their local plane"
            )

        expected_data_offset = 0
        expected_scale_offset = 0
        aliases = {}
        for name, shape in SHAPES:
            data, scales = cache.views[name]
            if data.storage_offset() != expected_data_offset:
                raise AssertionError(f"{name}: unexpected FP8 plane storage offset")
            if scales.storage_offset() != expected_scale_offset:
                raise AssertionError(f"{name}: unexpected scale plane storage offset")
            if (
                data.untyped_storage().data_ptr()
                != cache.local_data.untyped_storage().data_ptr()
            ):
                raise AssertionError(
                    f"{name}: FP8 view does not share the singleton data storage"
                )
            if (
                scales.untyped_storage().data_ptr()
                != cache.local_scales.untyped_storage().data_ptr()
            ):
                raise AssertionError(
                    f"{name}: scale view does not share the singleton scale storage"
                )
            aliases[name] = {
                "data_offset_bytes": expected_data_offset,
                "scale_offset_elements": expected_scale_offset,
            }
            numel = prod(shape)
            expected_data_offset += numel
            expected_scale_offset += numel // TILE_NUMEL

        if cache.prepared_generation != -1 or cache.ready_event is not None:
            raise AssertionError("new cache is unexpectedly marked ready")

        # A singleton returns before distributed collectives, so no process
        # group needs to be initialized for this production cache path.
        first_result = cache.gather(
            master=master, group=None, stream=stream, async_op=False
        )
        if first_result.work:
            raise AssertionError(
                "singleton cache unexpectedly launched distributed gather work"
            )
        if cache.prepared_generation != cache.generation or cache.ready_event is None:
            raise AssertionError("generation 0 did not become ready after gather")
        first_event = cache.ready_event
        if first_result.all_gather_event is not first_event:
            raise AssertionError(
                "singleton result did not expose the cache readiness event"
            )
        first_event.synchronize()

        for name, _ in SHAPES:
            resolved_data, resolved_scales, resolved_generation = cache.resolve(
                name, expected_generation=0
            )
            if (
                resolved_data is not cache.views[name][0]
                or resolved_scales is not cache.views[name][1]
                or resolved_generation != 0
            ):
                raise AssertionError(
                    f"{name}: ready cache did not resolve generation 0 views"
                )

        first_checks = compare_legacy(0, master, cache)
        first_tile_scales = {
            name: float(cache.views[name][1][0, 0, 0].item()) for name, _ in SHAPES
        }

        reused_result = cache.gather(
            master=master, group=None, stream=stream, async_op=False
        )
        if cache.ready_event is not first_event:
            raise AssertionError(
                "same-generation gather replaced the ready event instead of reusing cache"
            )
        if reused_result.all_gather_event is not first_event or reused_result.work:
            raise AssertionError(
                "same-generation gather did not reuse the singleton cache result"
            )
        reused_result.all_gather_event.synchronize()

        cache.invalidate(stream)
        if cache.generation != 1:
            raise AssertionError("cache invalidation did not advance to generation 1")
        try:
            cache.resolve("w13", expected_generation=0)
        except RuntimeError as exc:
            if "not prepared" not in str(exc):
                raise
        else:
            raise AssertionError(
                "invalidated cache resolved before generation 1 was prepared"
            )

        # Invalidation fences the writer stream before this generation mutates
        # the FP32 master, matching the cache owner's production ordering.
        master.mul_(1.0625)
        inject_outliers(master, marker=224.0)
        second_result = cache.gather(
            master=master, group=None, stream=stream, async_op=False
        )
        if second_result.work:
            raise AssertionError(
                "singleton generation 1 unexpectedly launched distributed gather work"
            )
        if cache.prepared_generation != cache.generation or cache.generation != 1:
            raise AssertionError(
                "generation 1 did not become ready after invalidated refresh"
            )
        second_event = cache.ready_event
        if second_event is None or second_event is first_event:
            raise AssertionError("generation 1 did not publish a new readiness event")
        if second_result.all_gather_event is not second_event:
            raise AssertionError(
                "generation 1 result did not expose its readiness event"
            )
        second_event.synchronize()

        try:
            cache.resolve("w13", expected_generation=0)
        except RuntimeError as exc:
            if "generation changed" not in str(exc):
                raise
        else:
            raise AssertionError("generation 1 accepted a stale generation 0 consumer")

        second_tile_scales = {
            name: float(cache.views[name][1][0, 0, 0].item()) for name, _ in SHAPES
        }
        for name, _ in SHAPES:
            if second_tile_scales[name] == first_tile_scales[name]:
                raise AssertionError(
                    f"{name}: invalidated refresh retained the old marker-tile scale"
                )
        second_checks = compare_legacy(1, master, cache)

    torch.cuda.synchronize(device)
    return {
        "status": "PASS",
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_path": str(Path(torch.__file__).resolve()),
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "seed": SEED,
        "random_input": f"torch.randn(fp32) * {RANDOM_SCALE}",
        "shapes": {name: list(shape) for name, shape in SHAPES},
        "cache": {
            "singleton_plane_aliases": aliases,
            "reuse_generation": 0,
            "invalidated_generation": 1,
            "generation_0_marker_scales": first_tile_scales,
            "generation_1_marker_scales": second_tile_scales,
            "readiness_and_stale_generation_checks": True,
        },
        "legacy_bf16_parity": {
            "generation_0": first_checks,
            "generation_1": second_checks,
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    _add_task_source_path()
    print(json.dumps(_run_probe(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
