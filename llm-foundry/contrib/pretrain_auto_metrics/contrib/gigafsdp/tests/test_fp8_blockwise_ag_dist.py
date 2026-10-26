"""Real multi-rank test of the fp8 blockwise all-gather invariant.

The single-process tests simulate ranks by slicing a buffer.  This one runs the
actual NCCL collectives, on actual separate GPUs, which is the only way to catch:

  * ranks disagreeing on a block scale (different GPU, different SM schedule),
  * a wrong packed-amax layout that a single-process simulation happens to get
    right because every "rank" shares one allocation,
  * collective ordering / shape mismatches.

Run:
    cd contrib/gigafsdp
    torchrun --nproc_per_node=2 tests/test_fp8_blockwise_ag_dist.py
    torchrun --nproc_per_node=4 tests/test_fp8_blockwise_ag_dist.py
    torchrun --nproc_per_node=8 tests/test_fp8_blockwise_ag_dist.py

Exit code is non-zero on failure, so it drops straight into CI.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist

from giga_fsdp.float8_utils import (  # noqa: E402
    FP8_TILE_SIZE,
    compute_local_tile_amaxes,
    fold_and_quantize_param,
)

TILE = FP8_TILE_SIZE
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def ref_quantize_full(w):
    """Single-rank blockwise quantization of the full [E, N, K] weight."""
    E, N, K = w.shape
    blocks = w.float().reshape(E, N // TILE, TILE, K // TILE, TILE)
    amax = blocks.abs().amax(dim=(2, 4)).double()
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax.clamp(min=1e-12) / FP8_MAX))).float()
    q = (blocks / scale.reshape(E, N // TILE, 1, K // TILE, 1)).reshape(E, N, K)
    return q.to(torch.float8_e4m3fn), scale


def run_case(shape, rank, world_size, device, seed):
    E, N, K = shape
    total = E * N * K
    align = world_size * TILE
    padded = ((total + align - 1) // align) * align
    shard_size = padded // world_size
    local_blocks = shard_size // TILE
    n_blocks = E * (N // TILE) * (K // TILE)

    # Identical weights on every rank, then each rank keeps only its slice —
    # mirrors what the optimizer produces after a synchronized init.
    g = torch.Generator(device=device).manual_seed(seed)
    full = torch.zeros(padded, dtype=torch.bfloat16, device=device)
    full[:total] = torch.randn(total, generator=g, device=device).to(torch.bfloat16)
    shard = full[rank * shard_size:(rank + 1) * shard_size].contiguous()

    # --- amax -> all-gather -------------------------------------------------
    local_amaxes = compute_local_tile_amaxes(shard)
    gathered = torch.empty(world_size * local_blocks, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(gathered, local_amaxes)

    # --- fold + quantize (each rank writes only its own elements) -----------
    shard_fp8 = torch.zeros(shard_size, dtype=torch.float8_e4m3fn, device=device)
    block_scale = torch.empty(n_blocks, dtype=torch.float32, device=device)
    fold_and_quantize_param(
        gathered_scales=gathered,
        scale_block_offset=0,
        packed_scale_stride=local_blocks,
        local_scale_blocks=local_blocks,
        shard=shard,
        out_fp8=shard_fp8,
        out_block_scale=block_scale,
        n_experts=E,
        n_row_elem=N,
        n_cols=K,
        elem_offset=0,
        tile_offset=0,
        rank=rank,
        world_size=world_size,
    )

    # --- all-gather the fp8 payload ----------------------------------------
    gathered_fp8 = torch.empty(padded, dtype=torch.uint8, device=device)
    dist.all_gather_into_tensor(gathered_fp8, shard_fp8.view(torch.uint8))
    gathered_fp8 = gathered_fp8.view(torch.float8_e4m3fn)

    failures = []

    # (1) every rank must have produced a bit-identical scale grid
    all_scales = torch.empty(world_size * n_blocks, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(all_scales, block_scale)
    all_scales = all_scales.reshape(world_size, n_blocks)
    for r in range(1, world_size):
        if not torch.equal(all_scales[r].view(torch.int32), all_scales[0].view(torch.int32)):
            n_diff = (all_scales[r] != all_scales[0]).sum().item()
            failures.append(f"rank {r} scale grid differs from rank 0 in {n_diff} blocks")

    # (2) the gathered fp8 must equal single-rank quantization, bitwise
    ref_q, ref_s = ref_quantize_full(full[:total].reshape(shape))
    got_q = gathered_fp8[:total].reshape(shape)
    if not torch.equal(got_q.view(torch.int8), ref_q.view(torch.int8)):
        n_diff = (got_q.view(torch.int8) != ref_q.view(torch.int8)).sum().item()
        failures.append(f"fp8 payload differs from single-rank reference in {n_diff}/{total} elements")
    if not torch.equal(block_scale.reshape(ref_s.shape), ref_s):
        failures.append("block scales differ from single-rank reference")

    # (3) padding tail must be untouched
    if padded > total and (gathered_fp8[total:].view(torch.int8) != 0).any():
        failures.append("padding region was written")

    # (4) no overflow anywhere
    qf = got_q.float()
    if not torch.isfinite(qf).all():
        failures.append("fp8 payload contains Inf/NaN")
    if qf.abs().max().item() > FP8_MAX:
        failures.append(f"|q|={qf.abs().max().item()} exceeds fp8 max")

    return failures


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    cases = [
        (2, 256, 128),
        (4, 256, 256),
        (1, 384, 128),      # total not a multiple of world_size*128 -> real padding
        (3, 128, 256),
    ]

    total_failures = 0
    for i, shape in enumerate(cases):
        failures = run_case(shape, rank, world_size, device, seed=1000 + i)

        # Fail the whole job if ANY rank failed — a per-rank assert would hang
        # the others inside the next collective.
        flag = torch.tensor([len(failures)], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.SUM)
        if failures:
            for f in failures:
                print(f"[rank {rank}] FAIL {shape}: {f}", flush=True)
        elif rank == 0 and flag.item() == 0:
            print(f"[rank 0] ok {shape}", flush=True)
        total_failures += flag.item()

    dist.barrier()
    if rank == 0:
        print(f"\nworld_size={world_size}: "
              f"{'ALL PASSED' if total_failures == 0 else f'{total_failures} FAILURES'}",
              flush=True)
    dist.destroy_process_group()
    return 1 if total_failures else 0


if __name__ == "__main__":
    sys.exit(main())
