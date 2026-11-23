"""Check NCCL rank-slice aliasing on two real H100s and the replay weight."""

import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from replay import FIXTURE


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        world = dist.get_world_size()
        assert world == 2 and rank == dist.get_rank()
        weight = torch.load(FIXTURE, map_location="cpu", mmap=True, weights_only=True)["ce"]["weight"]
        master = weight.flatten().chunk(world)[rank].to(device="cuda", dtype=torch.float32)
        ordinary_shard = torch.empty_like(master, dtype=torch.bfloat16)
        ordinary_output = torch.empty(weight.numel(), device="cuda", dtype=torch.bfloat16)
        alias_output = torch.empty_like(ordinary_output)
        alias_shard = alias_output.chunk(world)[rank]
        shard_bytes = alias_shard.numel() * alias_shard.element_size()
        assert alias_shard.data_ptr() == alias_output.data_ptr() + rank * shard_bytes
        results = []
        for async_op in (False, True):
            for sign in (1, -1):
                inputs = master * sign
                ordinary_shard.copy_(inputs)
                alias_shard.copy_(inputs)
                work = dist.all_gather_into_tensor(ordinary_output, ordinary_shard, async_op=async_op)
                if work is not None:
                    work.wait()
                work = dist.all_gather_into_tensor(alias_output, alias_shard, async_op=async_op)
                if work is not None:
                    work.wait()
                assert torch.equal(ordinary_output, alias_output), (rank, async_op, sign)
                results.append({"async_op": async_op, "sign": sign, "bitwise": True})
        path = Path(sys.argv[1]) / f"rank-{rank}.json"
        path.write_text(json.dumps({
            "world_size": world,
            "gpu_rank": rank,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "allocator": torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"],
            "shard_bytes": shard_bytes,
            "cases": results,
        }, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
