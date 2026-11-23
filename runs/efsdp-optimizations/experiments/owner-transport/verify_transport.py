#!/usr/bin/env python3
"""Check the overlay's real GFSDP reduce_scatter entry point on an eFSDP2 pair."""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

EXPERIMENTS = Path(__file__).resolve().parents[1]


def run_reductions(collectives, owner, populate, device, stream, rank, group):
    """Run a full FQN cycle twice on the same slot and persistent accumulator."""
    elements = owner.OWNER_ELEMENTS
    slot = torch.empty(2 * elements, dtype=torch.float32, device=device)
    results = []
    for route in ("native", "owner"):
        accumulator = torch.empty(elements, dtype=torch.float32, device=device)
        first = torch.empty_like(accumulator)
        params = [
            SimpleNamespace(
                sharded_param=SimpleNamespace(requires_grad=True, grad=None),
                sharded_param_grad=shard,
            )
            for shard in accumulator.chunk(2)
        ]
        transport = (
            owner.ExpertOwnerTransport(group, device, stream, lambda: True)
            if route == "owner"
            else None
        )
        for microbatch in range(2):
            for index, fqn in enumerate(owner.FQNS):
                if microbatch == 1 and rank == 1:
                    slot.zero_()
                else:
                    populate(slot, rank, {})
                event = collectives.reduce_scatter(
                    params,
                    params,
                    accumulator,
                    slot,
                    group,
                    None,
                    stream,
                    torch.float32,
                    torch.bfloat16,
                    torch.float32,
                    device,
                    1.0,
                    False,
                    False,
                    None,
                    owner_transport=transport,
                    owner_fqn=fqn,
                )
                # Mirror production's slot reuse edge before the next producer
                # overwrite. No host synchronization between reductions.
                event.wait()
                if microbatch == index == 0:
                    first.copy_(accumulator)
                if any(
                    p.sharded_param.grad is not p.sharded_param_grad for p in params
                ):
                    raise RuntimeError("GFSDP replaced the persistent gradient views")
            if transport is not None:
                transport.finish_backward()
        results.append((first, accumulator))

    native, candidate = results
    bits_equal = [
        torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        for actual, expected in zip(candidate, native, strict=True)
    ]
    if not all(bits_equal):
        raise RuntimeError(
            f"Rank {rank} first/accumulated FP32 parity failed: {bits_equal}"
        )
    return {
        "rank": rank,
        "first_out_of_place_bitwise_equal": bits_equal[0],
        "accumulated_in_place_bitwise_equal": bits_equal[1],
        "fp32_owner_elements": elements,
        "reductions_per_arm": 2 * len(owner.FQNS),
        "next_producer_waits_on_reduce_event": True,
        "owner_scratch_bytes": elements * torch.bfloat16.itemsize // 2,
    }


def main():
    sys.path.insert(0, str(EXPERIMENTS))
    from bf16_owner_transport import populate
    from torchtitan.distributed.gfsdp import _collectives, _owner_transport

    if any(os.environ.get(key) != "2" for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        raise RuntimeError("Use a two-rank local torchrun")
    rank = int(os.environ["LOCAL_RANK"])
    if rank != int(os.environ["RANK"]) or torch.cuda.device_count() != 2:
        raise RuntimeError("Expose exactly the two verified H100 devices")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=60))
    try:
        group = dist.group.WORLD
        _owner_transport.require_runtime(device, group)
        stream = torch.cuda.Stream(device=device, priority=-1)
        receipt = run_reductions(
            _collectives, _owner_transport, populate, device, stream, rank, group
        )
        receipt.update(
            torch_revision=torch.version.git_version,
            nccl_version=torch.cuda.nccl.version(),
            gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
        )
        print(json.dumps(receipt), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
