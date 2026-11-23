"""Check the deferred dW dependency with a real two-rank NCCL slot reader."""

from argparse import ArgumentParser
from datetime import timedelta
import hashlib
import json
from math import prod
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.distributed as dist

from torchtitan.distributed.gfsdp._param_group import (
    GFSDPBufferContext,
    GFSDPParamGroup,
)
from torchtitan.ops.float8 import direct_expert_dw as direct


SHAPES = ((2, 256, 128), (2, 128, 128))
OFFSETS = (0, prod(SHAPES[0]))
SLOT_ELEMENTS = OFFSETS[1] + prod(SHAPES[1])
SLOT_BYTES = SLOT_ELEMENTS * torch.float32.itemsize
SCHEDULING_ENV = (
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_LAUNCH_BLOCKING",
    "TORCH_NCCL_BLOCKING_WAIT",
    "NCCL_BLOCKING_WAIT",
    "NCCL_COMM_BLOCKING",
)


def make_group(context, slot):
    group = object.__new__(GFSDPParamGroup)
    group._module_fqn = "direct_dw_component[efsdp]"
    group.device_handle = torch.cuda
    group.grad_buffer_ctx = context
    group.reduce_grads = True
    group._direct_expert_dw_lease = None
    group._direct_expert_dw_shapes = SHAPES
    group._direct_expert_dw_offsets = OFFSETS
    group.fsdp_params = [
        SimpleNamespace(
            unsharded_param=SimpleNamespace(grad=None),
            unsharded_accumulated_grad=None,
            _unsharded_accumulated_grad=slot.narrow(0, offset, prod(shape)).view(shape),
        )
        for offset, shape in zip(OFFSETS, SHAPES, strict=True)
    ]
    return group


def warm_slot(group, slot, compute, write, zero):
    """Warm both slot writers and release without an older overlapping owner."""
    group._acquire_direct_expert_dw(slot)
    write()
    zero()
    group._finish_direct_expert_dw(group._direct_expert_dw_lease)
    assert group.grad_buffer_ctx.release(group, compute)
    for param in group.fsdp_params:
        param.unsharded_accumulated_grad = None


def check_order(device, reader, writer, operands, counts, zero_first, delay_cycles):
    # Use two different overlapping context objects backed by the same FP32 bytes.
    backing = GFSDPBufferContext(GFSDPBufferContext.BufferType.REDUCE_SCATTER)
    backing.lazy_init({None: SLOT_BYTES}, device)
    old_region = backing.region({None: SLOT_BYTES}, {None: 0})
    new_region = backing.region({None: SLOT_BYTES}, {None: 0})
    slot = new_region.buffers[None].view(torch.float32)
    group = make_group(new_region, slot)
    old_owner = object()
    compute = torch.cuda.current_stream(device)
    rank = dist.get_rank()
    ks, ks_tensor = counts
    write_index = int(zero_first)
    a, b = operands[write_index]
    reference = torch.empty(SHAPES[write_index], dtype=torch.float32, device=device)
    grad_output = torch.empty(0, device=device)
    writer(a, b, reference, ks, ks_tensor)

    # Warm the exact writer, real custom-op dispatch and NCCL on these shapes.
    def write():
        direct.write_direct_expert_dw(
            *a, *b, slot, OFFSETS[write_index], ks, None, ks_tensor
        )

    def zero():
        direct.zero_direct_expert_dw(slot, OFFSETS[1 - write_index], grad_output)

    warm_slot(group, slot, compute, write, zero)
    old_output = torch.empty(SLOT_ELEMENTS // 2, dtype=torch.float32, device=device)
    independent = torch.zeros(4096, device=device)
    independent.add_(1)
    expected_old_bits = torch.full_like(old_output, -3.0).view(torch.int32)
    slot.fill_(-1.0 - rank)
    start, independent_done, delay_done, reader_done, writes_done = [
        torch.cuda.Event(enable_timing=True) for _ in range(5)
    ]
    for event in (start, independent_done, delay_done, reader_done, writes_done):
        event.record(compute)

    # The pinned synchronous API enqueues NCCL on its caller's CUDA stream.
    with torch.cuda.stream(reader):
        reader.wait_event(writes_done)
        torch.cuda._sleep(1)
        dist.reduce_scatter_tensor(old_output, slot, async_op=False)
    torch.cuda.synchronize(device)

    def trial(*, acquire):
        slot.fill_(-1.0 - rank)
        old_output.fill_(float("nan"))
        independent.zero_()
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[device.index])

        old_region.owner = old_owner
        host_start = perf_counter()
        start.record(compute)
        with torch.cuda.stream(reader):
            reader.wait_event(start)
            torch.cuda._sleep(delay_cycles)
            delay_done.record(reader)
            host_sleep = perf_counter()
            dist.reduce_scatter_tensor(old_output, slot, async_op=False)
            host_nccl = perf_counter()
            reader_done.record(reader)
            assert old_region.release(old_owner, reader)
        host_release = perf_counter()
        reader_ready_at_release = reader_done.query()

        host_acquire_start = perf_counter()
        if acquire:
            group._acquire_direct_expert_dw(slot)
            lease = group._direct_expert_dw_lease
            assert new_region.owner is group
            # Finalization can clear the field while the lease still needs its event.
            old_region.release_event = None
        host_acquire = perf_counter()
        independent.add_(1)
        independent_done.record(compute)
        host_marker = perf_counter()
        reader_ready_after_marker = reader_done.query()
        if acquire:
            if zero_first:
                zero()
            write()
            if not zero_first:
                zero()
            writes_done.record(compute)
            group._finish_direct_expert_dw(lease)
            assert new_region.release(group, compute)
        host_writers = perf_counter()
        torch.cuda.synchronize(device)

        independent_ms = start.elapsed_time(independent_done)
        old_read_ms = start.elapsed_time(reader_done)
        return {
            "independent_ms": independent_ms,
            "delay_done_ms": start.elapsed_time(delay_done),
            "old_reader_ms": old_read_ms,
            "new_writers_ms": start.elapsed_time(writes_done) if acquire else None,
            "marker_before_reader": independent_ms < old_read_ms,
            "marker_value_matches": bool(torch.all(independent == 1).item()),
            "old_receiver_bits_match": torch.equal(
                old_output.view(torch.int32), expected_old_bits
            ),
            "reader_ready_at_release": reader_ready_at_release,
            "reader_ready_after_marker": reader_ready_after_marker,
            "host_ms": {
                "through_delay_submit": (host_sleep - host_start) * 1e3,
                "nccl_submit": (host_nccl - host_sleep) * 1e3,
                "release_record": (host_release - host_nccl) * 1e3,
                "lease_acquire": (host_acquire - host_acquire_start) * 1e3,
                "through_marker_submit": (host_marker - host_start) * 1e3,
                "through_writers_submit": (host_writers - host_start) * 1e3,
            },
        }

    control = trial(acquire=False)
    measured = trial(acquire=True)
    actual = [param.unsharded_accumulated_grad for param in group.fsdp_params]
    real_write_bits_match = torch.equal(
        actual[write_index].view(torch.int32), reference.view(torch.int32)
    )
    zero_write_bits_match = (
        torch.count_nonzero(actual[1 - write_index].view(torch.int32)).item() == 0
    )
    new_slot_bits_match = real_write_bits_match and zero_write_bits_match
    writers_after_reader = measured["old_reader_ms"] <= measured["new_writers_ms"]
    return {
        "first_writer": "zero" if zero_first else "deepgemm",
        "shapes": SHAPES,
        "ks": ks,
        "slot_elements": SLOT_ELEMENTS,
        "scheduler_control": control,
        **measured,
        "writers_after_reader": writers_after_reader,
        "real_write_bits_match": real_write_bits_match,
        "zero_write_bits_match": zero_write_bits_match,
        "old_receiver_and_new_slot_bits_match": (
            measured["old_receiver_bits_match"] and new_slot_bits_match
        ),
        "passed": (
            writers_after_reader
            and new_slot_bits_match
            and all(
                result["marker_before_reader"]
                and result["marker_value_matches"]
                and result["old_receiver_bits_match"]
                for result in (control, measured)
            )
        ),
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from direct_dw import load_direct_dw
    from verify_direct_dw import make_operand

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--num-sms", type=int, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--delay-cycles", type=int, default=100_000_000)
    args = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != 2 or args.delay_cycles <= 0:
        parser.error("Use exactly two ranks and a positive delay")

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=2), device_id=device)
    try:
        extension = load_direct_dw(args.build_root, num_sms=args.num_sms)
        writer = extension.k_grouped_fp8_gemm_nt_contiguous_rounded_f32
        direct.configure_direct_expert_dw(writer)
        rank = dist.get_rank()
        compute = torch.cuda.Stream(device=device)
        reader = torch.cuda.Stream(device=device, priority=-1)
        with torch.cuda.stream(compute):
            generator = torch.Generator(device=device).manual_seed(192 + rank)
            ks = [128, 0]
            counts = ks, torch.tensor(ks, dtype=torch.int32, device=device)
            operands = [
                (
                    make_operand(ks, m, device, generator),
                    make_operand(ks, n, device, generator),
                )
                for _, m, n in SHAPES
            ]
            results = [
                check_order(
                    device, reader, writer, operands, counts, zero_first, args.delay_cycles
                )
                for zero_first in (False, True)
            ]
        passed = all(result["passed"] for result in results)
        receipt = {
            "rank": rank,
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(device),
            "num_sms": args.num_sms,
            "delay_cycles": args.delay_cycles,
            "reader_mode": "synchronous NCCL submission on priority -1 stream",
            "scheduling_env": {name: os.environ.get(name) for name in SCHEDULING_ENV},
            "manifest_sha256": digest(args.build_root / "manifest.json"),
            "direct_expert_dw_sha256": digest(direct.__file__),
            "param_group_sha256": digest(
                Path(direct.__file__).parents[2] / "distributed/gfsdp/_param_group.py"
            ),
            "checks": results,
            "passed": passed,
        }
        args.evidence.mkdir(parents=True, exist_ok=True)
        with (args.evidence / f"rank{rank}.json").open("x") as output:
            json.dump(receipt, output, indent=2)
            output.write("\n")
        dist.barrier(device_ids=[device.index])
        if not passed:
            raise RuntimeError(f"Rank {rank} failed the component gate. See rank receipt.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
