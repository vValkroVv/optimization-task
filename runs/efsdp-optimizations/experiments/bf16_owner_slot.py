#!/usr/bin/env python3
"""Compare two-H100 EP4 dW slots and owner reduction, with optional real producer timing.

Run with the pinned environment and two visible H100s via torchrun. Add a
verified direct dW --build-root and --num-sms to check both DeepGEMM producers.
The transport gates and timings do not require a DeepGEMM build.
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
import hashlib
from itertools import cycle, islice
import json
from math import prod
import os
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from bf16_owner_transport import (
    ENVIRONMENT,
    FP32_TIE,
    METRICS,
    NCCL_REVISION,
    OWNER_ELEMENTS,
    RANK_RECEIPTS,
    SPECIAL,
    SUMMARY_RECEIPT,
    TORCH_REVISION,
    VERIFICATION_RECEIPTS,
    WORLD_SIZE,
    _fill_bf16_owner_input,
    check_canaries,
    empty_plane,
    measure,
    memory_snapshot,
    native_reduce,
    owner_transport,
    positive_int,
    special_results,
    verify_next_collective,
    write_json,
)

NATIVE = "native_fp32_reduce_scatter"
PACKED = "fp32_slot_packed_owner_alias_receive"
DIRECT = "bf16_slot_send_view"
ROUTES = (NATIVE, PACKED, DIRECT)
CHUNKS = 2
WEIGHT_SHAPES = ((32, 3072, 4096), (32, 4096, 1536))
FULL_SLOT_ELEMENTS = WORLD_SIZE * OWNER_ELEMENTS
BF16_SPECIAL = {name: pair for name, pair in SPECIAL.items() if FP32_TIE not in pair}
# The second reduction rounds 2**24 + 1 to 2**24 before adding the stored
# -2**24. The two lanes swap which rank contributes the large BF16 value.
ACCUMULATOR_FIRST = {
    "large_rank0": (0xCB800000, 0),
    "large_rank1": (0, 0xCB800000),
}
ACCUMULATOR_SECOND = {
    "large_rank0": (0x4B800000, 0x3F800000),
    "large_rank1": (0x3F800000, 0x4B800000),
}
FIRST = "first_owner_output"
ACCUMULATE = "subsequent_fp32_accumulation"


@triton.jit
def _write_owner(
    local, peer, destination, n, add_previous: tl.constexpr, block: tl.constexpr
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < n
    local_value = tl.load(local + index, mask, other=0).to(tl.float32)
    peer_value = tl.load(peer + index, mask, other=0).to(tl.float32)
    reduced = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [local_value, peer_value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    if add_previous:
        previous = tl.load(destination + index, mask, other=0)
        reduced = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [previous, reduced],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(destination + index, reduced, mask)


@dataclass(frozen=True)
class Layout:
    route: str
    slot: torch.Tensor
    scratch: torch.Tensor | None

    def memory(self) -> dict[str, int]:
        scratch_bytes = self.scratch.nbytes if self.scratch is not None else 0
        return {
            "input_slot_bytes": self.slot.nbytes,
            "send_scratch_bytes": scratch_bytes if self.route == PACKED else 0,
            "receive_scratch_bytes": scratch_bytes if self.route == DIRECT else 0,
            "additional_fp32_owner_temporary_bytes": 0,
        }


def make_layout(route, owner_elements, device, canaries=None):
    dtype = torch.bfloat16 if route == DIRECT else torch.float32
    slot = empty_plane(WORLD_SIZE * owner_elements, device, dtype, canaries)
    scratch = (
        empty_plane(owner_elements // CHUNKS, device, torch.bfloat16, canaries)
        if route != NATIVE
        else None
    )
    return Layout(route, slot, scratch)


def populate(slot, rank, scenario, probes=None):
    if scenario == "empty":
        slot.zero_()
        return
    for owner, shard in enumerate(slot.chunk(WORLD_SIZE)):
        n = shard.numel()
        _fill_bf16_owner_input[(triton.cdiv(n, 1024),)](shard, n, rank, owner, 1024)
        if scenario == "mixed":
            for empty in shard.chunk(4)[(rank + owner) % 2 :: 2]:
                empty.zero_()
        if probes:
            dtype = torch.int16 if slot.dtype == torch.bfloat16 else torch.int32
            shift = 16 if dtype == torch.int16 else 0
            bits = torch.tensor(
                [pair[rank] >> shift for pair in probes.values()],
                dtype=torch.int64,
                device=slot.device,
            )
            shard.view(dtype)[: len(bits)].copy_(bits.to(dtype))


def direct_exchange(slot, output, receive, rank, group, *, accumulate):
    shards = slot.chunk(WORLD_SIZE)
    own, other = shards[rank], shards[1 - rank]
    chunk = receive.numel()
    splits = [0, chunk] if rank == 0 else [chunk, 0]
    for offset in range(0, own.numel(), chunk):
        # The send view comes directly from the producer slot. NCCL receives
        # into a disjoint buffer so both input chunks remain intact.
        dist.all_to_all_single(
            receive,
            other.narrow(0, offset, chunk),
            output_split_sizes=splits,
            input_split_sizes=splits,
            group=group,
            async_op=False,
        )
        _write_owner[(triton.cdiv(chunk, 1024),)](
            own.narrow(0, offset, chunk),
            receive,
            output.narrow(0, offset, chunk),
            chunk,
            accumulate,
            1024,
        )


def reduce(layout, output, rank, op, group, *, accumulate=False):
    if layout.route == DIRECT:
        direct_exchange(
            layout.slot, output, layout.scratch, rank, group, accumulate=accumulate
        )
        return
    # Match GFSDP's output choice after the first microbatch. The FP32 owner
    # slice serves as temporary output and the existing accumulator uses add_.
    result = layout.slot.chunk(WORLD_SIZE)[rank] if accumulate else output
    if layout.route == NATIVE:
        native_reduce(layout.slot, result, op, group)
    else:
        owner_transport(layout.slot, result, layout.scratch, None, rank, group)
    if accumulate:
        output.add_(result)


def verify_transport(layouts, output, reference, rank, op, group, canaries):
    rows = []
    native = layouts[NATIVE]
    for scenario in ("full", "empty", "mixed"):
        probes = BF16_SPECIAL if scenario == "full" else {}
        populate(native.slot, rank, scenario, probes)
        native_reduce(native.slot, reference, op, group)
        for route in (PACKED, DIRECT):
            treatment = layouts[route]
            populate(treatment.slot, rank, scenario, probes)
            output.fill_(float("nan"))
            reduce(treatment, output, rank, op, group)
            # An immediately following collective consumes the result on the same stream.
            following = verify_next_collective(reference, output, group)
            special = special_results(reference, output, probes, rank)
            prefix = len(probes)
            equal = torch.equal(
                reference.view(torch.int32)[prefix:], output.view(torch.int32)[prefix:]
            )
            guarded = check_canaries(canaries)
            rows.append(
                {
                    "route": route,
                    "scope": FIRST,
                    "scenario": scenario,
                    "compared_body_elements": output.numel() - prefix,
                    "body_bits_equal": equal,
                    "special": special,
                    "following_collective_bits_equal": following,
                    "canaries_intact": guarded,
                    "passed": equal
                    and following
                    and guarded
                    and all(item["passed"] for item in special),
                }
            )

    for route in (PACKED, DIRECT):
        treatment = layouts[route]
        for index, scenario in enumerate(("full", "mixed", "empty", "full")):
            probes = (ACCUMULATOR_FIRST, ACCUMULATOR_SECOND, None, None)[index]
            for layout, accumulator in ((native, reference), (treatment, output)):
                populate(layout.slot, rank, scenario, probes)
                reduce(layout, accumulator, rank, op, group, accumulate=index > 0)
            following = verify_next_collective(reference, output, group)
            equal = torch.equal(reference.view(torch.int32), output.view(torch.int32))
            cancellation_is_positive_zero = (
                index != 1
                or not torch.count_nonzero(
                    reference.view(torch.int32)[: len(ACCUMULATOR_SECOND)]
                ).item()
            )
            guarded = check_canaries(canaries)
            rows.append(
                {
                    "route": route,
                    "scope": ACCUMULATE if index else FIRST,
                    "scenario": f"reuse_{index}_{scenario}",
                    "compared_body_elements": output.numel(),
                    "body_bits_equal": equal,
                    "following_collective_bits_equal": following,
                    "rounding_probe_is_positive_zero": cancellation_is_positive_zero
                    if index == 1
                    else None,
                    "canaries_intact": guarded,
                    "passed": equal
                    and following
                    and guarded
                    and cancellation_is_positive_zero,
                }
            )
    return rows


def timed_block(args, operation, start, end, prepare=None):
    for _ in range(args.warmup):
        measure(operation, start, end, prepare)
    device = torch.cuda.current_device()
    before = memory_snapshot(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = [measure(operation, start, end, prepare) for _ in range(args.samples)]
    return {
        "memory_before": before,
        "memory_after": memory_snapshot(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "samples": samples,
    }


def prepare_transport(slot, output, rank):
    # Restore both layouts that reuse an input slice for output or receive.
    # Apply the same producer and accumulator initialization to every arm.
    populate(slot, rank, "full")
    output.fill_(1.0)


def benchmark_transport(args, output, rank, op, group, start, end):
    device = output.device
    blocks = []
    order = (*ROUTES, *reversed(ROUTES))
    for scope in (FIRST, ACCUMULATE):
        for route in order * args.repeats:
            torch.cuda.empty_cache()
            common = memory_snapshot(device)
            layout = make_layout(route, output.numel(), device)
            dist.barrier(group=group)

            block = timed_block(
                args,
                partial(
                    reduce,
                    layout,
                    output,
                    rank,
                    op,
                    group,
                    accumulate=scope == ACCUMULATE,
                ),
                start,
                end,
                partial(prepare_transport, layout.slot, output, rank),
            )
            blocks.append(
                {
                    "scope": scope,
                    "route": route,
                    **layout.memory(),
                    "persistent_output_bytes": output.nbytes,
                    "baseline_live_plus_common_tensors_bytes": common[
                        "live_allocated_bytes"
                    ],
                    "live_delta_bytes": block["memory_before"]["live_allocated_bytes"]
                    - common["live_allocated_bytes"],
                    "peak_delta_bytes": block["peak_allocated_bytes"]
                    - common["live_allocated_bytes"],
                    **block,
                }
            )
            del layout
    return blocks


def pair_passed(condition, device, group):
    flag = torch.tensor(int(condition), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def run_producers(args, output, rank, op, group, start, end):
    from direct_dw import load_direct_dw, verify_baseline
    from verify_direct_dw import MIXED_K, make_operand, probe_values

    stock = verify_baseline()
    stock.set_num_sms(args.num_sms)
    direct = load_direct_dw(args.build_root, num_sms=args.num_sms)
    device = output.device
    guard = []
    slots = {
        route: make_layout(route, output.numel(), device, guard)
        for route in (NATIVE, DIRECT)
    }
    bf16_views, fp32_views = [
        tuple(
            block.view(shape)
            for block, shape in zip(
                slots[route].slot.split([prod(shape) for shape in WEIGHT_SHAPES]),
                WEIGHT_SHAPES,
                strict=True,
            )
        )
        for route in (DIRECT, NATIVE)
    ]
    reference = torch.empty_like(output)
    generator = torch.Generator(device=device).manual_seed(1944 + rank)
    experts = WEIGHT_SHAPES[0][0]
    cases = (
        ("full", [4096] * experts),
        ("mixed", list(islice(cycle(MIXED_K), experts))),
        ("empty", [0] * experts),
    )
    rows, blocks = [], []
    special = probe_values(direct, device)
    for scenario, counts in cases:
        operands = [
            (
                make_operand(counts, m, device, generator),
                make_operand(counts, n, device, generator),
            )
            for _, m, n in WEIGHT_SHAPES
        ]
        gpu_counts = torch.tensor(counts, device=device, dtype=torch.int32)
        has_tokens = any(counts)

        def stock_write():
            for (a, b), destination in zip(operands, bf16_views, strict=True):
                destination.zero_()
                if has_tokens:
                    stock.k_grouped_fp8_gemm_nt_contiguous(
                        a, b, destination, counts, gpu_counts, destination
                    )

        def direct_write():
            for (a, b), destination in zip(operands, fp32_views, strict=True):
                direct.k_grouped_fp8_gemm_nt_contiguous_rounded_f32(
                    a, b, destination, counts, gpu_counts
                )

        for layout in slots.values():
            layout.slot.fill_(float("nan"))
        stock_write()
        direct_write()
        write_equal = all(
            torch.equal(a.float().view(torch.int32), b.view(torch.int32))
            for a, b in zip(bf16_views, fp32_views, strict=True)
        )
        empty_are_positive_zero = all(
            not torch.count_nonzero(view[index].view(bits_dtype)).item()
            for views, bits_dtype in (
                (bf16_views, torch.int16),
                (fp32_views, torch.int32),
            )
            for view in views
            for index, count in enumerate(counts)
            if not count
        )
        native_reduce(slots[NATIVE].slot, reference, op, group)
        reduce(slots[DIRECT], output, rank, op, group)
        result_equal = torch.equal(
            reference.view(torch.int32), output.view(torch.int32)
        )
        canaries_intact = check_canaries(guard)
        passed = pair_passed(
            write_equal
            and empty_are_positive_zero
            and result_equal
            and canaries_intact,
            device,
            group,
        )
        rows.append(
            {
                "scenario": scenario,
                "expert_counts": counts,
                "producer_bits_equal": write_equal,
                "empty_experts_are_positive_zero": empty_are_positive_zero,
                "owner_bits_equal_after_real_producer": result_equal,
                "canaries_intact": canaries_intact,
                "pair_passed": passed,
            }
        )
        if passed and not args.verify_only:
            stock_run = ("stock_zero_and_bf16_gemm", stock_write)
            direct_run = ("direct_rounded_fp32_gemm", direct_write)
            for route, operation in (
                stock_run,
                direct_run,
                direct_run,
                stock_run,
            ) * args.repeats:
                dist.barrier(group=group)
                blocks.append(
                    {
                        "scope": "producer_only",
                        "route": route,
                        "scenario": scenario,
                        **timed_block(args, operation, start, end),
                    }
                )
    return {
        "passed": all(row["pair_passed"] for row in rows),
        "checks": rows,
        "blocks": blocks,
        "stock_deepgemm_version": stock.__version__,
        "num_sms": args.num_sms,
        "build_manifest_sha256": digest(args.build_root / "manifest.json"),
        "weight_shapes": WEIGHT_SHAPES,
        "epilogue_special_values": special,
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(left_blocks, right_blocks):
    rows = []
    for left, right in zip(left_blocks, right_blocks, strict=True):
        keys = ("scope", "route", "scenario")
        if any(left.get(key) != right.get(key) for key in keys):
            raise RuntimeError("Ranks used a different timing order")
        row = {key: left[key] for key in keys if key in left}
        paired = list(zip(left["samples"], right["samples"], strict=True))
        for index, metric in enumerate(METRICS):
            row[metric] = median(max(a[index], b[index]) for a, b in paired)
        if "live_delta_bytes" in left:
            row.update(
                {
                    key: max(left[key], right[key])
                    for key in ("live_delta_bytes", "peak_delta_bytes")
                }
            )
        rows.append(row)
    return rows


def run_rank(args, rank, device):
    group = dist.group.WORLD
    op = torch._C._distributed_c10d._make_nccl_premul_sum(1.0)
    stream = torch.cuda.Stream(device=device, priority=-1)
    with torch.cuda.stream(stream):
        canaries = []
        output = empty_plane(OWNER_ELEMENTS, device, torch.float32, canaries)
        reference = empty_plane(OWNER_ELEMENTS, device, torch.float32, canaries)
        layouts = {
            route: make_layout(route, OWNER_ELEMENTS, device, canaries)
            for route in ROUTES
        }
        tensor_bytes = {route: layout.memory() for route, layout in layouts.items()}
        verification = verify_transport(
            layouts, output, reference, rank, op, group, canaries
        )
        passed = pair_passed(all(row["passed"] for row in verification), device, group)
        write_json(
            args.output_dir / VERIFICATION_RECEIPTS[rank],
            {
                "rank": rank,
                "pair_passed": passed,
                "checks": verification,
                "route_tensor_bytes": tensor_bytes,
                "persistent_output_bytes": output.nbytes,
            },
        )
        del layouts, reference, canaries
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        blocks = (
            benchmark_transport(args, output, rank, op, group, start, end)
            if passed and not args.verify_only
            else []
        )
        producers = (
            run_producers(args, output, rank, op, group, start, end)
            if passed and args.build_root
            else None
        )
    passed = passed and (producers is None or producers["passed"])
    receipt = {
        "rank": rank,
        "pair_passed": passed,
        "torch_version": torch.__version__,
        "torch_revision": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version(),
        "triton_version": triton.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "script_sha256": digest(__file__),
        "transport_fixture_sha256": digest(
            Path(__file__).with_name("bf16_owner_transport.py")
        ),
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT},
        "owner_elements": OWNER_ELEMENTS,
        "full_slot_elements": FULL_SLOT_ELEMENTS,
        "route_tensor_bytes": tensor_bytes,
        "persistent_output_bytes": output.nbytes,
        "chunks": CHUNKS,
        "accumulation": "two ordered FP32 RN adds in the BF16-slot arm",
        "native_operation": "FP32 NCCL PREMUL_SUM(1.0)",
        "bf16_slot_transport": "NCCL view sends to disjoint BF16 receive storage, no pack",
        "process_group_timeout_seconds": args.pg_timeout,
        "sample_fields": METRICS,
        "warmups_per_block": args.warmup,
        "samples_per_block": args.samples,
        "repeats": args.repeats,
        "verify_only": args.verify_only,
        "transport_blocks": blocks,
        "producers": producers,
    }
    write_json(args.output_dir / RANK_RECEIPTS[rank], receipt)
    dist.barrier(group=group)
    if rank == 0:
        left, right = (
            json.loads((args.output_dir / name).read_text()) for name in RANK_RECEIPTS
        )
        write_json(
            args.output_dir / SUMMARY_RECEIPT,
            {
                "pair_passed": passed,
                "transport_block_medians_of_samplewise_rank_maxima": summarize(
                    left["transport_blocks"], right["transport_blocks"]
                ),
                "producer_block_medians_of_samplewise_rank_maxima": summarize(
                    (left["producers"] or {}).get("blocks", []),
                    (right["producers"] or {}).get("blocks", []),
                ),
            },
        )
    dist.barrier(group=group)
    if not passed:
        raise RuntimeError("BF16-slot parity gate failed. See rank receipts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Existing fresh evidence directory",
    )
    parser.add_argument(
        "--build-root", type=Path, help="Optional verified rank-local direct dW build"
    )
    parser.add_argument(
        "--num-sms", type=positive_int, help="Resolved sparse GEMM SM budget"
    )
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--samples", type=positive_int, default=10)
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--pg-timeout", type=positive_int, default=90)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if (args.build_root is None) != (args.num_sms is None):
        parser.error("--build-root and --num-sms must be supplied together")
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        parser.error("Use an existing empty output directory")
    if (
        torch.version.git_version != TORCH_REVISION
        or torch.cuda.nccl.version() != NCCL_REVISION
    ):
        raise RuntimeError(f"Requires Torch {TORCH_REVISION} and NCCL {NCCL_REVISION}")
    if any(
        os.environ.get(key) != str(WORLD_SIZE)
        for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")
    ):
        raise RuntimeError("Use torchrun with exactly two local ranks")
    if os.environ.get("TORCH_NCCL_NAN_CHECK", "").lower() in ("1", "true"):
        raise RuntimeError(
            "The special-value check requires NCCL NaN checking to be disabled"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Expose exactly two H100s to this launcher")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "H100" not in torch.cuda.get_device_name(
        device
    ) or torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("The component requires an H100 on each rank")
    if sum(prod(shape) for shape in WEIGHT_SHAPES) != FULL_SLOT_ELEMENTS:
        raise RuntimeError("The producer shapes must match the full EP4 transport slot")
    dist.init_process_group(
        "nccl", device_id=device, timeout=timedelta(seconds=args.pg_timeout)
    )
    try:
        run_rank(args, rank, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
