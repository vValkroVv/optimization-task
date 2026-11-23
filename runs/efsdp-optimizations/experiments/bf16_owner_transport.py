#!/usr/bin/env python3
"""Compare EP4/eFSDP2 FP32 reduce-scatter with peer-only BF16 transport."""

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable
from datetime import timedelta
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import torch.distributed as dist
import triton
import triton.language as tl

TORCH_REVISION = "8b353065ce260336c5fdfa74d754e8bd9b91b062"
NCCL_REVISION = (2, 30, 7)
OWNER_ELEMENTS = 301_989_888
WORLD_SIZE = 2
ABBA = ("native", "owner", "owner", "native")
PEER_ALIAS = "owner_alias_receive"
ABC_CBA = ("native", "owner", PEER_ALIAS, PEER_ALIAS, "owner", "native")
METRICS = ("host_submit_us", "device_span_us")
RANK_RECEIPTS = ("rank-0.json", "rank-1.json")
VERIFICATION_RECEIPTS = ("rank-0-verification.json", "rank-1-verification.json")
SUMMARY_RECEIPT = "summary.json"
ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_NVLS_ENABLE",
    "TORCH_NCCL_BLOCKING_WAIT",
    "TORCH_NCCL_NAN_CHECK",
    "TORCH_NCCL_HIGH_PRIORITY",
)
# First two entries include the real producer's +0 and an injected raw -0.
# The FP32-only probes deliberately sit outside the narrowed transport domain.
SPECIAL = {
    "producer_zeros": (0x00000000, 0x00000000),
    "raw_negative_zeros": (0x80000000, 0x80000000),
    "opposite_zeros": (0x00000000, 0x80000000),
    "cancellation": (0x3F800000, 0xBF800000),
    "fp32_sum_of_bf16_values": (0x3F800000, 0x3B800000),
    "bf16_subnormals": (0x00010000, 0x00010000),
    "opposite_subnormals": (0x80010000, 0x00010000),
    "largest_finite": (0x7F7F0000, 0x7F7F0000),
    "positive_inf": (0x7F800000, 0x3F800000),
    "negative_inf": (0xBF800000, 0xFF800000),
    "opposite_inf": (0x7F800000, 0xFF800000),
    "producer_nan_rank0": (0x7FFF0000, 0x3F800000),
    "producer_nan_rank1": (0x3F800000, 0x7FFF0000),
    "raw_nan_payloads": (0x7FC10000, 0xFFC20000),
    "fp32_only_rank0": (0x3F808000, 0x00000000),
    "fp32_only_rank1": (0x00000000, 0x3F808000),
}
FP32_TIE = 0x3F808000  # 1 + 2**-8, which RNE narrowing sends as 1.
RAW_NAN_PROBE = "raw_nan_payloads"
CANARY_ELEMENTS = 64
CANARY_VALUE = -120


@triton.jit
def _fill_bf16_owner_input(
    destination, n, source_rank, owner_rank, block: tl.constexpr
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    # Use different magnitudes by source so the FP32 sums need more than 8 bits.
    numerator = ((index % 127) * 37 + source_rank * 11 + owner_rank * 13) % 127 - 63
    value = numerator.to(tl.float32) * tl.where(source_rank == 0, 0.03125, 1.0)
    tl.store(destination + index, value, index < n)


@triton.jit
def _add_bf16_owner_chunk(local, peer, destination, n, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    local_value = tl.load(local + index, index < n, other=0)
    peer_value = tl.load(peer + index, index < n, other=0).to(tl.float32)
    total = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[local_value, peer_value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(destination + index, total, index < n)


def populate_shard(
    shard: torch.Tensor,
    rank: int,
    owner: int,
    special: dict[str, tuple[int, int]],
) -> None:
    n = shard.numel()
    _fill_bf16_owner_input[(triton.cdiv(n, 1024),)](shard, n, rank, owner, 1024)
    if special:
        values = [pair[rank] for pair in special.values()]
        # Preserve the signed-zero and NaN payload bits.
        bits = torch.tensor(values, dtype=torch.int64, device=shard.device).to(
            torch.int32
        )
        shard.view(torch.int32)[: len(bits)].copy_(bits)


def populate(
    slot: torch.Tensor, rank: int, special: dict[str, tuple[int, int]]
) -> None:
    n = slot.numel() // WORLD_SIZE
    for owner, shard in enumerate(slot.split(n)):
        populate_shard(shard, rank, owner, special)


def native_reduce(
    slot: torch.Tensor,
    output: torch.Tensor,
    op: dist.ReduceOp,
    group: dist.ProcessGroup,
) -> None:
    dist.reduce_scatter_tensor(output, slot, op=op, group=group, async_op=False)


def owner_transport(
    slot: torch.Tensor,
    output: torch.Tensor,
    send: torch.Tensor,
    receive: torch.Tensor | None,
    rank: int,
    group: dist.ProcessGroup,
) -> None:
    own, other = (
        slot.narrow(0, rank * output.numel(), output.numel()),
        slot.narrow(0, (1 - rank) * output.numel(), output.numel()),
    )
    chunk = send.numel()
    splits = [0, chunk] if rank == 0 else [chunk, 0]
    for offset in range(0, output.numel(), chunk):
        peer_input = other.narrow(0, offset, chunk)
        send.copy_(peer_input)
        # Once packed on this stream, only this peer chunk is dead. Receiving
        # into its first half keeps NCCL's send and receive ranges disjoint.
        incoming = (
            receive if receive is not None else peer_input.view(torch.bfloat16)[:chunk]
        )
        dist.all_to_all_single(
            incoming,
            send,
            output_split_sizes=splits,
            input_split_sizes=splits,
            group=group,
            async_op=False,
        )
        _add_bf16_owner_chunk[(triton.cdiv(chunk, 1024),)](
            own.narrow(0, offset, chunk),
            incoming,
            output.narrow(0, offset, chunk),
            chunk,
            1024,
        )


def empty_plane(
    elements: int,
    device: torch.device,
    dtype: torch.dtype,
    canaries: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    if canaries is None:
        return torch.empty(elements, dtype=dtype, device=device)
    storage = torch.empty(elements + 2 * CANARY_ELEMENTS, dtype=dtype, device=device)
    canaries.extend((storage[:CANARY_ELEMENTS], storage[-CANARY_ELEMENTS:]))
    for guard in canaries[-2:]:
        guard.fill_(CANARY_VALUE)
    return storage[CANARY_ELEMENTS:-CANARY_ELEMENTS]


def stage_buffers(
    slot: torch.Tensor,
    chunks: int,
    *,
    alias_receive: bool = False,
    canaries: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    owner_elements = slot.numel() // WORLD_SIZE
    chunk, remainder = divmod(owner_elements, chunks)
    if remainder or chunk == 0 or (alias_receive and chunk % 2):
        raise ValueError(
            "Expected evenly sized chunks and complete FP32 lanes for receive aliasing"
        )
    send = empty_plane(chunk, slot.device, torch.bfloat16, canaries)
    receive = (
        None
        if alias_receive
        else empty_plane(chunk, slot.device, torch.bfloat16, canaries)
    )
    return send, receive


def bits_string(bits: int) -> str:
    return f"0x{bits & 0xFFFFFFFF:08x}"


def is_nan_bits(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) > 0x7F800000


def special_results(
    original: torch.Tensor,
    candidate: torch.Tensor,
    probes: dict[str, tuple[int, int]],
    rank: int,
) -> list[dict]:
    n = len(probes)
    native_bits = original.view(torch.int32)[:n].tolist()
    owner_bits = candidate.view(torch.int32)[:n].tolist()
    rows = []
    for (name, inputs), native, owner in zip(
        probes.items(), native_bits, owner_bits, strict=True
    ):
        if FP32_TIE in inputs:
            # One owner keeps the FP32-only contribution locally. Its peer
            # receives the same value narrowed to BF16 and loses the tie bit.
            expected_owner = FP32_TIE if inputs[rank] == FP32_TIE else 0x3F800000
            kind = "outside_bf16_domain"
            passed = native == FP32_TIE and owner == expected_owner
        elif name == RAW_NAN_PROBE:
            kind = "raw_nan_diagnostic"
            passed = is_nan_bits(native) and is_nan_bits(owner)
        else:
            kind = "bitwise_required"
            passed = native == owner
        rows.append(
            {
                "name": name,
                "input_bits_by_source_rank": [bits_string(x) for x in inputs],
                "native_bits": bits_string(native),
                "owner_bits": bits_string(owner),
                "bitwise_equal": native == owner,
                "kind": kind,
                "passed": passed,
            }
        )
    return rows


def verify_next_collective(
    reference: torch.Tensor, result: torch.Tensor, group: dist.ProcessGroup
) -> bool:
    # Immediately send raw first and last lane bits through the same group.
    expected, actual = reference.view(torch.int32), result.view(torch.int32)
    lanes = torch.stack((expected[0], expected[-1], actual[0], actual[-1]))
    gathered = torch.empty(
        WORLD_SIZE * lanes.numel(), dtype=lanes.dtype, device=lanes.device
    )
    dist.all_gather_into_tensor(gathered, lanes, group=group, async_op=False)
    paired = gathered.view(WORLD_SIZE, -1)
    return torch.equal(paired[:, :2], paired[:, 2:])


def verify_peer_slot(
    peer_slot: torch.Tensor,
    reference: torch.Tensor,
    send: torch.Tensor,
    rank: int,
    probes: dict[str, tuple[int, int]],
    *,
    empty_peer: bool,
    alias_receive: bool,
) -> dict:
    if empty_peer and rank == 1:
        reference.zero_()
    else:
        populate_shard(reference, rank, 1 - rank, probes)
    same = torch.equal(peer_slot.view(torch.int32), reference.view(torch.int32))
    if not alias_receive:
        return {"peer_slot_unchanged": same, "passed": same}

    chunk = send.numel()
    suffixes_preserved = all(
        torch.equal(actual[chunk // 2 :], expected[chunk // 2 :])
        for actual, expected in zip(
            peer_slot.view(torch.int32).split(chunk),
            reference.view(torch.int32).split(chunk),
            strict=True,
        )
    )

    if empty_peer and rank == 0:
        reference.zero_()
    else:
        populate_shard(reference, 1 - rank, rank, probes)
    receives_match = True
    for actual, expected in zip(
        peer_slot.split(chunk), reference.split(chunk), strict=True
    ):
        # Reuse the drained send buffer to reconstruct the peer's exact BF16 bytes.
        send.copy_(expected)
        if not torch.equal(actual.view(torch.int16)[:chunk], send.view(torch.int16)):
            receives_match = False
    return {
        "peer_slot_unchanged": same,
        "received_prefixes_bitwise_equal": receives_match,
        "unwritten_suffixes_bitwise_equal": suffixes_preserved,
        "overwritten_bytes_per_chunk": chunk * send.element_size(),
        "passed": not same and receives_match and suffixes_preserved,
    }


def verify_reuse_accumulation(
    slot: torch.Tensor,
    output: torch.Tensor,
    reference: torch.Tensor,
    send: torch.Tensor,
    receive: torch.Tensor | None,
    rank: int,
    op: dist.ReduceOp,
    group: dist.ProcessGroup,
) -> bool:
    own = slot.narrow(0, rank * output.numel(), output.numel())
    control = partial(native_reduce, slot, own, op, group)
    treatment = partial(owner_transport, slot, own, send, receive, rank, group)
    for accumulator, reduce in ((reference, control), (output, treatment)):
        accumulator.zero_()
        for empty_second_rank in (False, True):
            if empty_second_rank and rank == 1:
                slot.zero_()
            else:
                populate(slot, rank, {})
            reduce()
            accumulator.add_(own)
    # Each second producer runs directly after the previous accumulation on
    # the caller stream, with no host read or completion wait between them.
    return torch.equal(reference.view(torch.int32), output.view(torch.int32))


def check_canaries(canaries: list[torch.Tensor]) -> bool:
    return all(bool((guard == CANARY_VALUE).all().item()) for guard in canaries)


def verify_chunk_choice(
    slot: torch.Tensor,
    output: torch.Tensor,
    reference: torch.Tensor,
    rank: int,
    chunks: int,
    op: dist.ReduceOp,
    group: dist.ProcessGroup,
    *,
    route: str = "owner",
    canaries: list[torch.Tensor] | None = None,
) -> list[dict]:
    if canaries is not None:
        canaries = canaries.copy()
    alias_receive = route == PEER_ALIAS
    send, receive = stage_buffers(
        slot, chunks, alias_receive=alias_receive, canaries=canaries
    )
    rows = []
    for alias in (False, True):
        result = (
            slot.narrow(0, rank * output.numel(), output.numel()) if alias else output
        )
        for empty_peer in (False, True):
            # Only rank 1 is empty. The empty source still participates in NCCL.
            probes = (
                SPECIAL
                if not empty_peer
                else {
                    name: (inputs[0], 0)
                    for name, inputs in SPECIAL.items()
                    if FP32_TIE not in inputs
                }
            )
            if empty_peer and rank == 1:
                slot.zero_()
            else:
                populate(slot, rank, probes)
            if not alias:
                result.fill_(float("nan"))
            native_reduce(slot, result, op, group)
            reference.copy_(result)

            if empty_peer and rank == 1:
                slot.zero_()
            else:
                populate(slot, rank, probes)
            if not alias:
                result.fill_(float("nan"))
            owner_transport(slot, result, send, receive, rank, group)
            ordering_ok = canaries is None or verify_next_collective(
                reference, result, group
            )

            n = len(probes)
            # torch.equal compares integers so signed zero and NaN bits count.
            body_equal = torch.equal(
                reference.view(torch.int32)[n:], result.view(torch.int32)[n:]
            )
            special = special_results(reference, result, probes, rank)
            passed = body_equal and all(row["passed"] for row in special)
            extra = {}
            if canaries is not None:
                peer_slot = slot.narrow(0, (1 - rank) * output.numel(), output.numel())
                peer_slot_check = verify_peer_slot(
                    peer_slot,
                    reference,
                    send,
                    rank,
                    probes,
                    empty_peer=empty_peer,
                    alias_receive=alias_receive,
                )
                canaries_intact = check_canaries(canaries)
                extra = {
                    "canaries_intact": canaries_intact,
                    "next_collective_bitwise_equal": ordering_ok,
                    "peer_slot_check": peer_slot_check,
                }
                passed = (
                    passed
                    and canaries_intact
                    and ordering_ok
                    and peer_slot_check["passed"]
                )
            rows.append(
                {
                    "chunks": chunks,
                    "route": route,
                    "layout": "alias_owned_slot_slice" if alias else "out_of_place",
                    "scenario": "rank1_empty" if empty_peer else "both_populated",
                    "compared_regular_elements": result.numel() - n,
                    "regular_bitwise_equal": body_equal,
                    "special": special,
                    **extra,
                    "passed": passed,
                }
            )
    if canaries is not None:
        accumulated_equal = verify_reuse_accumulation(
            slot, output, reference, send, receive, rank, op, group
        )
        canaries_intact = check_canaries(canaries)
        rows.append(
            {
                "chunks": chunks,
                "route": route,
                "layout": "alias_owned_slot_slice",
                "scenario": "producer_overwrite_then_accumulate",
                "compared_regular_elements": output.numel(),
                "regular_bitwise_equal": accumulated_equal,
                "canaries_intact": canaries_intact,
                "passed": accumulated_equal and canaries_intact,
            }
        )
    return rows


def measure(
    operation: Callable[[], None],
    start: torch.cuda.Event,
    end: torch.cuda.Event,
    prepare: Callable[[], None] | None = None,
) -> tuple[float, float]:
    if prepare is not None:
        prepare()
    start.record()
    before = perf_counter_ns()
    operation()
    host_submit_us = (perf_counter_ns() - before) / 1_000
    end.record()
    end.synchronize()
    return host_submit_us, start.elapsed_time(end) * 1_000


def memory_snapshot(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "live_allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "device_used_bytes_snapshot": total - free,
        "device_capacity_bytes": total,
    }


def benchmark(
    args: argparse.Namespace,
    slot: torch.Tensor,
    output: torch.Tensor,
    rank: int,
    op: dist.ReduceOp,
    group: dist.ProcessGroup,
) -> list[dict]:
    populate(slot, rank, {})
    order = ABC_CBA if args.alias_peer_receive else ABBA
    # The aliased route destroys the peer inputs. Apply the same untimed
    # producer to every arm before every sample, including warmups.
    prepare = partial(populate, slot, rank, {}) if args.alias_peer_receive else None
    device = slot.device
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    blocks = []
    for chunks in args.chunks:
        for route in order * args.repeats:
            torch.cuda.empty_cache()
            baseline = memory_snapshot(device)
            if route != "native":
                scratch = stage_buffers(slot, chunks, alias_receive=route == PEER_ALIAS)
                operation = partial(
                    owner_transport, slot, output, *scratch, rank, group
                )
            else:
                scratch = ()
                operation = partial(native_reduce, slot, output, op, group)
            dist.barrier(group=group)
            for _ in range(args.warmup):
                measure(operation, start, end, prepare)
            torch.cuda.reset_peak_memory_stats(device)
            before = memory_snapshot(device)
            samples = [
                measure(operation, start, end, prepare) for _ in range(args.samples)
            ]
            after = memory_snapshot(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            blocks.append(
                {
                    "chunks": chunks,
                    "route": route,
                    "staging_tensor_bytes": sum(
                        t.numel() * t.element_size() for t in scratch if t is not None
                    ),
                    "aliased_receive_bytes": (
                        output.numel() * torch.bfloat16.itemsize // chunks
                        if route == PEER_ALIAS
                        else 0
                    ),
                    "baseline_memory": baseline,
                    "memory_before_samples": before,
                    "memory_after_samples": after,
                    "sample_peak_allocated_bytes": peak_allocated,
                    "sample_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                        device
                    ),
                    "incremental_peak_allocated_bytes": peak_allocated
                    - baseline["live_allocated_bytes"],
                    "incremental_live_allocated_bytes": before["live_allocated_bytes"]
                    - baseline["live_allocated_bytes"],
                    "samples": samples,
                }
            )
            del operation, scratch
    return blocks


def write_json(path: Path, data: dict) -> None:
    with path.open("x") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")


def summarize(ranks: list[dict]) -> list[dict]:
    blocks = []
    for left, right in zip(*(rank["blocks"] for rank in ranks), strict=True):
        for key in ("chunks", "route", "staging_tensor_bytes", "aliased_receive_bytes"):
            if left[key] != right[key]:
                raise RuntimeError(f"Ranks measured different {key}")
        blocks.append(
            {
                "chunks": left["chunks"],
                "route": left["route"],
                "staging_tensor_bytes": left["staging_tensor_bytes"],
                "aliased_receive_bytes": left["aliased_receive_bytes"],
                **{
                    metric: median(
                        max(a[index], b[index])
                        for a, b in zip(left["samples"], right["samples"], strict=True)
                    )
                    for index, metric in enumerate(METRICS)
                },
                "worst_rank_incremental_peak_allocated_bytes": max(
                    left["incremental_peak_allocated_bytes"],
                    right["incremental_peak_allocated_bytes"],
                ),
                "worst_rank_incremental_live_allocated_bytes": max(
                    left["incremental_live_allocated_bytes"],
                    right["incremental_live_allocated_bytes"],
                ),
            }
        )
    return blocks


def run_rank(args: argparse.Namespace, rank: int, device: torch.device) -> None:
    group = dist.group.WORLD
    op = torch._C._distributed_c10d._make_nccl_premul_sum(1.0)
    stream = torch.cuda.Stream(device=device, priority=-1)
    with torch.cuda.stream(stream):
        canaries = [] if args.alias_peer_receive else None
        slot = empty_plane(WORLD_SIZE * OWNER_ELEMENTS, device, torch.float32, canaries)
        output = empty_plane(OWNER_ELEMENTS, device, torch.float32, canaries)
        reference = torch.empty_like(output)
        verification = [
            row
            for chunks in args.chunks
            for route in (
                ("owner", PEER_ALIAS) if args.alias_peer_receive else ("owner",)
            )
            for row in verify_chunk_choice(
                slot,
                output,
                reference,
                rank,
                chunks,
                op,
                group,
                route=route,
                canaries=canaries,
            )
        ]
        del reference, canaries
        passed = torch.tensor(
            int(all(row["passed"] for row in verification)),
            device=device,
            dtype=torch.int32,
        )
        dist.all_reduce(passed, op=dist.ReduceOp.MIN, group=group)
        pair_passed = bool(passed.item())
        del passed
        write_json(
            args.output_dir / VERIFICATION_RECEIPTS[rank],
            {
                "rank": rank,
                "pair_passed": pair_passed,
                "owner_elements": OWNER_ELEMENTS,
                "verification": verification,
            },
        )
        blocks = (
            benchmark(args, slot, output, rank, op, group)
            if pair_passed and not args.verify_only
            else []
        )

    properties = torch.cuda.get_device_properties(device)
    receipt = {
        "rank": rank,
        "host": platform.node(),
        "gpu": properties.name,
        "gpu_uuid": str(properties.uuid),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "python": sys.version,
        "torch_version": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version(),
        "triton_version": triton.__version__,
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "process_group_backend": dist.get_backend(group),
        "world_size": WORLD_SIZE,
        "process_group_timeout_seconds": args.pg_timeout,
        "owner_elements": OWNER_ELEMENTS,
        "full_slot_elements": WORLD_SIZE * OWNER_ELEMENTS,
        "native_operation": "FP32 NCCL PREMUL_SUM(1.0)",
        "owner_operation": "BF16 peer-only all_to_all_single plus FP32 owner add",
        "compares_aliased_receive": args.alias_peer_receive,
        "restores_slot_before_every_sample": args.alias_peer_receive,
        "timed_output_layout": "out_of_place",
        "async_op": False,
        "pair_passed": pair_passed,
        "verification": verification,
        "verify_only": args.verify_only,
        "sample_fields": METRICS,
        "warmups_per_block": args.warmup,
        "samples_per_block": args.samples,
        "repeats": args.repeats,
        "blocks": blocks,
    }
    write_json(args.output_dir / RANK_RECEIPTS[rank], receipt)
    dist.barrier(group=group)
    if rank == 0:
        ranks = [
            json.loads((args.output_dir / name).read_text()) for name in RANK_RECEIPTS
        ]
        write_json(
            args.output_dir / SUMMARY_RECEIPT,
            {
                "argv": sys.argv,
                "interpreter": sys.executable,
                "pair_passed": pair_passed,
                "timing_performed": bool(blocks),
                "block_medians_of_samplewise_rank_maxima": summarize(ranks),
            },
        )
    if not pair_passed:
        raise RuntimeError(
            "Bitwise verification or the expected FP32 trap failed; see receipts"
        )


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunks", type=int, choices=(1, 2, 8), nargs="+")
    parser.add_argument(
        "--alias-peer-receive",
        action="store_true",
        help="compare native, disjoint owner, and owner receiving into its dead peer slice",
    )
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--samples", type=positive_int, default=20)
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--pg-timeout", type=positive_int, default=60)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.chunks is None:
        args.chunks = (1, 2) if args.alias_peer_receive else (1, 8)

    if len(set(args.chunks)) != len(args.chunks):
        parser.error("--chunks must list each choice once")
    if torch.version.git_version != TORCH_REVISION:
        raise RuntimeError(
            f"Expected Torch {TORCH_REVISION}, found {torch.version.git_version}"
        )
    if any(
        os.environ.get(key) != str(WORLD_SIZE)
        for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")
    ):
        raise RuntimeError("Use torchrun with two ranks on one node")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Expose exactly two H100s to the two local ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "H100" not in torch.cuda.get_device_name(
        device
    ) or torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError(f"Rank {rank} requires an H100")
    if args.alias_peer_receive and torch.cuda.nccl.version() != NCCL_REVISION:
        raise RuntimeError(
            f"Expected NCCL {NCCL_REVISION} for the aliased receive comparison"
        )
    if os.environ.get("TORCH_NCCL_NAN_CHECK", "").lower() in ("1", "true"):
        raise RuntimeError(
            "The NaN probe requires TORCH_NCCL_NAN_CHECK to be unset or disabled"
        )
    if not args.output_dir.is_dir() or any(
        (args.output_dir / name).exists()
        for name in (*RANK_RECEIPTS, *VERIFICATION_RECEIPTS, SUMMARY_RECEIPT)
    ):
        raise RuntimeError("Use an existing fresh output directory")

    dist.init_process_group(
        "nccl", device_id=device, timeout=timedelta(seconds=args.pg_timeout)
    )
    try:
        run_rank(args, rank, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
