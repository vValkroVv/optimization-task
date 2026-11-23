#!/usr/bin/env python3
"""Reconstruct the saved embedding reduction and probe the exact NCCL operation."""

import argparse
from datetime import timedelta
from functools import cache
import hashlib
from itertools import combinations, permutations
import json
import os
from pathlib import Path
import re
import struct
import sys


RANKS = tuple(range(8))
WIDTH, VOCAB = 4096, 128256
WORDS_PER_SHARD = VOCAB * WIDTH // len(RANKS)
TORCH_GIT = "8b353065ce260336c5fdfa74d754e8bd9b91b062"
TERMS = (2**24, 1, -(2**24), 1, 0, 0, 0, 0)
F32 = struct.Struct("<f")


def round_f32(value):
    return F32.unpack(F32.pack(value))[0]


def fold(values, order):
    total = 0.0
    for rank in order:
        total = round_f32(total + values[rank])
    return total


def order_for(ring, destination):
    split = ring.index(destination) + 1
    return ring[split:] + ring[:split]


def original_columns(destination):
    return tuple(tuple(TERMS[(rank + destination + col) % 8] for rank in RANKS) for col in RANKS)


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


@cache
def full_shard_digest(values):
    digest = hashlib.sha256(b"".join(F32.pack(value) for value in values))
    zeros = bytes(1 << 20)
    remaining = (WORDS_PER_SHARD - len(values)) * F32.size
    for _ in range(remaining // len(zeros)):
        digest.update(zeros)
    digest.update(zeros[: remaining % len(zeros)])
    return digest.hexdigest()


def offline(args):
    source = args.saved_result.read_bytes()
    saved = json.loads(source)
    rows = saved["ranks"]
    if tuple(row["rank"] for row in rows) != RANKS:
        raise ValueError("expected complete saved eight-rank evidence")
    known = []
    for row in rows:
        rank = row["rank"]
        case = row["cases"]["synthetic-fp32-order-probe"]
        if not case["exact"]["source_gradient_bf16_sha256"]:
            raise RuntimeError(f"rank {rank} has different BF16 source operands")
        mismatch = case["owner"]["passes"]["initial"].get("local_gradient_fp32_mismatch", {})
        examples = {word["flat_index"]: word for word in mismatch.get("examples", ())}
        if len(examples) != mismatch.get("different_words", 0) or any(col not in RANKS for col in examples):
            raise RuntimeError(f"rank {rank} lacks complete affected FP32 words")
        owner_words = tuple(fold(operands, RANKS) for operands in original_columns(rank))
        dense_words = list(owner_words)
        for column, words in examples.items():
            if int.from_bytes(F32.pack(owner_words[column]), "little") != int(words["owner_word"], 16):
                raise RuntimeError(f"rank {rank} source-order owner prediction differs")
            dense_words[column] = F32.unpack(int(words["dense_word"], 16).to_bytes(4, "little"))[0]
        # The probe touches only the first eight words of each local shard.
        for arm, values in (("dense", dense_words), ("owner", owner_words)):
            for name, scale in (("initial", 1), ("accumulated", 2), ("reset", 1)):
                predicted = full_shard_digest(tuple(round_f32(value * scale) for value in values))
                if predicted != case[arm]["passes"][name]["local_gradient_fp32_sha256"]:
                    raise RuntimeError(f"rank {rank} {name} {arm} full shard hash differs")
        known.append({"rank": rank, "dense_words": dense_words, "owner_words": owner_words,
                      "changed_words_per_pass": len(examples), "all_six_full_shard_hashes_match": True})

    rings = []
    for tail in permutations(RANKS[1:]):
        ring = (RANKS[0], *tail)
        if all(
            all(fold(values, order_for(ring, row["rank"])) == word for values, word in
                zip(original_columns(row["rank"]), row["dense_words"], strict=True))
            for row in known
        ):
            rings.append(ring)
    if rings != [RANKS]:
        raise RuntimeError("saved words did not identify the expected unique ring")
    write(args.output, {
        "saved_result_sha256": hashlib.sha256(source).hexdigest(), "dense_matching_rings": rings,
        "directed_rings_tested": 5040, "owner_order": RANKS,
        "changed_words_per_pass": sum(row["changed_words_per_pass"] for row in known),
        "ranks": known,
    })
    return 0


def distinguishing_columns():
    # +1 survives exactly when its rank is last among the three nonzero
    # operands in a left fold. The first two positions commute.
    columns = []
    for members in combinations(RANKS, 3):
        for small in members:
            positive, negative = [rank for rank in members if rank != small]
            terms = [0] * len(RANKS)
            terms[positive], terms[negative], terms[small] = 2**25, -(2**25), 1
            columns.append(tuple(terms))
    return tuple(columns)


def gpu_probe(args):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.profiler import ProfilerActivity, profile

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    gpu = torch.cuda.get_device_properties(device)
    if (int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_WORLD_SIZE"])) != (8, 8):
        raise ValueError("expected eight local ranks")
    if ("H100" not in gpu.name or gpu.multi_processor_count != 132
            or torch.__version__ != "2.15.0.dev20260822+cu130" or torch.version.git_version != TORCH_GIT
            or torch.version.cuda != "13.0" or torch.cuda.nccl.version() != (2, 30, 7)):
        raise ValueError("expected the saved H100, PyTorch, CUDA and NCCL pins")
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        if dist.get_rank() != rank:
            raise ValueError("expected rank-to-GPU identity")
        group = init_device_mesh("cuda", (8,), mesh_dim_names=("fsdp",)).get_group()
        if dist.get_process_group_ranks(group) != list(RANKS):
            raise ValueError("unexpected dense shard membership")
        extras = distinguishing_columns()
        samples = [original_columns(target) + extras for target in RANKS]
        count = len(samples[rank])
        source = torch.zeros((8, WORDS_PER_SHARD), device=device, dtype=torch.float32)
        source[:, :count] = torch.tensor(
            [[column[rank] for column in columns] for columns in samples],
            device=device, dtype=torch.float32,
        )
        separate_output = torch.empty_like(source[rank])
        # Native GFSDP sets gradient divide factor 1.0 on this component.
        # Its ordinary non-forced reduction lowers that to PreMulSum, not Avg.
        op = dist._make_nccl_premul_sum(1.0)
        dist.barrier(group=group, device_ids=[rank])
        results = {}
        for alias, output in (("separate", separate_output), ("own_shard_alias", source[rank])):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                dist.reduce_scatter_tensor(output, source.view(-1), op=op, group=group)
                torch.cuda.synchronize()
            kernels = [
                event.name for event in trace.events()
                if "nccl" in event.name.lower() and "reducescatter" in event.name.lower()
                and str(event.device_type).endswith("CUDA")
            ]
            observed = output[:count].cpu().tolist()
            expected = [fold(terms, order_for(RANKS, rank)) for terms in samples[rank]]
            valid_orders = sum(
                all(fold(terms, (*order, rank)) == value for terms, value in zip(samples[rank], observed, strict=True))
                for order in permutations(donor for donor in RANKS if donor != rank)
            )
            results[alias] = {
                "exact_ring_words": all(F32.pack(a) == F32.pack(b) for a, b in zip(observed, expected, strict=True)),
                "matching_self_last_linear_orders": valid_orders,
                "observed_words": observed, "expected_words": expected, "cuda_kernels": kernels,
                "outside_probed_words_zero": bool((output[count:].view(torch.int32) == 0).all().item()),
            }
        rows = [None] * len(RANKS)
        dist.all_gather_object(rows, {"rank": rank, "gpu_uuid": str(gpu.uuid), "results": results})
        if rank == 0:
            channel_logs = []
            for path in sorted(args.nccl_log_dir.glob("nccl.*.log")):
                for channel, total, order in re.findall(
                    r"Channel (\d+)/(\d+) : ((?:\d+ ){7}\d+)\s*$", path.read_text(), re.M
                ):
                    channel_logs.append({"log": path.name, "channel": int(channel),
                                         "total_channels": int(total), "ring": list(map(int, order.split()))})
            sizes = {entry["total_channels"] for entry in channel_logs}
            channels_exact = (
                len(sizes) == 1 and {entry["channel"] for entry in channel_logs} == set(range(next(iter(sizes))))
                and all(tuple(entry["ring"]) == RANKS for entry in channel_logs)
            )
            exact = channels_exact and all(
                result["exact_ring_words"] and result["outside_probed_words_zero"]
                and result["matching_self_last_linear_orders"] == 2
                and result["cuda_kernels"] and all("RING" in name.upper() for name in result["cuda_kernels"])
                for row in rows for result in row["results"].values()
            )
            write(args.output, {
                "exact_native_ring": exact, "channel_topology_complete": channels_exact,
                "torch_git": TORCH_GIT, "nccl": torch.cuda.nccl.version(), "argv": sys.argv,
                "operator": "NCCL PreMulSum(float32, factor=1.0)",
                "send_bytes_per_rank": source.nbytes, "receive_bytes_per_rank": separate_output.nbytes,
                "offsets": [0, count - 1], "channel_logs": channel_logs, "ranks": rows,
            })
        else:
            exact = False
        receipt = [exact]
        dist.broadcast_object_list(receipt, src=0)
        return int(not receipt[0])
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    saved = commands.add_parser("offline")
    saved.add_argument("--saved-result", type=Path, required=True)
    saved.add_argument("--output", type=Path, required=True)
    native = commands.add_parser("cuda")
    native.add_argument("--nccl-log-dir", type=Path, required=True)
    native.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return offline(args) if args.command == "offline" else gpu_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
