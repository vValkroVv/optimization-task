"""Locate external CUDA memory changes in the ordinary row-owner NCCL protocol."""

import argparse
from datetime import datetime, timedelta
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import runpy
import time

import numpy as np
import torch
from torch import distributed as dist

from row_owner_route_numeric import SOURCE_ROOT, WORLD, check_runtime


ROWS, WIDTH = 16032, 4096
ARMS = ("control", "row-owner")
ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_DEBUG_FILE",
    "NCCL_DEBUG_TIMESTAMP_LEVELS",
    "NCCL_DEBUG_TIMESTAMP_FORMAT",
    "NCCL_CONF_FILE",
    "NCCL_CUMEM_ENABLE",
    "NCCL_CUMEM_HOST_ENABLE",
    "NCCL_BUFFSIZE",
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_P2P_DIRECT_DISABLE",
    "NCCL_ALLOC_P2P_NET_LL_BUFFERS",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_NVLS_ENABLE",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_NET",
    "NCCL_RUNTIME_CONNECT",
    "TORCH_NCCL_HIGH_PRIORITY",
    "TORCH_NCCL_BLOCKING_WAIT",
    "TZ",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_ids(fixture, rank):
    reference = json.loads(
        Path(__file__).with_name("embedding-replay-result.json").read_text()
    )
    loader = runpy.run_path(
        str(SOURCE_ROOT / "runs/production-data-abcd/component_fixtures.py")
    )["load_fixture"]
    tensors, receipt = loader(fixture, rank)
    if receipt["input_ids_int64_sha256"] != reference["input_ids_sha256_by_rank"][rank]:
        raise ValueError("input IDs differ from the accepted embedding replay capture")
    unique = torch.unique(tensors["input_ids"], sorted=True)
    if not (0 <= unique[0] <= unique[-1] < ROWS * WORLD):
        raise ValueError("captured input ID outside the production vocabulary")
    peers = [None] * WORLD
    dist.all_gather_object(peers, unique.tolist())  # Default group is CPU Gloo.
    return [torch.tensor(ids, dtype=torch.int64) for ids in peers], {
        "input_ids_int64_sha256": receipt["input_ids_int64_sha256"],
        "tensors_sha256": receipt["tensors_sha256"],
    }


def wire_layout(peers, rank):
    owners = torch.arange(WORLD + 1, dtype=torch.int64) * ROWS
    counts = torch.stack([torch.searchsorted(ids, owners).diff() for ids in peers])
    unique = peers[rank]
    received_ids = [ids[(ids // ROWS) == rank] for ids in peers]
    requested = torch.cat(received_ids)
    requesters = torch.arange(WORLD, dtype=torch.int64).repeat_interleave(
        counts[:, rank]
    )
    metadata = torch.cat((counts, torch.ones((WORLD, 1), dtype=torch.int64)), dim=1)
    return (
        unique,
        requested,
        requesters,
        metadata,
        counts[rank].tolist(),
        counts[:, rank].tolist(),
    )


def snapshot(device):
    torch.cuda.synchronize(device)
    dist.barrier()  # Avoid warming an additional NCCL communicator.
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    used = total - free
    reading = {
        "time_unix_ns": time.time_ns(),
        "device_total_bytes": total,
        "device_free_bytes": free,
        "device_used_bytes": used,
        "pytorch_reserved_bytes": reserved,
        "pytorch_allocated_bytes": torch.cuda.memory_allocated(device),
        "device_used_minus_pytorch_reserved_bytes": used - reserved,
    }
    dist.barrier()
    return reading


def measure(record, name, device, call):
    before = snapshot(device)
    call()
    after = snapshot(device)
    record[name] = {"before": before, "after": after}


def warm_dense_group(stages, rank, device, group):
    count = ROWS * WIDTH
    shard = torch.full((count,), rank + 1, device=device, dtype=torch.bfloat16)
    gathered = shard.new_empty(count * WORLD)
    measure(
        stages,
        "dense_bf16_all_gather",
        device,
        partial(dist.all_gather_into_tensor, gathered, shard, group=group),
    )
    if gathered[::count].cpu().tolist() != list(range(1, WORLD + 1)):
        raise RuntimeError("warm all-gather returned the wrong rank order")
    del shard, gathered
    gradient = torch.full(
        (count * WORLD,), rank + 1, device=device, dtype=torch.float32
    )
    reduced = gradient.new_empty(count)
    measure(
        stages,
        "dense_fp32_reduce_scatter",
        device,
        lambda: dist.reduce_scatter_tensor(reduced, gradient, group=group),
    )
    if reduced[:1].item() != sum(range(1, WORLD + 1)):
        raise RuntimeError("warm reduce-scatter returned the wrong value")


def signatures(ids, publisher, salt):
    """Encode ID, source and call in BF16 raw words, with no NaNs or zeros."""
    repeated = 0x3A00 + ((ids ^ (publisher << 6) ^ salt) & 0x1FF)
    low = 0x3800 + (ids & 0x7FF)
    high_sender = 0x3C00 + (ids >> 11) + (publisher << 6) + salt
    return torch.stack((repeated, low, high_sender), dim=1).to(torch.int16)


def fill_payload(output, signature):
    words = output.view(torch.int16)
    signature = signature.to(output.device)
    words.copy_(signature[:, :1].expand_as(words))
    words[:, :3].copy_(signature)


def verify(received, expected):
    actual = received.cpu()
    if received.dtype == torch.bfloat16:
        words = actual.view(torch.int16).numpy()
        signature = expected.numpy()
        matched = (
            np.array_equal(words[:, :3], signature)
            and np.equal(words[:, 3:], signature[:, :1]).all()
        )
    else:
        matched = torch.equal(actual, expected)
    if not matched:
        raise RuntimeError(
            "NCCL receive differs from the independently reconstructed wire"
        )
    return sha256(actual.contiguous().view(torch.uint8).numpy()).hexdigest()


def scratch_view(scratch, offset, rows):
    return (
        scratch.narrow(0, offset, 2 * WIDTH * rows)
        .view(torch.bfloat16)
        .view(rows, WIDTH)
    )


def protocol(stages, peers, rank, device, group, enabled):
    unique, requested, requesters, metadata, send, receive = wire_layout(peers, rank)
    u, r = len(unique), len(requested)
    value_offset = max(2 * WIDTH * u, 4 * WIDTH * r)
    gradient_offset = 2 * WIDTH * u
    size = max(
        value_offset + 2 * WIDTH * r,
        gradient_offset + 2 * WIDTH * r + 4 * WIDTH * max(receive),
    )
    scratch = torch.empty(size, dtype=torch.uint8, device=device)

    def exchange(name, incoming, outgoing, expected, call, splits=None):
        incoming.zero_()
        measure(stages, name, device, call if enabled else lambda: None)
        received_cpu = incoming.cpu()  # Match device-to-host copy in the control.
        stages[name]["wire"] = {
            "executed": enabled,
            "dtype": str(outgoing.dtype),
            "input_bytes": outgoing.nbytes,
            "output_bytes": incoming.nbytes,
            "send_receive_split_rows": splits,
            "received_raw_bytes_sha256": verify(received_cpu, expected)
            if enabled
            else None,
        }

    for iteration in ("first", "repeat"):
        stage = f"{iteration}."
        local_metadata = metadata[rank].to(device)
        matrix = torch.empty_like(metadata.reshape(-1), device=device)
        exchange(
            stage + "metadata_all_gather",
            matrix,
            local_metadata,
            metadata.reshape(-1),
            lambda: dist.all_gather_into_tensor(matrix, local_metadata, group=group),
        )

        local_unique = unique.to(device)
        received_ids = local_unique.new_empty(r)
        exchange(
            stage + "int64_requests",
            received_ids,
            local_unique,
            requested,
            lambda: dist.all_to_all_single(
                received_ids, local_unique, receive, send, group=group
            ),
            (send, receive),
        )

        salt = 1 if iteration == "first" else 17
        values = scratch_view(scratch, value_offset, r)
        returned = scratch_view(scratch, 0, u)
        fill_payload(values, signatures(requested, rank, salt))
        exchange(
            stage + "bf16_values",
            returned,
            values,
            signatures(unique, unique // ROWS, salt),
            lambda: dist.all_to_all_single(
                returned, values, send, receive, group=group
            ),
            (receive, send),
        )

        gradients = scratch_view(scratch, 0, u)
        received_gradient = scratch_view(scratch, gradient_offset, r)
        fill_payload(gradients, signatures(unique, rank, salt + 32))
        exchange(
            stage + "bf16_gradients",
            received_gradient,
            gradients,
            signatures(requested, requesters, salt + 32),
            lambda: dist.all_to_all_single(
                received_gradient, gradients, receive, send, group=group
            ),
            (send, receive),
        )
    return {
        "send_rows": send,
        "receive_rows": receive,
        "borrowed_scratch_stand_in_bytes": size,
    }


def replay(args, rank):
    device = torch.device("cuda", rank)
    props = torch.cuda.get_device_properties(device)
    nccl = torch.cuda.nccl.version()
    if nccl[:3] != (2, 30, 7) or props.multi_processor_count != 132:
        raise RuntimeError("requires NCCL 2.30.7 on 132-SM H100 GPUs")
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    try:
        peers, fixture = load_ids(args.fixture, rank)
        stages = {}
        before = snapshot(device)
        group = dist.new_group(
            list(range(WORLD)), backend="nccl", timeout=timedelta(minutes=5)
        )
        stages["create_dense_group"] = {"before": before, "after": snapshot(device)}
        try:
            if dist.get_process_group_ranks(group) != list(range(WORLD)):
                raise RuntimeError("unexpected dense-shard rank order")
            warm_dense_group(stages, rank, device, group)
            measure(
                stages, "release_dense_warmup_cache", device, torch.cuda.empty_cache
            )
            route = protocol(
                stages, peers, rank, device, group, args.arm == "row-owner"
            )
            measure(stages, "release_protocol_cache", device, torch.cuda.empty_cache)
            before = snapshot(device)
        finally:
            dist.destroy_process_group(group)
            del group
        stages["destroy_dense_group"] = {"before": before, "after": snapshot(device)}
        save(
            args.output / f"rank{rank}.json",
            {
                "arm": args.arm,
                "rank": rank,
                "pid": os.getpid(),
                "fixture": fixture,
                "route": route,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "torch_git": torch.version.git_version,
                    "cuda": torch.version.cuda,
                    "nccl": nccl,
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                },
                "environment": {name: os.environ.get(name) for name in ENVIRONMENT},
                "stages": stages,
            },
        )
    finally:
        dist.destroy_process_group()


CALL = re.compile(r"\bNCCL CALL\s+(nccl\w+)\s*\(")
TIMESTAMP = re.compile(r"^\[([0-9]{4}-[^\]]+)\]")


def nccl_calls(root, report):
    paths = list(root.glob(f"nccl-*-{report['pid']}.log"))
    if len(paths) != 1 or not paths[0].stat().st_size:
        raise ValueError(f"missing single nonempty NCCL log for rank {report['rank']}")
    totals = {}
    windows = {stage: {} for stage in report["stages"]}
    with paths[0].open(errors="replace") as stream:
        for line in stream:
            call = CALL.search(line)
            if call is None:
                continue
            name = call[1]
            totals[name] = totals.get(name, 0) + 1
            stamp = TIMESTAMP.search(line)
            if stamp is None:
                continue
            timestamp = datetime.fromisoformat(stamp[1])
            if timestamp.tzinfo is None:
                continue
            # NCCL timestamps have millisecond precision.
            ns = round(timestamp.timestamp() * 1000) * 1000000
            for stage, snapshots in report["stages"].items():
                if (
                    snapshots["before"]["time_unix_ns"] // 1000000 * 1000000
                    <= ns
                    <= snapshots["after"]["time_unix_ns"]
                ):
                    window = windows[stage]
                    window[name] = window.get(name, 0) + 1
    return {
        "file": paths[0].name,
        "observed_api_calls": totals,
        "approximate_stage_api_calls": windows,
    }


def summarize(root):
    arms = {
        arm: [
            json.loads((root / arm / f"rank{rank}.json").read_text())
            for rank in range(WORLD)
        ]
        for arm in ARMS
    }
    controls, owners = (arms[arm] for arm in ARMS)
    fields = ("fixture", "route", "rank", "runtime")
    for rank, (control, owner) in enumerate(zip(controls, owners, strict=True)):
        if control["rank"] != rank or (control["arm"], owner["arm"]) != ARMS:
            raise ValueError("mislabeled arm or rank")
        if any(control[field] != owner[field] for field in fields) or list(
            control["stages"]
        ) != list(owner["stages"]):
            raise ValueError("arm inputs or device stages differ")
        if {
            key: value
            for key, value in control["environment"].items()
            if key != "NCCL_DEBUG_FILE"
        } != {
            key: value
            for key, value in owner["environment"].items()
            if key != "NCCL_DEBUG_FILE"
        }:
            raise ValueError("non-logfile runtime environment differs between arms")
        for stage, result in owner["stages"].items():
            if "wire" in result and (
                not result["wire"]["received_raw_bytes_sha256"]
                or control["stages"][stage]["wire"]["executed"]
            ):
                raise ValueError("protocol verification receipt differs between arms")
    metric = "device_used_minus_pytorch_reserved_bytes"
    delta = {}
    for stage in owners[0]["stages"]:
        by_arm = {}
        after = {}
        for arm, reports in arms.items():
            spans = [report["stages"][stage] for report in reports]
            by_arm[arm] = [
                row["after"][metric] - row["before"][metric] for row in spans
            ]
            after[arm] = [row["after"][metric] for row in spans]
        delta[stage] = {
            "stage_delta_bytes_by_arm_and_rank": by_arm,
            "stage_delta_row_owner_minus_control_bytes_by_rank": [
                b - a
                for a, b in zip(by_arm["control"], by_arm["row-owner"], strict=True)
            ],
            "after_row_owner_minus_control_bytes_by_rank": [
                b - a for a, b in zip(after["control"], after["row-owner"], strict=True)
            ],
        }
    save(
        root / "comparison.json",
        {
            "scope": "device used minus this process's PyTorch reserved on otherwise idle GPUs; not training peak memory or throughput",
            "device_used_minus_pytorch_reserved": delta,
            "nccl_logs": {
                arm: [nccl_calls(root / arm, report) for report in reports]
                for arm, reports in arms.items()
            },
            "dispatch_caveat": "Read exact NCCL calls and allocations in the raw logs. Missing CALL text does not identify a compile-time branch.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output)
    else:
        if args.fixture is None or args.arm is None:
            parser.error("--execute requires --fixture and --arm")
        replay(args, check_runtime())


if __name__ == "__main__":
    main()
