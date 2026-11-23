#!/usr/bin/env python3
"""Measure an independent row-owner NCCL communicator alongside dense GFSDP."""

import argparse
from datetime import timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re


ARMS = {"shared": None, "dedicated-default": None, "dedicated-8": 8, "dedicated-4": 4}
RANKS = list(range(8))
METRIC = "device_used_minus_pytorch_reserved_bytes"
DENSE_READY = "release_dense_warmup_cache"
PROTOCOL_READY = "release_protocol_cache"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def environment(fixed_names):
    # Retain any runtime knob supplied by the launcher as well as the known
    # names that were absent. The experiment writes no environment variables.
    prefixes = ("NCCL_", "TORCH_NCCL_", "CUDA_", "PYTORCH_")
    keys = set(fixed_names) | {key for key in os.environ if key.startswith(prefixes)}
    return {name: os.environ.get(name) for name in sorted(keys)}


def replay(args):
    # Keep the offline summary and CLI independent of CUDA and the replay's
    # native GIFT imports.
    import row_owner_external_memory as previous

    torch, dist = previous.torch, previous.dist
    rank = previous.check_runtime()
    require(previous.WORLD == len(RANKS) and torch.cuda.device_count() == len(RANKS)
            and int(os.environ["LOCAL_WORLD_SIZE"]) == len(RANKS)
            and rank == int(os.environ["RANK"]), "Expected one node and eight rank-matched CUDA devices")
    device = torch.device("cuda", rank)
    properties = torch.cuda.get_device_properties(device)
    nccl = tuple(torch.cuda.nccl.version())
    require(nccl[:3] == (2, 30, 7) and properties.multi_processor_count == 132,
            "Expected NCCL 2.30.7 and 132-SM H100 GPUs")
    identity = str(properties.uuid)
    require(bool(identity), "The CUDA device UUID is missing")
    report_path = args.output / f"rank{rank}.json"
    require(not report_path.exists(), f"Refusing to replace {report_path}")

    cap = ARMS[args.arm]
    dedicated = args.arm != "shared"
    timeout = timedelta(minutes=5)
    dist.init_process_group("gloo", timeout=timeout)
    try:
        device_uuids = [None] * len(RANKS)
        dist.all_gather_object(device_uuids, identity)
        require(len(set(device_uuids)) == len(RANKS), "Ranks did not use eight distinct CUDA devices")
        peers, fixture = previous.load_ids(args.fixture, rank)
        stages = {}

        def create_group(stage, **kwargs):
            before = previous.snapshot(device)
            group = dist.new_group(RANKS, backend="nccl", timeout=timeout, **kwargs)
            stages[stage] = {"before": before, "after": previous.snapshot(device)}
            require(dist.get_process_group_ranks(group) == RANKS, f"Unexpected rank order for {stage}")
            return group

        dense = create_group("create_dense_group")
        options_max_ctas = None
        try:
            previous.warm_dense_group(stages, rank, device, dense)
            previous.measure(stages, DENSE_READY, device, torch.cuda.empty_cache)
            group = dense
            if dedicated:
                # A Gloo default group and no device_id keep this an ordinary,
                # independently initialized NCCL communicator.
                options = dist.ProcessGroupNCCL.Options()
                config = options.config
                if cap is not None:
                    config.max_ctas = cap
                options_max_ctas = options.config.max_ctas
                require(cap is None or options_max_ctas == cap, "NCCL options did not retain max_ctas")
                group = create_group("create_embedding_group", pg_options=options)
                require(group is not dense, "The embedding route reused the dense process group")
            else:
                previous.measure(stages, "create_embedding_group", device, lambda: None)
            try:
                route = previous.protocol(stages, peers, rank, device, group, enabled=True)
                # Measure residency with both groups still alive, after
                # dropping cached PyTorch allocations used for the replay.
                previous.measure(stages, PROTOCOL_READY, device, torch.cuda.empty_cache)
                require(dist.get_process_group_ranks(dense) == RANKS,
                        "The warmed dense process group was lost")
            finally:
                if dedicated:
                    before = previous.snapshot(device)
                    dist.destroy_process_group(group)
                    del group
                    stages["destroy_embedding_group"] = {"before": before, "after": previous.snapshot(device)}
                else:
                    previous.measure(stages, "destroy_embedding_group", device, lambda: None)
                    del group
        finally:
            before = previous.snapshot(device)
            dist.destroy_process_group(dense)
            del dense
            stages["destroy_dense_group"] = {"before": before, "after": previous.snapshot(device)}
        previous.save(report_path, {
            "schema_version": 1, "arm": args.arm, "rank": rank, "pid": os.getpid(),
            "fixture": fixture, "route": route, "stages": stages,
            "communicators": {
                "dense_ranks": RANKS, "embedding_ranks": RANKS, "dedicated": dedicated,
                "requested_max_ctas": cap, "options_max_ctas": options_max_ctas,
            },
            "runtime": {
                "python": platform.python_version(), "torch": torch.__version__,
                "torch_git": torch.version.git_version, "cuda": torch.version.cuda,
                "nccl": nccl, "gpu": properties.name, "sm_count": properties.multi_processor_count,
                "cuda_device_uuid": identity, "cuda_device_uuids_by_rank": device_uuids,
            },
            "environment": environment(previous.ENVIRONMENT),
            "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        })
    finally:
        dist.destroy_process_group()


def matched_environment(receipt):
    values = dict(receipt["environment"])
    logfile = values.get("NCCL_DEBUG_FILE")
    if logfile is not None:
        values["NCCL_DEBUG_FILE"] = Path(logfile).name  # Same format, arm-specific directory.
    return values


def wire_stages(receipt):
    stages = {}
    for iteration in ("first", "repeat"):
        for operation, dtype in (
            ("metadata_all_gather", "torch.int64"),
            ("int64_requests", "torch.int64"),
            ("bf16_values", "torch.bfloat16"),
            ("bf16_gradients", "torch.bfloat16"),
        ):
            stage = f"{iteration}.{operation}"
            wire = receipt["stages"][stage]["wire"]
            require(wire["executed"] is True and wire["dtype"] == dtype and wire["input_bytes"] > 0
                    and wire["output_bytes"] > 0
                    and re.fullmatch(r"[0-9a-f]{64}", wire["received_raw_bytes_sha256"]),
                    f"Missing exact wire evidence for {stage}")
            stages[stage] = wire
    return stages


def external_memory(receipt):
    stages = receipt["stages"]
    initial = stages["create_dense_group"]["before"][METRIC]
    warm_dense = stages[DENSE_READY]["after"][METRIC]
    route_ready = stages[PROTOCOL_READY]["after"][METRIC]
    online_stages = ["create_embedding_group", *wire_stages(receipt), PROTOCOL_READY]
    by_stage = {
        key: {edge: stages[key][edge][METRIC] - warm_dense for edge in ("before", "after")}
        for key in online_stages
    }
    return {
        "warmed_dense_above_initial_bytes": warm_dense - initial,
        "route_residency_above_warmed_dense_bytes": route_ready - warm_dense,
        "route_and_warmed_dense_above_initial_bytes": route_ready - initial,
        "highest_sampled_route_above_warmed_dense_bytes": max(
            value for edges in by_stage.values() for value in edges.values()
        ),
        "after_route_destroy_above_warmed_dense_bytes": stages["destroy_embedding_group"]["after"][METRIC] - warm_dense,
        "after_dense_destroy_above_initial_bytes": stages["destroy_dense_group"]["after"][METRIC] - initial,
        "online_samples_above_warmed_dense_bytes": by_stage,
    }


def summarize(output):
    arms = {
        arm: [json.loads((output / arm / f"rank{rank}.json").read_text()) for rank in RANKS]
        for arm in ARMS
    }
    control = arms["shared"]
    for arm, reports in arms.items():
        for rank, (reference, row) in enumerate(zip(control, reports, strict=True)):
            require(row["schema_version"] == 1 and row["arm"] == arm and row["rank"] == rank,
                    f"Unexpected receipt identity for {arm} rank {rank}")
            runtime = row["runtime"]
            uuids = runtime["cuda_device_uuids_by_rank"]
            require(len(uuids) == len(set(uuids)) == len(RANKS) and runtime["cuda_device_uuid"] == uuids[rank]
                    and uuids == control[0]["runtime"]["cuda_device_uuids_by_rank"],
                    f"CUDA device identity changed for {arm} rank {rank}")
            comm = row["communicators"]
            require(comm["dense_ranks"] == comm["embedding_ranks"] == RANKS
                    and comm["dedicated"] == (arm != "shared")
                    and comm["requested_max_ctas"] == ARMS[arm]
                    and (ARMS[arm] is None or comm["options_max_ctas"] == ARMS[arm]),
                    f"Unexpected communicator options for {arm} rank {rank}")
            require(all(row[field] == reference[field] for field in ("driver_sha256", "fixture", "route", "runtime"))
                    and matched_environment(row) == matched_environment(reference)
                    and list(row["stages"]) == list(reference["stages"])
                    and wire_stages(row) == wire_stages(reference),
                    f"Fixture, device, runtime, environment or wire differed for {arm} rank {rank}")
    memory = {arm: [external_memory(row) for row in reports] for arm, reports in arms.items()}
    combined = "route_and_warmed_dense_above_initial_bytes"
    delta = {
        arm: [row[combined] - shared[combined] for shared, row in zip(memory["shared"], rows, strict=True)]
        for arm, rows in memory.items()
    }
    result = {
        "schema_version": 1,
        "scope": "Component external CUDA memory sampled at synchronized boundaries, including both live communicators.",
        "metric": METRIC,
        "wire_exact_for_all_arms": True,
        "ranks": RANKS,
        "memory_by_arm_and_rank": memory,
        "route_and_warmed_dense_delta_vs_shared_bytes_by_rank": delta,
        "configured_max_ctas_by_arm": ARMS,
        "effective_channel_evidence": "Use raw NCCL INIT/P2P records for each communicator.",
    }
    with (output / "comparison.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm", choices=ARMS)
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_dir():
        parser.error("--output must be an existing directory")
    if args.summarize:
        summarize(args.output)
    else:
        if args.fixture is None:
            parser.error("--arm requires --fixture")
        replay(args)


if __name__ == "__main__":
    main()
