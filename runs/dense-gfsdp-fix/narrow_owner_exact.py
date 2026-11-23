#!/usr/bin/env python3
"""Exact real-ID row-owner replay with the native dedicated NCCL setup."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import re
import sys
from unittest.mock import patch

from campaign import differences, read_json, require, write_json
import compact_owner_numeric as replay


SCRIPT = Path(__file__).resolve()
RANKS = list(range(replay.WORLD))
SITES = ("counts", "request_exchange", "value_exchange", "gradient_exchange")
SCHEMA = 1


def capture_case(numeric, harness, mesh, ids, upstream, seed, max_ctas):
    from torchtitan.distributed.gfsdp.row_owner_embedding import RowOwnerEmbedding

    torch, dist = numeric.torch, numeric.dist
    require(torch.cuda.device_count() == len(RANKS)
            and tuple(torch.cuda.nccl.version())[:3] == (2, 30, 7),
            "Expected eight CUDA devices and NCCL 2.30.7")
    construct = RowOwnerEmbedding.from_embedding
    create_group = dist.new_group
    dense = mesh.get_group()
    groups, calls, counts = [], [], []

    def record_group(ranks, *args, **kwargs):
        require(not groups and list(ranks) == RANKS and kwargs.get("backend") == "nccl"
                and kwargs["pg_options"].config.max_ctas == max_ctas,
                "The native owner did not request one ordered NCCL group with the specified cap")
        group = create_group(ranks, *args, **kwargs)
        groups.append(group)
        require(group is not dense and dist.get_process_group_ranks(group) == RANKS,
                "The native owner group shares the dense handle or changed its rank order")
        return group

    def make_owner(cls, *args, **kwargs):
        owner = construct(*args, **kwargs)
        require(owner.mesh_info.shard_process_group is dense
                and dist.get_process_group_ranks(dense) == RANKS,
                "Component owner requires the unchanged eight-rank dense mesh")
        if max_ctas is not None:
            # This is the production call site, before root wrapping/attachment.
            with patch.object(dist, "new_group", record_group):
                owner.enable_dedicated_nccl_process_group(max_ctas)
            require(len(groups) == 1 and owner.mesh_info.shard_process_group is dense,
                    "The native method did not create one independent transport group")
        wire_group = groups[0] if max_ctas is not None else dense
        stage_context = owner._stage

        @contextmanager
        def trace_stage(stage):
            if stage not in SITES:
                with stage_context(stage):
                    yield
                return
            name = "all_gather_into_tensor" if stage == "counts" else "all_to_all_single"
            collective = getattr(dist, name)

            def checked(output, input, *args, **kwargs):
                require(kwargs.get("group") is wire_group,
                        f"{stage} bypassed the expected owner communicator")
                calls.append(stage)
                result = collective(output, input, *args, **kwargs)
                if stage == "counts":
                    require(input.dtype == output.dtype == torch.int64,
                            "Owner count metadata must remain int64")
                    digest = harness["tensor_sha256"]
                    counts.append({
                        "dtype": str(input.dtype),
                        "sent_bytes": input.nbytes, "received_bytes": output.nbytes,
                        "sent_sha256": digest(input, torch), "received_sha256": digest(output, torch),
                    })
                return result

            # The existing observer composes inside this wrapper to retain
            # exact request, value and gradient bytes/splits.
            with patch.object(dist, name, checked), stage_context(stage):
                yield

        owner._stage = trace_stage
        return owner

    try:
        with patch.object(RowOwnerEmbedding, "from_embedding", classmethod(make_owner)):
            tensors, coalescing = replay_capture(numeric, harness, mesh, ids, upstream, seed)
        require(calls == list(SITES) * len(replay.PASSES) and len(counts) == len(replay.PASSES),
                "Unexpected owner request/value/gradient launch count or order")
        for name, metadata in zip(replay.PASSES, counts, strict=True):
            tensors["passes"][name]["messages"]["counts"] = metadata
        tensors["communicator"] = {
            "dense_ranks": RANKS, "owner_ranks": RANKS, "dedicated": bool(groups),
            "requested_max_ctas": max_ctas, "wire_calls_by_pass": list(SITES),
            "cuda_device_uuid": str(torch.cuda.get_device_properties(dist.get_rank()).uuid),
        }
        return tensors, coalescing
    finally:
        if groups:
            torch.cuda.synchronize()
            dist.destroy_process_group(groups[0])


# Retain the shared replay's implementation when run() supplies a callback.
replay_capture = replay.capture_replay
replay_check_cases = replay.check_cases


def check_rank(row, cap, compact):
    replay_check_cases(row, "b" if compact else "a")
    devices = set()
    for case in row["cases"].values():
        tensors = case["exact_tensors"]
        group = tensors["communicator"]
        require(group["dense_ranks"] == group["owner_ranks"] == RANKS
                and group["requested_max_ctas"] == cap and group["dedicated"] == (cap is not None)
                and group["wire_calls_by_pass"] == list(SITES) and group["cuda_device_uuid"],
                f"Rank {row['rank']} lacks active communicator/launch evidence")
        devices.add(group["cuda_device_uuid"])
        for name in replay.PASSES:
            metadata = tensors["passes"][name]["messages"]["counts"]
            require(metadata["dtype"] == "torch.int64" and metadata["sent_bytes"] == 72
                    and metadata["received_bytes"] == 8 * 72 and all(
                        re.fullmatch(r"[0-9a-f]{64}", metadata[key])
                        for key in ("sent_sha256", "received_sha256")
                    ), f"Rank {row['rank']} has wrong int64 metadata size or digest")
    require(len(devices) == 1, f"Rank {row['rank']} changed devices between fixtures")
    return next(iter(devices))


def run(args):
    require((args.arm == "b") == (args.max_ctas is not None) and (args.arm == "b" or not args.compact_source),
            "The shared control has no cap. The dedicated arm requires --max-ctas 4 or 8.")

    def capture(numeric, harness, mesh, ids, upstream, seed):
        return capture_case(numeric, harness, mesh, ids, upstream, seed, args.max_ctas)

    def write(path, receipt):
        receipt.update(schema_version=SCHEMA, max_ctas=args.max_ctas, compact_source=args.compact_source)
        validate_receipt(receipt, args.arm)
        write_json(path, receipt)

    with patch.multiple(
        replay, SCRIPT=SCRIPT, capture_replay=capture, write_json=write,
        check_cases=lambda row, arm: check_rank(row, args.max_ctas, args.compact_source),
    ):
        return replay.run(args)


def validate_receipt(receipt, arm):
    cap, compact = receipt["max_ctas"], receipt["compact_source"]
    require(receipt["schema_version"] == SCHEMA and receipt["arm"] == arm
            and ((cap is None and not compact) if arm == "a" else cap in (4, 8)),
            "The replay receipt does not identify the expected shared or dedicated arm")
    rows = receipt["ranks"]
    require([row["rank"] for row in rows] == RANKS
            and all(row["owner_source_sha256"] == receipt["owner_source_sha256"] for row in rows),
            "Missing ranks or mismatched owner source hash")
    uuids = [check_rank(row, cap, compact) for row in rows]
    require(len(set(uuids)) == len(RANKS), "The replay requires eight distinct CUDA devices")
    low = [row["cases"][replay.LOW]["fixture"] for row in rows]
    provenance = low[0]["provenance"]
    cohort = tuple(provenance[key] for key in ("microbatch_zero_based", "lowest_u", "lowest_u_rank"))
    require(all(tuple(fixture["provenance"][key] for key in (
        "microbatch_zero_based", "lowest_u", "lowest_u_rank"
    )) == cohort for fixture in low) and 0 <= cohort[2] < len(RANKS)
        and low[cohort[2]]["unique_tokens"] == cohort[1] < 132,
        "The replay does not contain the accepted natural low-U cohort")
    return uuids


def comparable(case):
    exact = {key: value for key, value in case["exact_tensors"].items() if key != "communicator"}
    return {"fixture": case["fixture"], "upstream": case["upstream"], "exact_tensors": exact}


def compare(args):
    control, candidate = read_json(args.control), read_json(args.candidate)
    require(validate_receipt(control, "a") == validate_receipt(candidate, "b"),
            "A/B changed CUDA device order")
    require(control["source_root"] != candidate["source_root"]
            and control["owner_source_sha256"] != candidate["owner_source_sha256"]
            and all(control[key] == candidate[key] for key in ("source_head", "seed", "driver_sha256")),
            "Expected separate pinned sources with the same base revision and replay")
    rows, mismatches = [], []
    for a, b in zip(control["ranks"], candidate["ranks"], strict=True):
        rank = a["rank"]
        if a["runtime"] != b["runtime"]:
            mismatches.append(f"rank{rank}.runtime")
        matches = {}
        for name in (replay.HIGH, replay.LOW):
            changed = list(differences(comparable(a["cases"][name]), comparable(b["cases"][name])))
            mismatches.extend(f"rank{rank}.{name}.{path}" for path in changed)
            matches[name] = not changed
        rows.append({"rank": rank, "case_matches": matches})
    require(not args.output.exists(), f"Refusing to replace {args.output}")
    write_json(args.output, {
        "schema_version": SCHEMA,
        "exact": not mismatches, "mismatches": mismatches, "ranks": rows,
        "max_ctas": candidate["max_ctas"], "compact_source": candidate["compact_source"],
        "control": str(args.control), "candidate": str(args.candidate),
        "scope": "Real high/low-U IDs with a shared seeded BF16 upstream; exact owner wire, output, and local FP32 masters/gradients.",
    })
    return int(bool(mismatches))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run", help="execute one arm in a fresh torchrun")
    launch.add_argument("--arm", choices=("a", "b"), required=True)
    launch.add_argument("--max-ctas", type=int, choices=(4, 8))
    launch.add_argument("--compact-source", action="store_true")
    launch.add_argument("--source-root", type=Path, required=True)
    launch.add_argument("--owner-source-sha256", required=True)
    launch.add_argument("--high-fixture", type=Path, required=True)
    launch.add_argument("--low-census", type=Path, required=True)
    launch.add_argument("--data-root", type=Path, required=True)
    launch.add_argument("--seed", type=int, default=2048)
    launch.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("compare", help="verify and compare the completed receipts offline")
    verify.add_argument("control", type=Path)
    verify.add_argument("candidate", type=Path)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1) from error
