#!/usr/bin/env python3
"""Compare exact eight-rank row-owner replays from two fresh native checkouts."""

import argparse
from contextlib import contextmanager
from datetime import timedelta
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from unittest.mock import patch

from campaign import digest, differences, read_json, require, write_json


SCRIPT = Path(__file__).resolve()
OWNER = Path("gift/torchtitan/distributed/gfsdp/row_owner_embedding.py")
REFERENCE = SCRIPT.with_name("embedding-replay-result.json")
WORLD = 8
VOCAB, WIDTH, TOKENS = 128256, 4096, 8192
PASSES = ("initial", "accumulated", "reset")
HIGH, LOW = "real-high-u", "real-low-u"
SCHEMA = 1
SOURCE_STAGES = {"source_coalescing", "compact_source_coalescing"}


def capture_replay(numeric, harness, mesh, ids, upstream, seed):
    torch, dist = numeric.torch, numeric.dist
    gradient_messages = numeric.GradientMessages
    observers = []

    class Messages(gradient_messages):
        def __init__(self):
            super().__init__()
            self.routes = {}
            self.source_calls = []
            self.source_by_pass = []
            observers.append(self)

        @contextmanager
        def __call__(self, stage):
            if stage in ("request_exchange", "value_exchange"):
                with self.capture_route(stage):
                    yield
            elif stage in SOURCE_STAGES:
                op = torch.ops.aten.embedding_dense_backward
                original = op.default

                def observed(grad, indices, num_weights, *args, **kwargs):
                    self.source_calls.append({"stage": stage, "aten_output_rows": num_weights})
                    return original(grad, indices, num_weights, *args, **kwargs)

                with patch.object(op, "default", observed):
                    yield
            else:
                with super().__call__(stage):
                    yield

        @contextmanager
        def capture_route(self, stage):
            exchange = dist.all_to_all_single
            expected = torch.int64 if stage == "request_exchange" else torch.bfloat16

            def observed(output, input, output_split_sizes, input_split_sizes, **kwargs):
                require(input.dtype == output.dtype == expected and stage not in self.routes,
                        f"Unexpected {stage} message count or dtype")
                send = input.clone()
                result = exchange(output, input, output_split_sizes, input_split_sizes, **kwargs)
                self.routes[stage] = send, output.clone(), list(input_split_sizes), list(output_split_sizes)
                return result

            with patch.object(dist, "all_to_all_single", observed):
                yield

        def collect(self, tensor_digest):
            result = super().collect(tensor_digest)
            require(self.routes.keys() == {"request_exchange", "value_exchange"} and len(self.source_calls) == 1,
                    "Expected two forward exchanges and one ATen source coalescing call")
            self.source_by_pass.append(self.source_calls.pop())
            for stage, (send, receive, tx, rx) in self.routes.items():
                result[stage] = {
                    "dtype": str(send.dtype), "sent_sha256": tensor_digest(send, torch),
                    "received_sha256": tensor_digest(receive, torch), "send_split_rows": tx, "receive_split_rows": rx,
                }
            self.routes.clear()
            return result

    with patch.object(numeric, "GradientMessages", Messages):
        result = numeric.run_arm(harness, mesh, ids, upstream, seed, fixed=False)
    (observer,) = observers
    return result, dict(zip(PASSES, observer.source_by_pass, strict=True))


def check_cases(row, arm):
    rank = row["rank"]
    require(row["cases"].keys() == {HIGH, LOW}, f"Rank {rank} lacks one of the two real cases")
    for name, case in row["cases"].items():
        fixture = case["fixture"]
        n, unique = fixture["tokens"], fixture["unique_tokens"]
        sms = row["runtime"]["sm_count"]
        require(n == TOKENS and sms == 132 and 0 < unique <= n, f"Rank {rank} {name}: unexpected pinned H100 shape")
        exact = case["exact_tensors"]
        require(case["coalescing"].keys() == exact["passes"].keys() == set(PASSES),
                f"Rank {rank} {name}: missing forward/backward pass")
        compact_rows = max(unique, ((4 * sms - 1) * min(WIDTH, 1024)) // WIDTH + 1)
        expected = {"stage": "compact_source_coalescing", "aten_output_rows": compact_rows} if arm == "b" else {
            "stage": "source_coalescing", "aten_output_rows": VOCAB,
        }
        require(all(call == expected for call in case["coalescing"].values()),
                f"Rank {rank} {name}: active source stage or actual ATen row count differs from {expected}")
        require(name != HIGH or unique >= compact_rows, f"Rank {rank}: high-U fixture unexpectedly needs padding")
        require(name != LOW or rank != fixture["provenance"]["lowest_u_rank"]
                or unique == fixture["provenance"]["lowest_u"] < compact_rows,
                f"Rank {rank}: low-U fixture did not exercise padding")
        require(exact["passes"]["initial"] == exact["passes"]["reset"],
                f"Rank {rank} {name}: reset did not reproduce the first backward")
        for step in exact["passes"].values():
            messages = step["messages"]
            request, values = messages["request_exchange"], messages["value_exchange"]
            hashes = (fixture["input_ids_int64_sha256"], exact["master_fp32_sha256"], step["output_bf16_sha256"],
                      step["gradient_fp32_sha256"], messages["sent_bf16_sha256"], messages["received_bf16_sha256"],
                      request["sent_sha256"], request["received_sha256"], values["sent_sha256"], values["received_sha256"])
            require(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes),
                    f"Rank {rank} {name}: missing or malformed tensor digest")
            send, receive = messages["send_split_rows"], messages["receive_split_rows"]
            require(len(send) == len(receive) == WORLD and sum(send) == unique
                    and messages["sent_bytes"] == sum(send) * WIDTH * 2
                    and messages["received_bytes"] == sum(receive) * WIDTH * 2
                    and step["gradient_fp32_bytes"] == (VOCAB // WORLD) * WIDTH * 4
                    and request["dtype"] == "torch.int64" and values["dtype"] == "torch.bfloat16"
                    and request["send_split_rows"] == values["receive_split_rows"] == send
                    and request["receive_split_rows"] == values["send_split_rows"] == receive,
                    f"Rank {rank} {name}: row splits, message sizes or dtype differ")


def run(args):
    root = args.source_root.resolve(strict=True)
    owner_file = (root / OWNER).resolve(strict=True)
    require(re.fullmatch(r"[0-9a-f]{64}", args.owner_source_sha256)
            and digest(owner_file) == args.owner_source_sha256, "Row-owner source hash differs from the pin")
    require(not args.output.exists(), f"Refusing to replace {args.output}")
    os.environ["ROW_OWNER_REPLAY_SOURCE_ROOT"] = str(root)

    import row_owner_route_numeric as numeric
    from compact_coalescing_replay import low_fixture
    from torchtitan.distributed.gfsdp import row_owner_embedding

    require(numeric.SOURCE_ROOT == root and Path(row_owner_embedding.__file__).resolve() == owner_file,
            "Loaded row-owner implementation differs from --source-root")
    torch, dist = numeric.torch, numeric.dist
    rank = numeric.check_runtime()
    require(int(os.environ["RANK"]) == rank and int(os.environ["LOCAL_WORLD_SIZE"]) == WORLD,
            "Replay requires one node with matching global and local ranks")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    sm_count = torch.cuda.get_device_properties(rank).multi_processor_count
    require(sm_count == 132, f"Expected H100 with 132 SMs, got {sm_count}")
    runtime = {"torch": torch.__version__, "torch_git": torch.version.git_version, "cuda": torch.version.cuda,
               "nccl": list(torch.cuda.nccl.version()), "gpu": torch.cuda.get_device_name(rank), "sm_count": sm_count}
    harness = runpy.run_path(str(root / "runs/ffn-row-owner/embedding_numeric.py"))
    load_fixture = runpy.run_path(str(root / "runs/production-data-abcd/component_fixtures.py"))["load_fixture"]
    high_reference = read_json(REFERENCE)["input_ids_sha256_by_rank"]
    tensor_digest = harness["tensor_sha256"]

    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        require(dist.get_rank() == rank and dist.get_world_size() == WORLD, "Unexpected distributed rank or mesh size")
        captured, receipt = load_fixture(args.high_fixture, rank)
        high_ids = captured.pop("input_ids")
        del captured
        require(tensor_digest(high_ids, torch) == receipt["input_ids_int64_sha256"] == high_reference[rank],
                f"Rank {rank}: high-U IDs differ from the accepted native capture")
        low_ids, low_provenance = low_fixture(args.low_census, args.data_root, rank)
        mesh = numeric.init_device_mesh("cuda", (WORLD,), mesh_dim_names=("fsdp",))
        cases = {}
        for name, ids, provenance in (
            (HIGH, high_ids, {"tensors_sha256": receipt["tensors_sha256"]}), (LOW, low_ids, low_provenance),
        ):
            require(ids.dtype == torch.int64 and tuple(ids.shape) == (2, 4096)
                    and bool(((ids >= 0) & (ids < VOCAB)).all()), f"Rank {rank} {name}: invalid real fixture IDs")
            fixture = {"input_ids_int64_sha256": tensor_digest(ids, torch), "tokens": ids.numel(),
                       "unique_tokens": ids.unique().numel(), "provenance": provenance}
            ids = ids.cuda()
            upstream, upstream_receipt = harness["build_shared_upstream"](ids, args.seed, rank, torch)
            tensors, coalescing = capture_replay(numeric, harness, mesh, ids, upstream, args.seed)
            cases[name] = {"fixture": fixture, "upstream": upstream_receipt, "exact_tensors": tensors,
                           "coalescing": coalescing}
            del upstream, ids

        ranks = [None] * WORLD
        row = {"rank": rank, "runtime": runtime, "cases": cases, "owner_source_sha256": digest(owner_file)}
        dist.all_gather_object(ranks, row)
        require(all(row["owner_source_sha256"] == args.owner_source_sha256 for row in ranks),
                "Row-owner implementation changed during replay")
        if rank == 0:
            write_json(args.output, {
                "schema_version": SCHEMA, "arm": args.arm, "source_root": str(root), "source_head": head,
                "owner_source_sha256": args.owner_source_sha256, "driver_sha256": digest(SCRIPT),
                "seed": args.seed, "ranks": ranks,
            })
        for row in ranks:
            check_cases(row, args.arm)
    finally:
        dist.destroy_process_group()


def compare(args):
    control, candidate = (read_json(path) for path in (args.control, args.candidate))
    require((control["arm"], candidate["arm"]) == ("a", "b")
            and control["schema_version"] == candidate["schema_version"] == SCHEMA
            and control["source_root"] != candidate["source_root"]
            and control["owner_source_sha256"] != candidate["owner_source_sha256"],
            "Expected separate control and candidate source receipts")
    require(control["source_head"] == candidate["source_head"]
            and control["seed"] == candidate["seed"] and control["driver_sha256"] == candidate["driver_sha256"],
            "The pinned source HEAD, seed or harness differs between arms")
    rows, mismatches, padded = [], [], []
    for arm in (control, candidate):
        require([rank["rank"] for rank in arm["ranks"]] == list(range(WORLD))
                and all(rank["owner_source_sha256"] == arm["owner_source_sha256"] for rank in arm["ranks"]),
                "Missing ranks or distinct owner sources within one arm")
        for rank in arm["ranks"]:
            check_cases(rank, arm["arm"])
    cohort = set()
    for left, right in zip(control["ranks"], candidate["ranks"], strict=True):
        rank = left["rank"]
        compare_fields = ("fixture", "upstream", "exact_tensors")
        match = {name: list(differences({k: left["cases"][name][k] for k in compare_fields},
                                        {k: right["cases"][name][k] for k in compare_fields}))
                 for name in (HIGH, LOW)}
        if left["runtime"] != right["runtime"]:
            mismatches.append(f"rank{rank}.runtime")
        mismatches.extend(f"rank{rank}.{name}.{key}" for name, keys in match.items() for key in keys)
        cases = right["cases"]
        low = cases[LOW]["fixture"]
        provenance = low["provenance"]
        cohort.add((provenance["microbatch_zero_based"], provenance["lowest_u"], provenance["lowest_u_rank"]))
        rows.append({"rank": rank, "case_matches": {name: not keys for name, keys in match.items()},
                     "unique_tokens": {name: cases[name]["fixture"]["unique_tokens"] for name in (HIGH, LOW)},
                     "coalescing": {name: cases[name]["coalescing"] for name in (HIGH, LOW)}})
        if cases[LOW]["coalescing"]["initial"]["aten_output_rows"] > low["unique_tokens"]:
            padded.append(rank)
    require(len(cohort) == 1 and next(iter(cohort))[2] in padded, "The same natural eight-rank padded cohort is required")
    require(not args.output.exists(), f"Refusing to replace {args.output}")
    write_json(args.output, {
        "schema_version": SCHEMA, "exact": not mismatches, "mismatches": mismatches,
        "controls": {"a": str(args.control), "b": str(args.candidate)},
        "owner_source_sha256": {"a": control["owner_source_sha256"], "b": candidate["owner_source_sha256"]},
        "low_fixture_cohort": dict(zip(("microbatch_zero_based", "lowest_u", "lowest_u_rank"), next(iter(cohort)))),
        "low_fixture_padded_ranks": padded, "ranks": rows,
        "scope": "Real IDs with a shared seeded BF16 upstream derivative, exact embedding replay across eight ranks",
    })
    return int(bool(mismatches))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run", help="run one exact arm under torchrun")
    execute.add_argument("--arm", choices=("a", "b"), required=True)
    execute.add_argument("--source-root", type=Path, required=True)
    execute.add_argument("--owner-source-sha256", required=True)
    execute.add_argument("--high-fixture", type=Path, required=True)
    execute.add_argument("--low-census", type=Path, required=True)
    execute.add_argument("--data-root", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--seed", type=int, default=2048)
    verify = commands.add_parser("compare", help="compare the completed arm receipts offline")
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
