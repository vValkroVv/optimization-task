#!/usr/bin/env python3
"""Replay the final row-owner changes with identical inputs, outside training."""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import runpy
import subprocess
import sys

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent.parent / "embedding-combined-001"))

from campaign import differences, digest, read_json, require, write_json
from compact_owner_numeric import HIGH, LOW, OWNER, PASSES, REFERENCE, VOCAB, WIDTH, WORLD
from narrow_owner_exact import capture_case, comparable

CAPS = {"shared": None, "narrow-4": 4, "combined": 4}
CASE_NAMES = (HIGH, LOW, "single-remote-owner", "padding-boundary", "token-count-boundary", "all-unique")
ROWS_PER_OWNER = VOCAB // WORLD
SEED = 2048


def repeated_ids(torch, n, unique, seed):
    if not n:
        return torch.empty((1, 0), dtype=torch.int64)
    # Include the last row of some shards and the first row of the others.
    i = torch.arange(unique, dtype=torch.int64)
    row = i // WORLD
    row = torch.where(i.remainder(2) == 0, row, ROWS_PER_OWNER - 1 - row)
    vocabulary = (i % WORLD) * ROWS_PER_OWNER + row
    reserve = min(n - unique, 16)
    ids = vocabulary[torch.arange(n - reserve) % unique]
    ids = torch.cat((vocabulary[:1].expand(reserve), ids[torch.randperm(
        n - reserve, generator=torch.Generator().manual_seed(seed)
    )]))
    return ids.reshape(1, n)


def make_cases(args, rank, torch, root):
    from compact_coalescing_replay import low_fixture

    load_fixture = runpy.run_path(str(root / "runs/production-data-abcd/component_fixtures.py"))["load_fixture"]
    captured, receipt = load_fixture(args.high_fixture, rank)
    high_ids = captured.pop("input_ids")
    del captured
    require(receipt["input_ids_int64_sha256"] == read_json(REFERENCE)["input_ids_sha256_by_rank"][rank],
            "High-U input is not the pinned real fixture")
    low_ids, low_provenance = low_fixture(args.low_census, args.data_root, rank)
    yield HIGH, high_ids, {"capture_id_sha256": receipt["input_ids_int64_sha256"]}, False
    yield LOW, low_ids, low_provenance, False

    # Every source sends the same row to rank 7, so seven ranks receive no requests.
    yield "single-remote-owner", torch.full((2, 4096), VOCAB - 1, dtype=torch.int64), {}, True
    counts = (1, 12, 54, 55, 130, 131, 132, 133)
    yield "padding-boundary", repeated_ids(torch, 8192, counts[rank], SEED + rank), {}, True
    sizes = (0, 1, 3071, 3072, 3073, 3073, 3072, 3073)
    uniques = (0, 1, 131, 131, 131, 132, 133, 133)
    yield "token-count-boundary", repeated_ids(torch, sizes[rank], uniques[rank], SEED + rank), {}, True
    order = torch.randperm(8192, generator=torch.Generator().manual_seed(SEED + rank))
    ids = (order % WORLD) * ROWS_PER_OWNER + order // WORLD + rank * 1024
    yield "all-unique", ids.reshape(2, 4096), {}, False


def fixture_inputs(torch, harness, name, ids, rank, provenance, cancellation):
    hash_tensor = harness["tensor_sha256"]
    fixture = {
        "tokens": ids.numel(),
        "unique_tokens": ids.unique().numel(),
        "ids_sha256": hash_tensor(ids, torch),
        "provenance": provenance,
    }
    require(ids.dtype == torch.int64 and bool(((ids >= 0) & (ids < VOCAB)).all()), f"{name}: invalid token IDs")
    if name == HIGH:
        require(fixture["ids_sha256"] == provenance["capture_id_sha256"], "Real high-U token bytes changed")
    ids = ids.cuda()
    gradient, upstream = harness["build_shared_upstream"](ids, SEED, rank, torch)
    if cancellation and ids.numel() >= 16:
        # The first 16 tokens are duplicates. Exercise FP32 summation, BF16
        # subnormal input handling, and donor-order cancellation separately.
        values = torch.tensor([
            [16777216, 1, -16777216, -1],
            [1, 0.5, -1, -0.5],
            [2.0 ** -133, -2.0 ** -133, 2.0 ** -133, 0.0],
            [4096 if rank % 2 == 0 else -4096, 0.5, 0.0, 0.0],
        ], dtype=torch.bfloat16, device=ids.device).T
        gradient.view(-1, WIDTH)[:16, :4] = values.repeat(4, 1)
        upstream["cancellation"] = True
        upstream["combined_bf16_sha256"] = hash_tensor(gradient, torch)
    return fixture, ids, gradient, upstream


def validate_cases(row, variant):
    require(set(row["cases"]) == set(CASE_NAMES), "Missing scenarios")
    for name, case in row["cases"].items():
        n, unique = (case["fixture"][key] for key in ("tokens", "unique_tokens"))
        exact = case["exact_tensors"]
        require(set(exact["passes"]) == set(case["coalescing"]) == set(PASSES),
                f"{name}: missing forward/backward pass")
        compact = variant == "combined"
        rows = (unique if n <= 3072 else max(unique, 132)) if compact else VOCAB
        stage = "compact_source_coalescing" if compact else "source_coalescing"
        expected = {"stage": stage, "aten_output_rows": rows}
        require(all(call == expected for call in case["coalescing"].values()),
                f"{name}: wrong coalescing branch or output rows on rank {row['rank']}")
        require(exact["communicator"]["requested_max_ctas"] == CAPS[variant], f"{name}: wrong NCCL cap")
        require(exact["passes"]["initial"] == exact["passes"]["reset"], f"{name}: reset differs from first backward")
        for step in exact["passes"].values():
            messages = step["messages"]
            send, receive = messages["send_split_rows"], messages["receive_split_rows"]
            require(len(send) == len(receive) == WORLD and sum(send) == unique
                    and messages["sent_bytes"] == sum(send) * WIDTH * 2
                    and messages["received_bytes"] == sum(receive) * WIDTH * 2
                    and step["gradient_fp32_bytes"] == ROWS_PER_OWNER * WIDTH * 4,
                    f"{name}: unexpected gradient splits or sizes")
            if name == "single-remote-owner":
                require(send == [0] * 7 + [1] and receive == [int(row["rank"] == 7)] * WORLD,
                        "Single-owner scenario did not produce the expected empty splits")


def run(args):
    root = args.source_root.resolve(strict=True)
    owner_file = (root / OWNER).resolve(strict=True)
    require(digest(owner_file) == args.owner_source_sha256, "Owner source differs from the pin")
    require(not args.output.exists(), f"Refusing to replace {args.output}")
    os.environ["ROW_OWNER_REPLAY_SOURCE_ROOT"] = str(root)
    import row_owner_route_numeric as numeric
    from torchtitan.distributed.gfsdp import row_owner_embedding

    require(numeric.SOURCE_ROOT == root and Path(row_owner_embedding.__file__).resolve() == owner_file,
            "Imported row-owner source differs from the expected checkout")
    torch, dist = numeric.torch, numeric.dist
    rank = numeric.check_runtime()
    gpu = torch.cuda.get_device_properties(rank)
    require(gpu.multi_processor_count == 132 and int(os.environ["LOCAL_WORLD_SIZE"]) == WORLD
            and int(os.environ["RANK"]) == rank, "Expected same-node eight H100 ranks with 132 SMs")
    harness = runpy.run_path(str(root / "runs/ffn-row-owner/embedding_numeric.py"))
    runtime = {"torch": torch.__version__, "torch_git": torch.version.git_version,
               "cuda": torch.version.cuda, "nccl": list(torch.cuda.nccl.version()),
               "gpu": gpu.name, "uuid": str(gpu.uuid)}
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        mesh = numeric.init_device_mesh("cuda", (WORLD,), mesh_dim_names=("fsdp",))
        cases = {}
        for name, ids, provenance, cancellation in make_cases(args, rank, torch, root):
            if rank == 0:
                print(f"{args.variant}: {name}", flush=True)
            fixture, ids, upstream, upstream_receipt = fixture_inputs(
                torch, harness, name, ids, rank, provenance, cancellation
            )
            exact, coalescing = capture_case(numeric, harness, mesh, ids, upstream, SEED, CAPS[args.variant])
            cases[name] = {"fixture": fixture, "upstream": upstream_receipt,
                           "exact_tensors": exact, "coalescing": coalescing}
            del upstream, ids
        row = {"rank": rank, "runtime": runtime, "cases": cases}
        ranks = [None] * WORLD
        dist.all_gather_object(ranks, row)
        for row in ranks:
            validate_cases(row, args.variant)
        require(digest(owner_file) == args.owner_source_sha256, "Owner source changed during replay")
        if rank == 0:
            write_json(args.output, {
                "variant": args.variant, "root": str(root),
                "owner_source_sha256": args.owner_source_sha256,
                "driver_sha256": digest(SCRIPT),
                "head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                "seed": SEED, "ranks": ranks,
            })
    finally:
        dist.destroy_process_group()


def compare(args):
    receipts = dict(zip(CAPS, map(read_json, (args.shared, args.narrow, args.combined)), strict=True))
    require({r["variant"] for r in receipts.values()} == set(CAPS), "Missing comparison variant")
    require(len({r["owner_source_sha256"] for r in receipts.values()}) == len(CAPS),
            "Expected three distinct source variants")
    for key in ("driver_sha256", "head", "seed"):
        require(len({r[key] for r in receipts.values()}) == 1, f"A/B changed {key}")
    for variant, receipt in receipts.items():
        require(receipt["variant"] == variant, "Incorrect comparison variant")
        require([row["rank"] for row in receipt["ranks"]] == list(range(WORLD)), "Missing GPU ranks")
        for row in receipt["ranks"]:
            validate_cases(row, variant)
    results = {}
    for left, right in (("shared", "narrow-4"), ("narrow-4", "combined"), ("shared", "combined")):
        cases = {name: {"ranks_equal": 0, "mismatches": []} for name in CASE_NAMES}
        for a, b in zip(receipts[left]["ranks"], receipts[right]["ranks"], strict=True):
            require(a["runtime"] == b["runtime"], "A/B runtime or GPU changed")
            for name, result in cases.items():
                changed = list(differences(comparable(a["cases"][name]), comparable(b["cases"][name])))
                result["ranks_equal"] += int(not changed)
                result["mismatches"].extend(f"rank{a['rank']}.{path}" for path in changed)
        results[f"{left} -> {right}"] = cases
    exact = all(case["ranks_equal"] == WORLD for cases in results.values() for case in cases.values())
    require(not args.output.exists(), f"Refusing to replace {args.output}")
    write_json(args.output, {"exact": exact, "gpu_ranks": WORLD, "replays_per_scenario": list(PASSES),
                             "comparisons": results})
    print(f"Bitwise match across all comparisons: {exact}", flush=True)
    return int(not exact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--variant", choices=CAPS, required=True)
    launch.add_argument("--owner-source-sha256", required=True)
    for name in ("source-root", "high-fixture", "low-census", "data-root", "output"):
        launch.add_argument(f"--{name}", type=Path, required=True)
    verify = commands.add_parser("compare")
    for name in ("shared", "narrow", "combined", "output"):
        verify.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
