"""Verify and summarize the eight-rank isolated LM-head timing matrix."""

import argparse
from collections import Counter
from itertools import permutations
import json
from math import isclose, isfinite
from pathlib import Path
import re
from statistics import median


ARMS = ("baseline", "regular", "alias")
BLOCKS = tuple(range(6))
CHUNKS = (2, 8)
RANKS = tuple(range(8))
WARMUP = SAMPLES = 10
ITERATIONS = list(range(WARMUP + 1, WARMUP + SAMPLES + 1))
INVOCATIONS = WARMUP + SAMPLES + 3
BASELINE = "e3871b36c0be1cb5c027b811d1684558fb11b02b"
FIXTURE_SHA = "15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727"
DIGESTS = {
    "loss": ([], "torch.float32"),
    "dx": ([2, 2, 4096, 4096], "torch.bfloat16"),
    "dw": ([16032, 4096], "torch.float32"),
    "z_loss": ([], "torch.float32"),
    "z_dx": ([2, 4096, 4096], "torch.bfloat16"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def group_name(block, chunks, arm):
    return f"b{block}-c{chunks}-{arm}"


def verify_schedule(schedule):
    names = {
        group_name(block, chunks, arm): (block, chunks, arm)
        for block in BLOCKS for chunks in CHUNKS for arm in ARMS
    }
    require(isinstance(schedule, list) and len(schedule) == len(names)
            and set(schedule) == names.keys(),
            "schedule.json must include each planned process group exactly once")
    for chunks in CHUNKS:
        observed = []
        for block in BLOCKS:
            observed.append(tuple(
                names[name][2] for name in schedule if names[name][:2] == (block, chunks)
            ))
        require(set(observed) == set(permutations(ARMS)),
                f"C{chunks}: launch order must cover all six arm permutations")
    for index in range(0, len(schedule), len(ARMS)):
        members = schedule[index:index + len(ARMS)]
        require(len({names[name][:2] for name in members}) == 1,
                f"schedule splits the matched arms for {members}")


def verify_series(values, field, *, positive=False):
    require(len(values) == SAMPLES and all(
        type(value) in (int, float) and isfinite(value)
        and (value > 0 if positive else value >= 0) for value in values
    ), f"{field}: expected {SAMPLES} valid samples")


def verify_digests(digests):
    require(digests.keys() == DIGESTS.keys(),
            "expected full loss, dX, dW, z-loss and z-dX digests")
    for name, digest in digests.items():
        require((digest["shape"], digest["dtype"]) == DIGESTS[name],
                f"{name} shape or dtype differs from the expected full tensor/local shard")
        require(re.fullmatch(r"[0-9a-f]{64}", digest["sha256"]), f"invalid {name} SHA256")


def verify_activation(receipt, chunks, arm):
    candidate = arm != "baseline"
    compile_config = receipt["compile"]
    require(compile_config["backend"] == "inductor" and compile_config["scope"] == "native loss"
            and compile_config["counters"]["unique_graphs"] > 0
            and compile_config["cuda_graphs"] is False,
            f"unexpected loss compile settings: {compile_config}")
    require(receipt["expandable_segments"] is candidate, "unexpected live allocator setting")
    require(receipt["folded_rows_per_chunk"] == 16384 // chunks,
            "unexpected active folded rows per chunk")
    require(receipt["capture_rank"] == 0 and receipt["fixture_sha256"] == FIXTURE_SHA,
            "unexpected production-tokenizer capture identity")
    require((receipt["source"] == BASELINE) == (arm == "baseline"),
            "baseline and candidate must use their distinct GIFT revisions")
    activation = receipt["activation"]
    for field in ("folded", "direct_bf16_dhidden", "early_logits_release", "fused_hidden_z_loss"):
        require(activation[field] is candidate, f"{field} does not match the planned arm")
    native_ce = "torchtitan.experiments.four_fusion.stable_ce" if candidate else "torchtitan.components.loss"
    require(activation["staged_chunk_calls"] == [candidate] * chunks
            and activation["ce"] == f"{native_ce}.cross_entropy_loss",
            "unexpected staging or compiled native CE activation")
    require(receipt["ce_route_calls"] == {
        "native": 0 if candidate else 2 * chunks,
        "exact_setup": chunks if candidate else 0,
    }, "unexpected native or exact CE route count")
    exact_calls = INVOCATIONS * chunks if candidate else 0
    require(activation["exact_ce_calls"] == exact_calls, "unexpected total exact CE calls")
    flags = activation["gift_flags"]
    matrix = "GIFT_EXPERIMENT_LM_HEAD_MATRIX"
    require(flags[matrix] == ("new" if candidate else "")
            and flags[f"{matrix}_CHUNKS"] == (str(chunks) if candidate else "")
            and all(value == str(int(candidate)) for key, value in flags.items()
                    if key not in (matrix, f"{matrix}_CHUNKS")), "unexpected GIFT flags")
    exact = receipt["candidate"]
    writer = activation["dw_writer"]
    if candidate:
        require(writer["pair"] == {2: "9.14", 8: "5.5"}[chunks],
                f"unexpected staged dW route: {writer}")
        require(exact["method"] == "cuda-13" and exact["calls"] == exact_calls,
                f"unexpected exact CE route: {exact}")
    else:
        require(exact is None and writer is None, "native baseline used a candidate path")


def verify_gather(gather, rank, arm):
    alias = arm == "alias"
    require(gather["mode"] == ("alias" if alias else "stack")
            and gather["group_ranks"] == list(RANKS)
            and (gather["group_rank"], gather["group_size"]) == (rank, len(RANKS)),
            "unexpected GFSDP all-gather mode or NCCL group")
    expected_calls = INVOCATIONS + 1  # Head gather before any replay iterations.
    require(gather["gather_calls"] == gather["native_all_gather_into_tensor_calls"]
            == expected_calls, "wrong number of native NCCL all-gather calls")
    casts = gather["master_to_bf16_casts"]
    require(0 < casts <= expected_calls and (not alias or casts == expected_calls),
            "wrong number of FP32 master-to-BF16 gather casts")
    input_bytes = 16032 * 4096 * 2  # BF16 elements in one rank's shard.
    offset = rank * input_bytes
    require(gather["input_bytes"] == input_bytes and gather["output_bytes"] == input_bytes * len(RANKS)
            and gather["rank_slot_offset_bytes"] == offset, "wrong BF16 gather shard extent")
    input_ptr, output_ptr = gather["input_data_ptr"], gather["output_data_ptr"]
    input_storage, output_storage = gather["input_storage_ptr"], gather["output_storage_ptr"]
    require(min(input_ptr, output_ptr, input_storage, output_storage) > 0,
            "missing BF16 gather buffer pointers")
    require((input_storage == output_storage) == alias
            and gather["distinct_bf16_gather_storages"] == (1 if alias else 2),
            "wrong BF16 gather storage ownership")
    if alias:
        require(gather["input_offset_bytes"] == offset == input_ptr - output_ptr,
                "alias input pointer does not address this rank's slot")
    else:
        require(gather["input_offset_bytes"] is None
                and (input_ptr + input_bytes <= output_ptr
                     or output_ptr + gather["output_bytes"] <= input_ptr),
                "regular gather input and output overlap")


def verify_rank(receipt, rank, chunks, arm):
    require(receipt["rank"] == rank and receipt["world_size"] == len(RANKS),
            f"expected rank {rank} of {len(RANKS)}")
    require(receipt["chunks"] == chunks and receipt["arm"] == arm,
            f"expected chunks={chunks}, arm={arm}")
    require((receipt["warmup"], receipt["samples"]) == (WARMUP, SAMPLES),
            f"expected {WARMUP} warmups and {SAMPLES} samples")
    require(receipt["group_ranks"] == list(RANKS), "expected collective group of ranks 0–7")
    require(receipt["sample_iterations"] == ITERATIONS, "timed iterations must be 11–20")
    require(type(receipt["process_id"]) is int and receipt["process_id"] > 0,
            "expected a worker PID")
    require(receipt["torchelastic_run_id"], "expected a torchrun rendezvous ID")
    verify_series(receipt["elapsed_ms"], "elapsed_ms", positive=True)
    verify_series(receipt["total_gpu_gib"], "total_gpu_gib", positive=True)
    verify_series(receipt["host_submit_ms"], "host_submit_ms")
    verify_series(receipt["host_wait_ms"], "host_wait_ms")
    require(all(isclose(elapsed, submit + wait, rel_tol=1e-9, abs_tol=1e-7)
                for elapsed, submit, wait in zip(
                    receipt["elapsed_ms"], receipt["host_submit_ms"], receipt["host_wait_ms"],
                    strict=True,
                )), "host submission plus synchronization wait must equal integrated elapsed time")
    require(receipt["allocator_retries_and_ooms"] == {"num_alloc_retries": 0, "num_ooms": 0},
            f"allocator retries or OOMs: {receipt['allocator_retries_and_ooms']}")
    require(receipt["native_fp32_sum_reduce_scatter_calls"] == INVOCATIONS
            and receipt["reduction"] ==
            "Eight-rank NCCL FP32 SUM reduce-scatter with gradient divide factor 1",
            "wrong native FP32 SUM reduce-scatter route")
    require(receipt["accumulation"] == "Primed sharded weight gradient retained across timed calls",
            "expected sharded gradient accumulation across the timed window")
    verify_gather(receipt["gather_probe"], rank, arm)
    verify_activation(receipt, chunks, arm)
    verify_digests(receipt["digests"])


def verify_invocation(directory, chunks, arm):
    require((directory / "exit-code").read_text().strip() == "0",
            f"{directory}: group launcher must exit successfully")
    invocation = read(directory / "invocation.json")
    expected = [
        "--nproc-per-node=8", "replay8.py", "--chunks", str(chunks), "--arm", arm,
        "--out", directory.name, "--warmup", str(WARMUP), "--samples", str(SAMPLES),
    ]
    cmd = [Path(part).name if part.startswith("/") else part for part in invocation["cmd"]]
    require(cmd[-len(expected):] == expected and "--standalone" in cmd,
            f"{directory}: invocation does not match the planned eight-rank arm")
    expandable = arm != "baseline"
    require(invocation["PYTORCH_ALLOC_CONF"] == f"expandable_segments:{expandable}",
            f"{directory}: unexpected allocator setting")
    devices = invocation["CUDA_VISIBLE_DEVICES"].split(",")
    require(len(set(devices)) == len(devices) == len(RANKS)
            and all(device.startswith("GPU-") for device in devices),
            f"{directory}: expected eight distinct GPU UUIDs")
    return devices


def read_group(directory, chunks, arm):
    paths = [directory / f"rank-{rank}.json" for rank in RANKS]
    require(set(directory.glob("rank-*.json")) == set(paths),
            f"{directory}: expected exactly eight rank receipts")
    devices = verify_invocation(directory, chunks, arm)
    receipts = []
    for rank, path in enumerate(paths):
        receipt = read(path)
        try:
            verify_rank(receipt, rank, chunks, arm)
            device = receipt["device"]
            require(device["visible_uuid"] == devices[rank]
                    and device["visible"] == ",".join(devices)
                    and device["rank"] == device["current_device"] == rank
                    and "H100" in device["name"],
                    "worker GPU UUID does not match launch topology")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}: {error}") from error
        receipts.append(receipt)
    return receipts


def distribution(values):
    return {
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def group_stats(receipts):
    slowest = [
        max(RANKS, key=lambda rank: receipts[rank]["elapsed_ms"][index])
        for index in range(SAMPLES)
    ]
    max_time = [receipts[rank]["elapsed_ms"][index] for index, rank in enumerate(slowest)]
    peak, rank, sample = max(
        (total, receipt["rank"], index)
        for receipt in receipts
        for index, total in enumerate(receipt["total_gpu_gib"])
    )
    return {
        "group_median_ms": median(max_time),
        "iteration_max_rank_elapsed_ms": max_time,
        "iteration_slowest_rank": slowest,
        "iteration_slowest_rank_host_submit_ms": [
            receipts[rank]["host_submit_ms"][index] for index, rank in enumerate(slowest)
        ],
        "iteration_slowest_rank_host_wait_ms": [
            receipts[rank]["host_wait_ms"][index] for index, rank in enumerate(slowest)
        ],
        "peak_gift_total_gib": peak,
        "peak_rank": rank,
        "peak_iteration": WARMUP + sample + 1,
        "per_rank_bf16_gather_cast_calls": [
            receipt["gather_probe"]["master_to_bf16_casts"] for receipt in receipts
        ],
        "per_rank_gather_init_allocation_delta_bytes": [
            receipt["gather_probe"]["lazy_init_allocation_delta_bytes"] for receipt in receipts
        ],
    }


def arm_stats(groups):
    medians = {name: group["group_median_ms"] for name, group in groups.items()}
    peak_group = max(groups, key=lambda name: groups[name]["peak_gift_total_gib"])
    return {
        "group_medians_ms": medians,
        "group_median_summary_ms": distribution(list(medians.values())),
        "median_group_host_submit_at_slowest_rank_ms": median(
            median(group["iteration_slowest_rank_host_submit_ms"]) for group in groups.values()
        ),
        "median_group_host_wait_at_slowest_rank_ms": median(
            median(group["iteration_slowest_rank_host_wait_ms"]) for group in groups.values()
        ),
        "peak_gift_total_gib": groups[peak_group]["peak_gift_total_gib"],
        "peak_group": peak_group,
        "peak_rank": groups[peak_group]["peak_rank"],
        "peak_iteration": groups[peak_group]["peak_iteration"],
    }


def paired_stats(groups, chunks, candidate, reference):
    # Pair matching launch blocks before taking medians across process groups.
    pairs = []
    for block in BLOCKS:
        left = groups[group_name(block, chunks, reference)]
        right = groups[group_name(block, chunks, candidate)]
        delta = right["group_median_ms"] - left["group_median_ms"]
        pairs.append({
            "block": block,
            "elapsed_delta_ms": delta,
            "elapsed_delta_pct_of_reference": 100 * delta / left["group_median_ms"],
            "peak_gift_total_delta_gib": right["peak_gift_total_gib"] - left["peak_gift_total_gib"],
        })
    elapsed = [pair["elapsed_delta_ms"] for pair in pairs]
    signs = Counter("slower" if value > 0 else "faster" if value < 0 else "equal"
                    for value in elapsed)
    return {
        "pairs": pairs,
        "elapsed_delta_summary_ms": distribution(elapsed),
        "elapsed_delta_summary_pct_of_reference": distribution(
            [pair["elapsed_delta_pct_of_reference"] for pair in pairs]
        ),
        "peak_gift_total_delta_summary_gib": distribution(
            [pair["peak_gift_total_delta_gib"] for pair in pairs]
        ),
        "elapsed_delta_signs": {sign: signs[sign] for sign in ("faster", "equal", "slower")},
    }


def digest_stats(raw_groups, chunks):
    comparisons = {}
    for candidate, reference in (("alias", "regular"), ("regular", "baseline")):
        mismatches = {name: [] for name in sorted(DIGESTS)}
        for block in BLOCKS:
            left = raw_groups[group_name(block, chunks, reference)]
            right = raw_groups[group_name(block, chunks, candidate)]
            for rank in RANKS:
                for name in sorted(DIGESTS):
                    if left[rank]["digests"][name] != right[rank]["digests"][name]:
                        mismatches[name].append({"block": block, "rank": rank})
        comparisons[f"{candidate}_vs_{reference}"] = {
            "compared_block_rank_pairs": len(BLOCKS) * len(RANKS),
            "all_equal": not any(mismatches.values()),
            "mismatches": mismatches,
        }
    return comparisons


def verify_matching_controls(raw_groups):
    processes = {row["process_id"] for rows in raw_groups.values() for row in rows}
    require(len(processes) == len(raw_groups) * len(RANKS),
            "expected a fresh set of eight workers per process group")
    run_ids = set()
    for name, rows in raw_groups.items():
        ids = {row["torchelastic_run_id"] for row in rows}
        require(len(ids) == 1 and run_ids.isdisjoint(ids), f"{name}: reused or mismatched torchrun IDs")
        run_ids.update(ids)
    for rank in RANKS:
        baseline = raw_groups[group_name(0, CHUNKS[0], "baseline")][rank]
        for name, rows in raw_groups.items():
            row = rows[rank]
            source_arm = "baseline" if row["arm"] == "baseline" else "regular"
            matched = raw_groups[group_name(0, row["chunks"], source_arm)][rank]
            for field in ("fixture", "fixture_sha256", "hidden_z_fixture_sha256", "device",
                          "harness_sha256", "historical_replay_sha256"):
                require(row[field] == baseline[field],
                        f"{name}, rank {rank}: {field} differs from the matched baseline")
            source = raw_groups[group_name(0, CHUNKS[0], source_arm)][0]
            for field in ("source", "source_hashes"):
                require(row[field] == source[field],
                        f"{name}, rank {rank}: {field} differs for the matched GIFT revision")
            for field in ("activation", "candidate", "compile"):
                require(row[field] == matched[field],
                        f"{name}, rank {rank}: {field} differs for its chunk/compute route")


def summarize(root):
    schedule = read(root / "schedule.json")
    verify_schedule(schedule)
    raw_groups = {
        group_name(block, chunks, arm):
        read_group(root / group_name(block, chunks, arm), chunks, arm)
        for block in BLOCKS for chunks in CHUNKS for arm in ARMS
    }
    verify_matching_controls(raw_groups)
    groups = {name: group_stats(raw_groups[name]) for name in schedule}
    results = {}
    for chunks in CHUNKS:
        arm_results = {
            arm: arm_stats({group_name(block, chunks, arm): groups[group_name(block, chunks, arm)]
                            for block in BLOCKS})
            for arm in ARMS
        }
        comparisons = {}
        for candidate, reference in (("alias", "regular"), ("regular", "baseline"),
                                     ("alias", "baseline")):
            comparison = paired_stats(groups, chunks, candidate, reference)
            comparison["arm_peak_gift_total_delta_gib"] = (
                arm_results[candidate]["peak_gift_total_gib"]
                - arm_results[reference]["peak_gift_total_gib"]
            )
            comparisons[f"{candidate}_minus_{reference}"] = comparison
        results[f"c{chunks}"] = {
            "arms": arm_results,
            "resolved_arm_configuration": {
                arm: {
                    field: raw_groups[group_name(0, chunks, arm)][0][field]
                    for field in ("source", "source_hashes", "compile", "activation", "candidate",
                                  "expandable_segments")
                }
                for arm in ARMS
            },
            "paired_comparisons": comparisons,
            "digest_comparisons": digest_stats(raw_groups, chunks),
        }
    return {
        "raw_directory": str(root),
        "warmup_iterations": WARMUP,
        "measured_iterations": SAMPLES,
        "ranks_per_group": len(RANKS),
        "launch_order": schedule,
        "latency_scope": "Head forward and backward through CUDA synchronization; maximum "
                         "rank-local elapsed time per aligned iteration. Barriers, memory "
                         "sampling and digest capture are outside the timer.",
        "host_scope": "Host submit and synchronization wait of the same slowest rank selected "
                      "for each integrated interval. Synchronization wait includes work still "
                      "running on the device.",
        "unavailable_timing_scopes": ["device compute", "device idle", "copies and communication"],
        "memory_scope": "Maximum GIFT total GPU memory estimate across timed iterations and ranks",
        "verified_receipt_fields": [
            "eight assigned GPU UUIDs, fresh worker PIDs and torchrun rendezvous IDs",
            "matched capture, source, compilation, candidate binaries and flags by route",
            "native NCCL gather count and per-rank pointer arithmetic",
            "native FP32 SUM reduce-scatter and retained sharded gradient",
            "zero timed allocator retries or OOMs",
        ],
        "groups": groups,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.raw)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Invalid matrix: {error}\n")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
