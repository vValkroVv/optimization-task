"""Verify and summarize six isolated eight-rank LM-head measurements."""

import argparse
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path
import re
from statistics import median


CHUNKS = (2, 8, 32)
ARMS = ("baseline", "final")
RANKS = tuple(range(8))
WARMUP = SAMPLES = 10
ITERATIONS = list(range(WARMUP + 1, WARMUP + SAMPLES + 1))
INVOCATIONS = WARMUP + SAMPLES + 3
BASELINE_SOURCE = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
FIXTURE_SHA = "15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727"
VOCAB, WIDTH, BATCH, SEQ_LEN = 128256, 4096, 2, 4096
TOKENS_PER_ITERATION = len(RANKS) * BATCH * SEQ_LEN
PAIRS = {2: "9.14", 8: "5.5", 32: "5.5"}
SCHEDULE = (
    "c2-baseline", "c2-final", "c8-final", "c8-baseline", "c32-baseline", "c32-final",
)
SCOPE = (
    "Host wall including native target packing, isolated head and z-loss forward/backward, "
    "collective submission and completion, and CUDA synchronization. Excludes start barrier, "
    "memory reads, and reset-output digest."
)
REDUCTION = "Eight-rank NCCL FP32 SUM reduce-scatter with gradient divide factor 1"
ACCUMULATION = "Primed sharded weight gradient retained across timed calls"
SWITCHES = (
    "GIFT_STABLE_CE_FOLDED_BACKWARD",
    "GIFT_DIRECT_BF16_DHIDDEN",
    "GIFT_STAGED_HEAD_DW",
    "GIFT_FUSED_HIDDEN_Z_LOSS",
)
DIGESTS = {
    "loss": ([], "torch.float32"),
    "dx": ([2, 2, 4096, 4096], "torch.bfloat16"),
    "dw": ([16032, 4096], "torch.float32"),
    "z_loss": ([], "torch.float32"),
    "z_dx": ([2, 4096, 4096], "torch.bfloat16"),
}
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cell_name(chunks, arm):
    return f"c{chunks}-{arm}"


def require_sha(value, field):
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None,
            f"{field}: expected a lowercase SHA256")


def verify_series(values, field, *, positive=False):
    require(isinstance(values, list) and len(values) == SAMPLES and all(
        type(value) in (int, float) and isfinite(value)
        and (value > 0 if positive else value >= 0) for value in values
    ), f"{field}: expected {SAMPLES} finite {'positive' if positive else 'nonnegative'} samples")


def verify_digests(digests):
    require(set(digests) == set(DIGESTS),
            "expected loss, dX, dW, z-loss and z-dX tensor digests")
    for name, (shape, dtype) in DIGESTS.items():
        digest = digests[name]
        require((digest["shape"], digest["dtype"]) == (shape, dtype),
                f"{name}: tensor shape or dtype differs from the expected full tensor/local shard")
        require_sha(digest["sha256"], f"{name} digest")


def verify_compile(receipt):
    compile_config = receipt["compile"]
    require(compile_config["backend"] == "inductor"
            and compile_config["scope"] == "native loss"
            and compile_config["counters"]["unique_graphs"] > 0
            and compile_config["cuda_graphs"] is False,
            f"unexpected native-loss compile settings: {compile_config}")


def verify_activation(receipt, chunks, arm):
    candidate = arm == "final"
    require(receipt["folded_rows_per_chunk"] == 16384 // chunks,
            "unexpected active folded rows per chunk")
    require(receipt["capture_rank"] == 0 and receipt["fixture_sha256"] == FIXTURE_SHA,
            "unexpected production-tokenizer capture identity")
    activation = receipt["activation"]
    for field in ("folded", "direct_bf16_dhidden", "early_logits_release", "fused_hidden_z_loss"):
        require(activation[field] is candidate, f"{field} does not match the planned {arm} arm")
    ce = (
        "torchtitan.experiments.four_fusion.stable_ce.cross_entropy_loss"
        if candidate else "torchtitan.components.loss.cross_entropy_loss"
    )
    require(activation["ce"] == ce, f"unexpected compiled cross-entropy callable: {activation['ce']}")
    require(activation["staged_chunk_calls"] == [candidate] * chunks,
            "unexpected staged-head chunk activation")
    require(receipt["ce_route_calls"] == {
        "native": 0 if candidate else 2 * chunks,
        "exact_setup": chunks if candidate else 0,
    }, "unexpected native or exact CE route count")
    exact_calls = INVOCATIONS * chunks if candidate else 0
    require(activation["exact_ce_calls"] == exact_calls,
            "unexpected total exact CE calls")

    expected_flags = {
        **{name: str(int(candidate)) for name in SWITCHES},
        "GIFT_EXPERIMENT_LM_HEAD_MATRIX": "new" if candidate else "",
        "GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS": str(chunks) if candidate else "",
    }
    require(activation["gift_flags"] == expected_flags, "unexpected GIFT activation flags")
    writer = activation["dw_writer"]
    exact = receipt["candidate"]
    if candidate:
        require(writer["name"] == "cublasLtMatmul" and writer["pair"] == PAIRS[chunks],
                f"unexpected staged dW writer: {writer}")
        require(type(writer["version"]) is int and writer["version"] > 0,
                "missing cuBLASLt writer version")
        require(exact["method"] == "cuda-13" and exact["calls"] == exact_calls,
                f"unexpected exact CE route: {exact}")
        # The replay records the main and MTP upstream scalars once, regardless
        # of how many chunk-specific dW writer calls ran.
        scales = exact["incoming_fp32_scales"]
        require(isinstance(scales, list) and len(scales) == 2 and all(
            type(scale) in (int, float) and isfinite(scale) and scale > 0 for scale in scales
        ), "expected the two finite positive main/MTP incoming FP32 scales")
        for field in ("binary_sha256", "hook_sha256", "launcher_sha256"):
            require_sha(exact[field], f"candidate {field}")
    else:
        require(writer is None and exact is None, "baseline used a candidate dW or CE path")


def verify_gather(gather, rank):
    expected_calls = INVOCATIONS + 1  # One priming gather precedes replay calls.
    require(gather["mode"] == "stack"
            and gather["group_ranks"] == list(RANKS)
            and (gather["group_rank"], gather["group_size"]) == (rank, len(RANKS)),
            "unexpected regular GFSDP all-gather mode or NCCL group")
    require(gather["gather_calls"] == gather["native_all_gather_into_tensor_calls"]
            == expected_calls, "wrong number of native NCCL all-gather calls")
    casts = gather["master_to_bf16_casts"]
    require(0 < casts <= expected_calls, "wrong number of FP32 master-to-BF16 gather casts")
    input_bytes = (VOCAB // len(RANKS)) * WIDTH * 2
    offset = rank * input_bytes
    require(gather["input_bytes"] == input_bytes
            and gather["output_bytes"] == input_bytes * len(RANKS)
            and gather["rank_slot_offset_bytes"] == offset,
            "wrong BF16 gather shard extent")
    input_ptr, output_ptr = gather["input_data_ptr"], gather["output_data_ptr"]
    input_storage, output_storage = gather["input_storage_ptr"], gather["output_storage_ptr"]
    require(min(input_ptr, output_ptr, input_storage, output_storage) > 0,
            "missing BF16 gather buffer pointers")
    require(input_storage != output_storage and gather["distinct_bf16_gather_storages"] == 2,
            "regular gather must use distinct BF16 input and output storages")
    require(gather["input_offset_bytes"] is None
            and (input_ptr + input_bytes <= output_ptr
                 or output_ptr + gather["output_bytes"] <= input_ptr),
            "regular gather input and output overlap")


def verify_rank(receipt, rank, chunks, arm):
    expected_arm = "regular" if arm == "final" else "baseline"
    require(receipt["rank"] == rank and receipt["world_size"] == len(RANKS),
            f"expected rank {rank} of {len(RANKS)}")
    require(receipt["chunks"] == chunks and receipt["arm"] == expected_arm,
            f"expected chunks={chunks}, arm={expected_arm}")
    require((receipt["warmup"], receipt["samples"]) == (WARMUP, SAMPLES),
            f"expected {WARMUP} warmups and {SAMPLES} samples")
    require(receipt["group_ranks"] == list(RANKS), "expected collective group of ranks 0–7")
    require(receipt["sample_iterations"] == ITERATIONS,
            "timed iterations must be aligned at 11–20")
    require(type(receipt["process_id"]) is int and receipt["process_id"] > 0,
            "expected a worker PID")
    require(isinstance(receipt["torchelastic_run_id"], str) and receipt["torchelastic_run_id"],
            "expected a torchrun rendezvous ID")
    for field, positive in (("elapsed_ms", True), ("total_gpu_gib", True),
                            ("host_submit_ms", False), ("host_wait_ms", False)):
        verify_series(receipt[field], field, positive=positive)
    require(all(isclose(elapsed, submit + wait, rel_tol=1e-9, abs_tol=1e-7)
                for elapsed, submit, wait in zip(
                    receipt["elapsed_ms"], receipt["host_submit_ms"], receipt["host_wait_ms"],
                    strict=True,
                )), "host submission plus synchronization wait must equal elapsed time")
    require(receipt["scope"] == SCOPE, "unexpected integrated head timing scope")
    require(receipt["allocator_retries_and_ooms"] == {"num_alloc_retries": 0, "num_ooms": 0},
            f"allocator retries or OOMs: {receipt['allocator_retries_and_ooms']}")
    require(receipt["expandable_segments"] is True,
            "both arms must use expandable CUDA allocator segments")
    require(receipt["native_fp32_sum_reduce_scatter_calls"] == INVOCATIONS
            and receipt["reduction"] == REDUCTION,
            "wrong native FP32 SUM reduce-scatter route")
    require(receipt["accumulation"] == ACCUMULATION,
            "expected primed sharded gradient accumulation across timed calls")
    verify_compile(receipt)
    verify_gather(receipt["gather_probe"], rank)
    verify_activation(receipt, chunks, arm)
    verify_digests(receipt["digests"])

    source = receipt["source"]
    require(isinstance(source, str) and GIT_SHA.fullmatch(source) is not None,
            "expected a 40-character source revision")
    if arm == "baseline":
        require(source == BASELINE_SOURCE, f"baseline source must be {BASELINE_SOURCE}")
    source_hashes = receipt["source_hashes"]
    require(isinstance(source_hashes, dict) and source_hashes,
            "missing source file fingerprints")
    for name, digest in source_hashes.items():
        require(isinstance(name, str) and name, "invalid source fingerprint path")
        require_sha(digest, f"source fingerprint {name}")
    for field in ("harness_sha256", "historical_replay_sha256", "isolated_pair_harness_sha256"):
        require_sha(receipt[field], field)


def verify_manifest(root):
    manifest = read(root / "manifest.json")
    require(isinstance(manifest.get("host"), str) and manifest["host"],
            "manifest must identify the SR008 host")
    devices = manifest["devices"]
    require(isinstance(devices, list) and len(devices) == len(RANKS),
            "manifest must record all eight assigned GPUs")
    require([int(device[0]) for device in devices] == list(RANKS),
            "manifest GPU indices must be 0–7 in rank order")
    uuids = [device[1] for device in devices]
    require(len(set(uuids)) == len(RANKS) and all(
        isinstance(uuid, str) and uuid.startswith("GPU-") for uuid in uuids
    ), "manifest must contain eight unique GPU UUIDs")
    require(all("H100" in device[2] for device in devices),
            "manifest devices must be H100 GPUs")
    require(manifest["schedule"] == list(SCHEDULE),
            "manifest schedule must contain each planned cell in launch order")
    require(isinstance(manifest.get("topology"), str) and manifest["topology"].strip(),
            "manifest must include the GPU topology")
    return manifest


def verify_invocation(directory, chunks, arm):
    require((directory / "exit-code").read_text().strip() == "0",
            f"{directory}: group launcher must exit successfully")
    command = read(directory / "invocation.json")
    require(isinstance(command, list) and all(isinstance(item, str) for item in command),
            f"{directory}: invocation must retain the actual argv list")
    require(any(command[index:index + 2] == ["-m", "torch.distributed.run"]
                for index in range(len(command) - 1)),
            f"{directory}: expected torch.distributed.run invocation")
    for flag in ("--standalone", "--nnodes=1", "--nproc-per-node=8", "--max-restarts=0"):
        require(flag in command, f"{directory}: invocation is missing {flag}")
    scripts = [index for index, part in enumerate(command) if Path(part).name == "replay_pair.py"]
    require(len(scripts) == 1, f"{directory}: expected the isolated replay_pair.py harness")
    tail = command[scripts[0] + 1:]
    expected_arm = "regular" if arm == "final" else "baseline"
    expected = ["--chunks", str(chunks), "--arm", expected_arm]
    require(tail[:len(expected)] == expected,
            f"{directory}: invocation uses the wrong chunk count or arm")
    rest = tail[len(expected):]
    require(len(rest) == 6 and rest[0] == "--out"
            and Path(rest[1]).resolve() == directory.resolve()
            and rest[2:] == ["--warmup", "10", "--samples", "10"],
            f"{directory}: invocation must use this output directory and a 10+10 window")
    paths = [directory / f"rank-{rank}.json" for rank in RANKS]
    require(set(directory.glob("rank-*.json")) == set(paths),
            f"{directory}: expected exactly eight rank receipts")


def read_cell(root, manifest, chunks, arm):
    name = cell_name(chunks, arm)
    directory = root / name
    verify_invocation(directory, chunks, arm)
    devices = manifest["devices"]
    uuids = [row[1] for row in devices]
    receipts = []
    for rank in RANKS:
        path = directory / f"rank-{rank}.json"
        receipt = read(path)
        try:
            verify_rank(receipt, rank, chunks, arm)
            device = receipt["device"]
            require(device["visible_uuid"] == uuids[rank]
                    and device["visible"] == ",".join(uuids)
                    and device["rank"] == device["current_device"] == rank
                    and device["visible_uuid"].startswith("GPU-")
                    and "H100" in device["name"],
                    "worker GPU UUID or rank differs from the campaign manifest")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}: {error}") from error
        receipts.append(receipt)
    return receipts


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
    effective_tps = [TOKENS_PER_ITERATION / (elapsed / 1000) for elapsed in max_time]
    return {
        "group_median_ms": median(max_time),
        "iteration_max_rank_elapsed_ms": max_time,
        "iteration_slowest_rank": slowest,
        "iteration_effective_head_input_tps": effective_tps,
        "median_effective_head_input_tps": median(effective_tps),
        "tokens_per_iteration": TOKENS_PER_ITERATION,
        "peak_gift_total_gib": peak,
        "peak_rank": rank,
        "peak_iteration": WARMUP + sample + 1,
        "per_rank_elapsed_ms": {str(row["rank"]): row["elapsed_ms"] for row in receipts},
        "per_rank_total_gpu_gib": {str(row["rank"]): row["total_gpu_gib"] for row in receipts},
        "per_rank_host_submit_ms": {str(row["rank"]): row["host_submit_ms"] for row in receipts},
        "per_rank_host_wait_ms": {str(row["rank"]): row["host_wait_ms"] for row in receipts},
    }


def stable_candidate(candidate):
    if candidate is None:
        return None
    return {
        key: value for key, value in candidate.items()
        if key not in ("incoming_fp32_scales", "calls")
    }


def verify_matched_controls(cells):
    all_rows = [row for receipts in cells.values() for row in receipts]
    processes = [row["process_id"] for row in all_rows]
    require(len(set(processes)) == len(processes), "expected a fresh worker process per cell and rank")
    run_ids = set()
    for name, rows in cells.items():
        ids = {row["torchelastic_run_id"] for row in rows}
        require(len(ids) == 1 and run_ids.isdisjoint(ids),
                f"{name}: reused or mismatched torchrun rendezvous IDs")
        run_ids.update(ids)

    baseline = cells[cell_name(CHUNKS[0], "baseline")]
    reference_by_rank = baseline
    shared_fields = (
        "fixture", "fixture_sha256", "hidden_z_fixture_sha256", "device",
        "harness_sha256", "historical_replay_sha256", "isolated_pair_harness_sha256",
    )
    for name, rows in cells.items():
        for rank, row in enumerate(rows):
            reference = reference_by_rank[rank]
            for field in shared_fields:
                require(row[field] == reference[field], f"{name}, rank {rank}: {field} is not matched")
        arm = "baseline" if name.endswith("baseline") else "final"
        for rank, row in enumerate(rows):
            source_rows = [cells[cell_name(chunks, arm)][rank] for chunks in CHUNKS]
            for field in ("source", "source_hashes"):
                require(row[field] == source_rows[0][field],
                        f"{name}, rank {rank}: {field} differs across matched {arm} cells")
            require(row["device"]["torch"] == reference_by_rank[rank]["device"]["torch"]
                    and row["device"]["torch_git"] == reference_by_rank[rank]["device"]["torch_git"]
                    and row["device"]["cuda"] == reference_by_rank[rank]["device"]["cuda"],
                    f"{name}, rank {rank}: runtime differs from matched baseline")

    per_arm_chunks = {
        arm: [cells[cell_name(chunks, arm)] for chunks in CHUNKS]
        for arm in ARMS
    }
    for arm, groups in per_arm_chunks.items():
        for rank in RANKS:
            first = groups[0][rank]
            for row_group in groups[1:]:
                row = row_group[rank]
                require(row["expandable_segments"] == first["expandable_segments"],
                        f"rank {rank}: allocator setting differs across {arm} chunk sizes")
                require(stable_candidate(row["candidate"]) == stable_candidate(first["candidate"]),
                        f"rank {rank}: candidate binaries or route differ across {arm} chunk sizes")


def digest_comparisons(cells):
    results = {}
    for chunks in CHUNKS:
        baseline = cells[cell_name(chunks, "baseline")]
        final = cells[cell_name(chunks, "final")]
        mismatches = {name: [] for name in sorted(DIGESTS)}
        for rank in RANKS:
            for name in sorted(DIGESTS):
                if baseline[rank]["digests"][name] != final[rank]["digests"][name]:
                    mismatches[name].append(rank)
        results[f"c{chunks}"] = {
            "compared_rank_pairs": len(RANKS),
            "all_equal": not any(mismatches.values()),
            "mismatches_by_tensor": mismatches,
        }
    return results


def comparison(baseline, final):
    reference_times = baseline["iteration_max_rank_elapsed_ms"]
    final_times = final["iteration_max_rank_elapsed_ms"]
    deltas = [right - left for left, right in zip(reference_times, final_times, strict=True)]
    require(all(base > 0 for base in reference_times), "baseline elapsed time must be positive")
    return {
        "baseline_group_median_ms": baseline["group_median_ms"],
        "final_group_median_ms": final["group_median_ms"],
        "final_minus_baseline_median_ms": final["group_median_ms"] - baseline["group_median_ms"],
        "final_minus_baseline_median_pct": (
            100 * (final["group_median_ms"] - baseline["group_median_ms"])
            / baseline["group_median_ms"]
        ),
        "aligned_iteration_elapsed_delta_ms": deltas,
        "peak_gift_total_delta_gib": (
            final["peak_gift_total_gib"] - baseline["peak_gift_total_gib"]
        ),
        "baseline_peak_gift_total_gib": baseline["peak_gift_total_gib"],
        "final_peak_gift_total_gib": final["peak_gift_total_gib"],
        "baseline_median_effective_head_input_tps": baseline["median_effective_head_input_tps"],
        "final_median_effective_head_input_tps": final["median_effective_head_input_tps"],
        "final_minus_baseline_median_effective_tps_pct": (
            100 * (final["median_effective_head_input_tps"]
                   - baseline["median_effective_head_input_tps"])
            / baseline["median_effective_head_input_tps"]
        ),
    }


def summarize(root):
    manifest = verify_manifest(root)
    cells = {
        cell_name(chunks, arm): read_cell(root, manifest, chunks, arm)
        for chunks in CHUNKS for arm in ARMS
    }
    verify_matched_controls(cells)
    groups = {name: group_stats(receipts) for name, receipts in cells.items()}
    results = {
        f"c{chunks}": {
            "baseline": groups[cell_name(chunks, "baseline")],
            "final": groups[cell_name(chunks, "final")],
            "final_minus_baseline": comparison(
                groups[cell_name(chunks, "baseline")], groups[cell_name(chunks, "final")]
            ),
        }
        for chunks in CHUNKS
    }
    return {
        "raw_directory": str(root),
        "host": manifest["host"],
        "devices": manifest["devices"],
        "topology": manifest["topology"],
        "launch_order": manifest["schedule"],
        "warmup_iterations": WARMUP,
        "measured_iterations": SAMPLES,
        "ranks": len(RANKS),
        "latency_scope": SCOPE,
        "latency_aggregation": (
            "For each aligned measured iteration 11–20, take the maximum elapsed wall time "
            "across eight ranks; report the median of those ten maxima."
        ),
        "memory_scope": (
            "Maximum GIFT total GPU memory estimate across all eight ranks and timed iterations 11–20."
        ),
        "throughput_scope": (
            f"Effective head-only physical input TPS uses {TOKENS_PER_ITERATION} tokens per replay "
            "(8 ranks × local batch 2 × sequence length 4096), divided by each iteration's "
            "maximum rank elapsed wall time. It is a component metric, not full-training TPS."
        ),
        "verified_receipt_fields": [
            "same eight manifest H100 UUIDs, rank mapping, source per arm, and fresh worker processes",
            "matched real production-tokenizer CE and hidden-z fixtures",
            "all activation flags, CE routes, cuBLASLt plan, and CUDA-13 call count",
            "native regular NCCL all-gather and native FP32 SUM reduce-scatter routes",
            "loss, dX, dW, z-loss, and z-dX tensor shape and dtype",
            "10 warmup and 10 measured iterations, compilation scope, wall timing scope, and memory scope",
            "zero allocator retries and OOMs with expandable segments enabled in both arms",
        ],
        "cells": cells,
        "groups": groups,
        "results": results,
        "digest_comparisons": digest_comparisons(cells),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_directory", type=Path)
    parser.add_argument("summary_json", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.campaign_directory)
        args.summary_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Invalid isolated EP8 campaign: {error}\n")
    for chunks in CHUNKS:
        row = result["results"][f"c{chunks}"]
        delta = row["final_minus_baseline"]
        print(
            f"C{chunks}: baseline={delta['baseline_group_median_ms']:.3f} ms, "
            f"final={delta['final_group_median_ms']:.3f} ms, "
            f"delta={delta['final_minus_baseline_median_ms']:+.3f} ms "
            f"({delta['final_minus_baseline_median_pct']:+.2f}%), "
            f"peak GIFT total={delta['baseline_peak_gift_total_gib']:.3f}/"
            f"{delta['final_peak_gift_total_gib']:.3f} GiB, "
            f"effective head TPS={delta['baseline_median_effective_head_input_tps']:.0f}/"
            f"{delta['final_median_effective_head_input_tps']:.0f}"
        )


if __name__ == "__main__":
    main()
