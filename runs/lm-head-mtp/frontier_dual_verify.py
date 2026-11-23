"""Verify both ranks and summarize the alternating n8 head replay."""

import hashlib
import json
from pathlib import Path
from statistics import median
import sys


GIB = 2**30
ORDER = ("0-stack", "1-alias", "2-stack", "3-alias")
NAMES = {"loss", "dx", "dw", "z_loss", "z_dx"}
SHARD_BYTES = 64128 * 4096 * 2


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder):
    assert (folder / "exit-code").read_text().strip() == "0"
    rows = []
    cases_by_rank = {0: [], 1: []}
    for case in ORDER:
        alias = case.endswith("-alias")
        ranks = []
        for rank in (0, 1):
            path = folder / f"{case}.rank-{rank}.json"
            result = json.loads(path.read_text())
            digest = json.loads(path.with_suffix(".digest.json").read_text())
            assert set(digest) == NAMES
            assert digest["dw"]["shape"] == [64128, 4096] and digest["dw"]["dtype"] == "torch.float32"
            device = result["device"]
            assert device["rank"] == device["current_device"] == rank
            assert device["visible_uuid"] == device["visible"].split(",")[rank]
            assert result["stack_arm"] == "stack"
            assert result["chunks"] == 8 and result["warmup"] == 10
            assert result["activation"]["staged_chunk_calls"] == [True] * 8
            assert result["activation"]["folded"] and result["ce_route_calls"] == {"folded": 8, "ce": 0}
            assert result["dw_writer"]["engine"] == "cublasLtMatmul" and result["dw_writer"]["calls"] == 23 * 8
            assert result["dw_writer"]["tuning_sha256"] == file_hash(folder.parent.parent / "frontier-dw/tune-8.json")
            assert result["lifetime"] == "combined" and result["early_release_gift_loop"]
            assert result["logit_storage"] == "reuse" and not result["profile"]
            assert result["compile"]["backend"] == "inductor" and not result["compile"]["cuda_graphs"]
            assert not any(result["allocator_retries_and_ooms"].values())
            assert result["native_fp32_sum_reduce_scatter_calls"] == 23
            probe = result["gather_probe"]
            assert probe["group_rank"] == rank and probe["group_size"] == 2
            assert probe["mode"] == ("alias" if alias else "stack")
            assert probe["native_all_gather_into_tensor_calls"] == probe["gather_calls"] == 24
            assert probe["master_to_bf16_casts"] == (24 if alias else 1)
            assert probe["input_bytes"] == SHARD_BYTES and probe["output_bytes"] == 2 * SHARD_BYTES
            assert probe["distinct_bf16_gather_storages"] == (1 if alias else 2)
            assert probe["lazy_init_allocation_delta_bytes"] == (0 if alias else SHARD_BYTES)
            assert (probe["input_storage_ptr"] == probe["output_storage_ptr"]) == alias
            if alias:
                assert probe["input_data_ptr"] - probe["output_data_ptr"] == rank * SHARD_BYTES
            assert len(result["elapsed_ms"]) == len(probe["memory"]) == 10
            assert result["median_ms"] == median(result["elapsed_ms"])
            assert result["peak_total_gpu_gib"] == max(result["total_gpu_gib"])
            memory = probe["memory"]
            ranks.append({
                "rank": rank,
                "receipt": path.name,
                "elapsed_ms": result["elapsed_ms"],
                "median_ms": result["median_ms"],
                "peak_allocated_bytes": max(item["allocated_peak_bytes"] for item in memory),
                "peak_reserved_bytes": max(item["reserved_peak_bytes"] for item in memory),
                "peak_device_estimate_gib": result["peak_total_gpu_gib"],
            })
            identity = (
                result["source"], result["fixture"], result["hidden_z_fixture_sha256"],
                result["compile"], result["activation"], result["dw_writer"], result["stack_sha256"],
                result["harness_sha256"], result["gather_script_sha256"], result["dual_script_sha256"],
                device, probe["nccl_version"],
            )
            cases_by_rank[rank].append((digest, identity))
        row_time = median(max(pair) for pair in zip(*(value["elapsed_ms"] for value in ranks), strict=True))
        rows.append({
            "case": case,
            "median_max_rank_ms": row_time,
            "max_rank_peak_allocated_gib": max(value["peak_allocated_bytes"] for value in ranks) / GIB,
            "max_rank_peak_reserved_gib": max(value["peak_reserved_bytes"] for value in ranks) / GIB,
            "max_rank_peak_device_estimate_gib": max(value["peak_device_estimate_gib"] for value in ranks),
            "ranks": ranks,
        })
    assert all(all(entry == cases[0] for entry in cases) for cases in cases_by_rank.values())
    assert cases_by_rank[0][0][0]["dw"] != cases_by_rank[1][0][0]["dw"]
    group = [(rows[0], rows[1]), (rows[2], rows[3])]
    differences = [
        {
            "control": control["case"],
            "alias": alias["case"],
            "median_max_rank_ms_pct": 100 * (alias["median_max_rank_ms"] / control["median_max_rank_ms"] - 1),
            **{
                key: alias[key] - control[key]
                for key in ("max_rank_peak_allocated_gib", "max_rank_peak_reserved_gib", "max_rank_peak_device_estimate_gib")
            },
        }
        for control, alias in group
    ]
    return {
        "source_revision": cases_by_rank[0][0][1][0],
        "scope": "Isolated 8-chunk captured main/MTP head and independent z-loss. Both ranks replay the same captured inputs with disjoint weight shards; the group performs real BF16 NCCL all-gather and FP32 SUM reduce-scatter.",
        "timing": "Synchronized host wall per invocation, rank-wise sample maximum then median. Separate device compute, host work, waits and communication durations are unavailable.",
        "benchmark": rows,
        "paired_alias_minus_stack": differences,
        "repeated_stack_max_rank_ms_range": abs(rows[0]["median_max_rank_ms"] - rows[2]["median_max_rank_ms"]),
        "identical_output_digests_per_rank": {rank: cases[0][0] for rank, cases in cases_by_rank.items()},
        "gather_output_bytes": 2 * SHARD_BYTES,
        "gather_input_bytes_eliminated_per_rank": SHARD_BYTES,
        "gather_calls_per_rank_per_arm": 24,
        "fp32_sum_reduce_scatter_calls_per_rank_per_arm": 23,
        "harness_sha256": {
            name: file_hash(folder.parent.parent / name)
            for name in ("frontier_dual.py", "frontier_dual_lane.sh", "frontier_dual_verify.py")
        },
    }


if __name__ == "__main__":
    (folder,) = map(Path, sys.argv[1:])
    summary = verify(folder)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
