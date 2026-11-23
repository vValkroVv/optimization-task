"""Verify and aggregate fixed-count gather alias comparisons."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import sys


GIB = 2**30
HEAD_BYTES = 128256 * 4096 * 2


def verify(lanes):
    samples = defaultdict(list)
    devices, revisions, harnesses = set(), set(), set()
    for lane in lanes:
        assert (lane / "exit-code").read_text().strip() == "0", lane
        paths = [path for path in lane.glob("*.json") if path.name.split("-")[0].isdigit()
                 and not path.name.endswith(".digest.json")]
        for path in sorted(paths, key=lambda item: int(item.name.split("-")[0])):
            result = json.loads(path.read_text())
            probe = result["gather_probe"]
            chunks = result["chunks"]
            arm = probe["mode"]
            alias = arm == "alias"
            assert path.stem.split("-")[1:] == [f"n{chunks}", arm]
            assert result["stack_arm"] == "stack"
            assert result["warmup"] == len(result["elapsed_ms"]) == len(probe["memory"]) == 10
            assert result["median_ms"] == median(result["elapsed_ms"])
            assert result["peak_total_gpu_gib"] == max(result["total_gpu_gib"])
            assert result["lifetime"] == "combined" and result["early_release_gift_loop"]
            assert result["logit_storage"] == "reuse" and not result["profile"]
            assert result["activation"]["staged_chunk_calls"] == [True] * chunks
            assert result["activation"]["folded"] and result["ce_route_calls"] == {"folded": chunks, "ce": 0}
            assert result["singleton_reduction"] == "force-sum"
            assert result["compile"]["backend"] == "inductor" and not result["compile"]["cuda_graphs"]
            assert not any(result["allocator_retries_and_ooms"].values())
            writer = result["dw_writer"]
            assert writer["calls"] == 23 * chunks and writer["engine"] == "cublasLtMatmul"
            tuning = lane.parent.parent / "frontier-dw" / f"tune-{chunks}.json"
            assert writer["tuning_sha256"] == hashlib.sha256(tuning.read_bytes()).hexdigest()
            assert probe["group_rank"] == 0 and probe["group_size"] == 1
            assert probe["distinct_bf16_gather_storages"] == (1 if alias else 2)
            assert probe["lazy_init_allocation_delta_bytes"] == (0 if alias else HEAD_BYTES)
            assert (probe["input_data_ptr"] == probe["output_data_ptr"]) == alias
            assert (probe["input_storage_ptr"] == probe["output_storage_ptr"]) == alias
            assert probe["input_bytes"] == probe["output_bytes"] == HEAD_BYTES
            calls = probe["native_all_gather_into_tensor_calls"]
            assert calls == probe["gather_calls"] == 24
            assert probe["master_to_bf16_casts"] == (calls if alias else 1)
            digest = json.loads(path.with_suffix(".digest.json").read_text())
            assert set(digest) == {"loss", "dw", "dx", "z_loss", "z_dx"}
            devices.add(tuple(result["device"].values()))
            revisions.add((result["source"], result["fixture"], result["hidden_z_fixture_sha256"]))
            harnesses.add((result["stack_sha256"], result["harness_sha256"], result["gather_script_sha256"]))
            samples[chunks].append({
                "arm": arm,
                "receipt": str(path),
                "median_ms": result["median_ms"],
                "total_gib": result["peak_total_gpu_gib"],
                "allocated_peak_gib": max(item["allocated_peak_bytes"] for item in probe["memory"]) / GIB,
                "reserved_peak_gib": max(item["reserved_peak_bytes"] for item in probe["memory"]) / GIB,
                "digest": digest,
            })
    assert len(devices) == len(revisions) == len(harnesses) == 1
    rows = []
    for chunks, group in sorted(samples.items()):
        digest = group[0]["digest"]
        assert all(item.pop("digest") == digest for item in group)
        order = [item["arm"] for item in group]
        assert len(order) == 4 and order[0] == order[3] and order[1] == order[2] and order[0] != order[1]
        row = {"chunks": chunks, "order": order, "samples": group, "identical_output_digests": digest}
        for arm in ("stack", "alias"):
            selected = [item for item in group if item["arm"] == arm]
            times = [item["median_ms"] for item in selected]
            row[arm] = {"wall_ms": median(times), "repeat_spread_ms": max(times) - min(times)}
            for memory in ("total_gib", "allocated_peak_gib", "reserved_peak_gib"):
                row[arm][memory] = max(item[memory] for item in selected)
        row["wall_change_pct"] = 100 * (row["alias"]["wall_ms"] / row["stack"]["wall_ms"] - 1)
        row["allocated_change_gib"] = row["alias"]["allocated_peak_gib"] - row["stack"]["allocated_peak_gib"]
        assert row["allocated_change_gib"] == -HEAD_BYTES / GIB
        rows.append(row)
    return rows


if __name__ == "__main__":
    output, *lanes = map(Path, sys.argv[1:])
    output.write_text(json.dumps(verify(lanes), indent=2) + "\n")
