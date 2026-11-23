"""Verify the repeated same-H100 stack and control receipts."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys


SOURCE = "2c67c182e8de6145f3796d62203714ab1b815446"


def read(path):
    return json.loads(path.read_text())


def verify(base):
    rows = []
    for chunks in (2, 4, 8, 16):
        lane = base / f"chunks-{chunks}"
        assert (lane / "exit-code").read_text().strip() == "0"
        order, hashes, devices, config, verified_arms = [], [], set(), set(), defaultdict(list)
        for path in sorted(lane.glob("*.json")):
            if path.name.endswith(".digest.json"):
                continue
            receipt = read(path)
            arm = receipt["stack_arm"]
            tuned = arm == "stack"
            order.append(arm)
            devices.add(receipt["device"]["visible"])
            config.add((receipt["fixture"], receipt["hidden_z_fixture_sha256"], receipt["device"]["torch"], receipt["device"]["cuda"]))
            assert receipt["source"] == SOURCE and receipt["chunks"] == chunks
            assert receipt["variant"] == "current" and receipt["folded_rows_per_chunk"] == 16384 // chunks
            assert receipt["warmup"] == 10 and receipt["profile"] is False
            assert len(receipt["elapsed_ms"]) == len(receipt["total_gpu_gib"]) == 10
            assert receipt["median_ms"] == median(receipt["elapsed_ms"])
            assert receipt["peak_total_gpu_gib"] == max(receipt["total_gpu_gib"])
            assert receipt["compile"]["backend"] == "inductor" and receipt["compile"]["cuda_graphs"] is False
            assert receipt["compile"]["counters"]["unique_graphs"] > 0
            assert receipt["early_release_gift_loop"] == tuned
            assert receipt["logit_storage"] == ("reuse" if tuned else "native")
            assert receipt["activation"]["staged_chunk_calls"] == [True] * chunks
            assert receipt["activation"]["folded"] and receipt["ce_route_calls"] == {"folded": chunks, "ce": 0}
            assert receipt["singleton_reduction"] == "force-sum"
            assert all(math.isfinite(loss) for loss in receipt["loss"])
            assert not any(receipt["allocator_retries_and_ooms"].values())
            writer = receipt["dw_writer"]
            assert writer["calls"] == 23 * chunks
            if tuned:
                plan = base.parent.parent / "frontier-dw" / f"tune-{chunks}.json"
                assert writer["tuning_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
                assert writer["cublaslt_version"] == read(plan)["cublasLt_version"]
                assert writer["engine"] == "cublasLtMatmul"
                assert len(writer["algorithm_pair"]) == 2
            else:
                assert writer["engine"] == "cublasGemmEx"
            digest = read(path.with_suffix(".digest.json"))
            assert set(digest) == {"loss", "dw", "dx", "z_loss", "z_dx"}
            assert digest["dw"]["dtype"] == "torch.float32" and digest["dw"]["shape"] == [128256, 4096]
            assert digest["dx"]["dtype"] == "torch.bfloat16"
            hashes.append(digest)
            verified_arms[arm].append(receipt)
        assert len(devices) == len(config) == 1 and len(hashes) == 4 and all(h == hashes[0] for h in hashes)
        assert sorted(order) == ["control", "control", "stack", "stack"]
        assert order[0] == order[-1] and order[1] == order[2]
        row = {"chunks": chunks, "gpu_uuid": next(iter(devices)), "order": order, "identical_output_digests": hashes[0]}
        for arm, receipts in verified_arms.items():
            medians = [item["median_ms"] for item in receipts]
            row[arm] = {
                "wall_ms": median(medians),
                "medians_ms": medians,
                "repeat_spread_ms": max(medians) - min(medians),
                "total_gib": max(item["peak_total_gpu_gib"] for item in receipts),
            }
        row["wall_change_pct"] = 100 * (row["stack"]["wall_ms"] / row["control"]["wall_ms"] - 1)
        row["total_change_gib"] = row["stack"]["total_gib"] - row["control"]["total_gib"]
        rows.append(row)
    return rows


if __name__ == "__main__":
    base = Path(sys.argv[1])
    result = json.dumps(verify(base), indent=2) + "\n"
    (base / "verified-summary.json").write_text(result)
