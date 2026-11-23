"""Validate the paired replay matrix and summarize the synchronized head timings."""

import argparse
import json
from pathlib import Path
from statistics import median


GIB = 1 << 30
ARMS = {
    "control": ("A", None),
    "65536/62720": ("B", 65536),
    "32768/95488": ("C", 32768),
}


def read(path):
    return json.loads(path.read_text())


def check_receipt(receipt, chunks, split, aliased):
    assert receipt["chunks"] == chunks and receipt["folded_rows_per_chunk"] == 16384 // chunks
    assert len(receipt["elapsed_ms"]) == 10
    assert receipt["warmup"] == 10 and len(receipt["memory"]) == 10
    assert receipt["panel"]["split"] == split and receipt["panel"]["active"] == (split is not None)
    assert receipt["panel"]["projection_calls_per_microbatch"] == chunks * (3 if split else 1)
    assert receipt["dw_writer"]["calls"] == (44 if split else 22) * chunks
    assert receipt["source"] == receipt["source_git"] == "2c67c182e8de6145f3796d62203714ab1b815446"
    assert receipt["singleton_reduction"] == "force-sum"
    assert receipt["ce_route_calls"] == {"folded": chunks, "ce": 0}
    assert all(receipt["activation"]["staged_chunk_calls"])
    assert not any(receipt["allocator_retries_and_ooms"].values())
    assert receipt["peak_total_gpu_gib"] == max(receipt["total_gpu_gib"])
    assert receipt["median_ms"] == median(receipt["elapsed_ms"])
    gather = receipt["gather_probe"]
    assert gather["gather_calls"] == gather["native_all_gather_into_tensor_calls"] == 23
    assert gather["distinct_bf16_gather_storages"] == (1 if aliased else 2)
    assert gather["master_to_bf16_casts"] == (23 if aliased else 1)


def summarize_arm(receipts):
    process_ms = [receipt["median_ms"] for receipt in receipts]
    return {
        "wall_ms": median(process_ms),
        "process_wall_ms": process_ms,
        "spread_ms": abs(process_ms[0] - process_ms[1]),
        "peak_total_gib": max(item["peak_total_gpu_gib"] for item in receipts),
        **{
            "peak_" + field + "_gib": max(
                sample[field + "_peak_bytes"] / GIB
                for item in receipts
                for sample in item["memory"]
            )
            for field in ("allocated", "reserved")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    base = args.base
    result = {"timing": {}, "native_control_digests": {}}
    for regime in ("regular", "alias"):
        for chunks in (2, 8):
            lane = base / f"{regime}-001" / f"chunks-{chunks}"
            assert (lane / "exit-code").read_text().strip() == "0"
            rows = {}
            devices, output_losses = set(), set()
            for name, (label, split) in ARMS.items():
                receipts = [read(lane / f"{label}{suffix}.json") for suffix in ("", "-repeat")]
                for receipt in receipts:
                    check_receipt(receipt, chunks, split, regime == "alias")
                    devices.add(receipt["device"]["visible"])
                    output_losses.add(tuple(receipt["loss"]))
                rows[name] = summarize_arm(receipts)
            assert len(devices) == len(output_losses) == 1
            control = rows["control"]
            for row in rows.values():
                row["delta_wall_ms"] = row["wall_ms"] - control["wall_ms"]
                row["delta_wall_pct"] = (row["wall_ms"] / control["wall_ms"] - 1) * 100
                row["delta_total_gib"] = row["peak_total_gib"] - control["peak_total_gib"]
                row["delta_total_pct"] = (row["peak_total_gib"] / control["peak_total_gib"] - 1) * 100
            result["timing"][f"{regime}-{chunks}"] = {"gpu": devices.pop(), "arms": rows}
    for chunks in (2, 8):
        measurements = read(base / "numerics-001" / f"chunks-{chunks}" / "verify.json")
        native = read(base / "native-control" / f"c{chunks}.digest.json")
        assert measurements["control"]["digests"] == native
        result["native_control_digests"][chunks] = "five digests matched"
        control = measurements["control"]
        for chunk in measurements["per_chunk"]:
            for checked_panels in chunk["split_checks"].values():
                for comparison in checked_panels.values():
                    assert all(panel["different_elements"] == 0 for panel in comparison.values())
        for row in measurements["candidates"].values():
            for name in ("loss_fp64", "dx_fp64", "dw_fp64"):
                assert row[name]["nonfinite"] == 0
            for name in ("loss", "dw", "z_loss", "z_dx"):
                assert row["digests"][name] == control["digests"][name]
            for name in ("dx_fp64", "dx_rounded_fp64"):
                assert row[name]["relative_l2"] < control[name]["relative_l2"]
            assert row["standalone_bf16_dx_vs_integrated"]["different_elements"] == 0
    (base / "verified-summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
