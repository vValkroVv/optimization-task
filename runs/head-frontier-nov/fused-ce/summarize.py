"""Verify and summarize saved isolated LM-head replay receipts."""

import argparse
import json
from pathlib import Path
from statistics import median


GIB = 2 ** 30


def read(path):
    return json.loads(path.read_text())


def summarize(series):
    paths = sorted(p for p in series.glob("[0-9]-*.json") if not p.name.endswith(".digest.json"))
    assert (series / "exit-code").read_text().strip() == "0", series
    receipts = [read(p) for p in paths]
    arms = [r["arm"] for r in receipts]
    assert len(arms) >= 4 and len(arms) % 2 == 0
    for i in range(0, len(arms), 2):
        assert set(arms[i:i + 2]) == {"control", "fused"}
    assert len(set(r["device"]["visible"] for r in receipts)) == 1
    assert len(set(r["source_loss_sha256"] for r in receipts)) == 1
    chunks = receipts[0]["chunks"]
    gather = receipts[0]["gather_mode"]
    for receipt in receipts:
        assert receipt["chunks"] == chunks
        assert receipt["gather_mode"] == gather
        assert receipt["dw_writer"]["engine"] == "cublasLtMatmul"
        assert receipt["expandable_segments"]
        assert len(receipt["elapsed_ms"]) == len(receipt["gpu_allocator_samples"]) == 10
        assert receipt["median_ms"] == median(receipt["elapsed_ms"])
        assert "inductor" == receipt["compile"]["backend"]
        assert not receipt["compile"]["cuda_graphs"]
        assert not any(receipt["allocator_retries_and_ooms"].values())
        if receipt["arm"] == "fused":
            assert receipt["candidate"]["calls"] == 23 * chunks
            assert receipt["candidate"]["input_output_alias"]
        if gather == "alias":
            probe = receipt["gather_probe"]
            assert probe["gather_calls"] == probe["native_all_gather_into_tensor_calls"] == 24
            assert probe["input_data_ptr"] == probe["output_data_ptr"]
    digests = [read(p.with_suffix(".digest.json")) for p in paths]
    parity = {k: len({p[k]["sha256"] for p in digests}) == 1 for k in digests[0]}
    arm_stats = {}
    for arm in ("control", "fused"):
        rows = [r for r in receipts if r["arm"] == arm]
        medians = [r["median_ms"] for r in rows]
        samples = [sample for r in rows for sample in r["gpu_allocator_samples"]]
        arm_stats[arm] = {
            "median_ms": median(medians),
            "process_medians_ms": medians,
            "process_median_spread_ms": max(medians) - min(medians),
            "peak_allocated_gib": max(sample["allocated_bytes"] for sample in samples) / GIB,
            "peak_reserved_gib": max(sample["reserved_bytes"] for sample in samples) / GIB,
            "peak_gift_total_gib": max(row["peak_total_gpu_gib"] for row in rows),
        }
    paired_delta = []
    for first, second in zip(receipts[::2], receipts[1::2], strict=True):
        if first["arm"] == "fused":
            first, second = second, first
        paired_delta.append(second["median_ms"] - first["median_ms"])
    delta = arm_stats["fused"]["median_ms"] - arm_stats["control"]["median_ms"]
    return {
        "series": series.name, "chunks": chunks, "gather": gather, "arms": arms,
        **arm_stats, "wall_change_ms": delta,
        "wall_change_pct": 100 * delta / arm_stats["control"]["median_ms"],
        "paired_fused_minus_control_ms": paired_delta,
        "complete_output_bitwise": parity,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    series = sorted(p for p in args.raw_directory.iterdir() if p.is_dir())
    results = [summarize(p) for p in series]
    assert all(all(r["complete_output_bitwise"].values()) for r in results if "v1" not in r["series"])
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
