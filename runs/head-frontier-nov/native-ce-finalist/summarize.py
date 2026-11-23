"""Verify native paired CE receipts and report wall and GIFT memory results."""

import argparse
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


def group_stats(receipts):
    medians = [r["median_ms"] for r in receipts]
    return {
        "wall_ms": median(medians),
        "per_process_wall_ms": medians,
        "process_spread_ms": max(medians) - min(medians),
        "peak_gift_total_gib": max(r["peak_total_gpu_gib"] for r in receipts),
        "host_submit_ms": median(sample for r in receipts for sample in r["host_submit_ms"]),
        "host_wait_ms": median(sample for r in receipts for sample in r["host_wait_ms"]),
    }


def series_summary(directory, chunks, method, order, historical, *, baseline="control", trial="fused"):
    assert (directory / "exit-code").read_text().strip() == "0"
    paths = [directory / f"{index}-{arm}.json" for index, arm in enumerate(order)]
    receipts = [read(path) for path in paths]
    digests = [read(path.with_suffix(".digest.json")) for path in paths]
    assert set(digests[0]) == {"loss", "dx", "dw", "z_loss", "z_dx"}
    assert all(digest == digests[0] for digest in digests), directory
    assert digests[0] == read(historical), directory
    first = receipts[0]
    for receipt, spec in zip(receipts, order, strict=True):
        arm = "control" if spec == "control" else "fused"
        assert receipt["arm"] == arm and receipt["chunks"] == chunks
        assert receipt["source"] == first["source"]
        assert receipt["source_hashes"] == first["source_hashes"]
        assert receipt["device"] == first["device"]
        assert receipt["fixture"] == first["fixture"]
        assert receipt["hidden_z_fixture_sha256"] == first["hidden_z_fixture_sha256"]
        assert receipt["expandable_segments"]
        assert receipt["ce_route_calls"] == {"folded_inplace": chunks}
        assert receipt["writer"]["pair"] == {2: "9.14", 8: "5.5"}[chunks]
        assert receipt["gather_mode"] == "regular"
        gather = receipt["gather"]
        assert gather["gather_calls"] == gather["native_all_gather_into_tensor_calls"] == 24
        assert gather["distinct_bf16_gather_storages"] == 2
        assert receipt["compile"]["backend"] == "inductor"
        assert receipt["compile"]["counters"]["unique_graphs"] == 1
        assert not receipt["compile"]["cuda_graphs"]
        assert not any(receipt["allocator_retries_and_ooms"].values())
        assert len(receipt["elapsed_ms"]) == len(receipt["memory"]) == receipt["warmup"] == 10
        assert median(receipt["elapsed_ms"]) == receipt["median_ms"]
        assert abs(sum(receipt["host_submit_ms"]) + sum(receipt["host_wait_ms"])
                   - sum(receipt["elapsed_ms"])) < 1e-7
        assert max(receipt["total_gpu_gib"]) == receipt["peak_total_gpu_gib"]
        if arm == "fused":
            candidate = receipt["candidate"]
            assert candidate["method"] == (method if spec == "fused" else spec)
            assert candidate["calls"] == 23 * chunks
            assert candidate["incoming_fp32_scales"] == candidate["prepared_fp32_scales"]
    groups = {
        label: [r for r, spec in zip(receipts, order, strict=True) if spec == label]
        for label in (baseline, trial)
    }
    control, fused = (group_stats(groups[label]) for label in groups)
    pairs = []
    for index in range(0, len(order), 2):
        assert set(order[index:index + 2]) == {baseline, trial}
        left, right = receipts[index:index + 2]
        control_member, fused_member = (left, right) if order[index] == baseline else (right, left)
        pairs.append(fused_member["median_ms"] - control_member["median_ms"])
    wall_delta = fused["wall_ms"] - control["wall_ms"]
    variants = {}
    for label, rows in groups.items():
        if label != "control":
            variants[label] = rows[0]["candidate"]
            assert all(r["candidate"] == variants[label] for r in rows)
    return {
        "chunks": chunks,
        "method": method,
        "arms": order,
        "device": first["device"],
        "source": first["source"],
        "source_hashes": first["source_hashes"],
        "candidate": variants[trial],
        "variants": variants,
        "fixture": first["fixture"],
        "hidden_z_fixture_sha256": first["hidden_z_fixture_sha256"],
        baseline: control,
        trial: fused,
        "wall_change_ms": wall_delta,
        "wall_change_pct": 100 * wall_delta / control["wall_ms"],
        f"adjacent_pairs_{trial}_minus_{baseline}_ms": pairs,
        "gift_total_change_gib": fused["peak_gift_total_gib"] - control["peak_gift_total_gib"],
        "external_gib": sorted({m["external_gib"] for r in receipts for m in r["memory"]}),
        "digest_matches_historical_control": str(historical.relative_to(historical.parents[3])),
        "bitwise_equal_digests": digests[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    historical = Path(__file__).resolve().parent.parent / "fused-ce" / "raw"
    results = {}
    for chunks, historical_control, order in (
        (2, "0-control", ["control", "fused", "fused", "control", "control", "fused"]),
        (8, "1-control", ["fused", "control", "control", "fused", "fused", "control"]),
    ):
        reference = historical / f"regular-c{chunks}-exact" / f"{historical_control}.digest.json"
        exact = results[f"cuda-13-c{chunks}"] = series_summary(
            args.raw / f"native-c{chunks}", chunks, "cuda-13", order, reference
        )
        packed = results[f"cuda-16-c{chunks}"] = series_summary(
            args.raw / f"native-opt-c{chunks}", chunks, "cuda-16",
            ["control", "fused", "fused", "control"], reference,
        )
        first, second = ("cuda-13", "cuda-16") if chunks == 2 else ("cuda-16", "cuda-13")
        interleaved = results[f"direct-c{chunks}"] = series_summary(
            args.raw / f"native-direct-c{chunks}", chunks, "cuda-16",
            [first, second, second, first, first, second],
            reference, baseline="cuda-13", trial="cuda-16",
        )
        for field in ("device", "source", "source_hashes", "fixture", "hidden_z_fixture_sha256"):
            assert exact[field] == packed[field] == interleaved[field], (chunks, field)
        assert exact["candidate"] == interleaved["variants"]["cuda-13"]
        assert packed["candidate"] == interleaved["variants"]["cuda-16"]
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
