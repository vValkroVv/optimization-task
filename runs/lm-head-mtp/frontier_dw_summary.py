"""Verify paired replay receipts and summarize component measurements."""

from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys


ARTIFACTS = Path(sys.argv[1])
PAIRS = {2: "9.14", 4: "13.13", 8: "5.5", 16: "5.5"}
LANES = {2: "replay-g5-001", 4: "replay-g5-001", 8: "replay-g4-001", 16: "replay-g4-001"}


def aggregate(rows):
    names = ("stream_ms", "submit_ms", "synchronized_wall_ms")
    return {name: statistics.median(row[name] for row in rows) for name in names}


def row(chunks, pair, lane):
    tune = json.loads((ARTIFACTS / f"tune-{chunks}.json").read_text())
    assert tune["chunks"] == chunks and tune["rows_per_chunk"] == 16384 // chunks
    baseline = tune["fp64"]["cublas"]
    assert baseline == tune["fp64"][pair] == tune["fp64"]["cublas_same"]
    assert baseline["nonfinite"] == 0
    rows = defaultdict(list)
    numerics = {}
    for path in sorted((ARTIFACTS / lane).glob(f"{chunks}-*.json")):
        data = json.loads(path.read_text())
        if path.name.endswith(".numerics.json"):
            writer = "lt" if "-lt-num." in path.name else "cublas"
            numerics[writer] = data
            continue
        writer = data["writer"]
        rows[writer].append(data)
        flags = data["activation"]
        assert data["chunks"] == chunks and data["singleton_reduction"] == "force-sum"
        assert data["compile"]["backend"] == "inductor" and not data["compile"]["cuda_graphs"]
        assert flags["folded"] and all(flags["staged_chunk_calls"]) and len(flags["staged_chunk_calls"]) == chunks
        assert all(flags[name] == "1" for name in flags if name.startswith("GIFT_"))
        assert data["ce_route_calls"] == {"ce": 0, "folded": chunks}
        assert not any(data["allocator_retries_and_ooms"].values())
        assert data["dw_writer_calls"] == (23 if "-num." in path.name else 22) * chunks
        assert data["device"]["visible"] == tune["gpu"]
        if writer == "lt":
            assert data["dw_cublasLt_algorithms"] == [tune["configurations"][int(i)] for i in pair.split(".")]
    assert set(rows) == set(numerics) == {"cublas", "lt"}
    assert numerics["cublas"] == numerics["lt"]
    assert numerics["cublas"]["dw_fp64"] == baseline
    assert all(len(values) == 2 for values in rows.values())
    all_rows = [item for arm in rows.values() for item in arm]
    assert len({(d["source"], d["device"]["visible"], d["fixture"], d["hidden_z_fixture_sha256"]) for d in all_rows}) == 1
    assert len({tuple(d["loss"]) for d in all_rows}) == 1
    timings = {
        writer: {
            "median_wall_ms": statistics.median(d["median_ms"] for d in data),
            "replicate_medians_ms": [d["median_ms"] for d in data],
            "peak_total_gpu_gib": max(d["peak_total_gpu_gib"] for d in data),
        }
        for writer, data in rows.items()
    }
    return {
        "chunks": chunks, "folded_rows": tune["rows_per_chunk"],
        "device": tune["gpu"], "plan": pair, "fp64": baseline,
        "tuning": {label: aggregate(tune["chain"][key])
                   for label, key in zip(("cublas", "cublas_same", "lt"), ("cublas", "cublas_same", pair), strict=True)},
        "replay": timings, "replay_numerics": numerics["lt"],
    }


if __name__ == "__main__":
    output = [row(chunks, pair, LANES[chunks]) for chunks, pair in PAIRS.items()]
    print(json.dumps(output, indent=2))
