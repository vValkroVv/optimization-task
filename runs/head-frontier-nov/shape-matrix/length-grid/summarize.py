"""Aggregate completed eight-rank isolated head measurements by local sequence length."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

CASES = [("100b", 2), ("100b", 8), ("700b", 8)]
LENGTHS = tuple(4096 << i for i in range(7))


def read(path):
    return json.loads(path.read_text())


def cell(directory, model, chunks, seq, variant):
    attempts = sorted(directory.glob("attempt-*/status.json"))
    if not attempts:
        return {"status": "pending"}
    path = attempts[-1].parent
    outcome = read(path / "status.json")
    launch = read(path / "invocation.json")
    spec = (launch["model"], launch["chunks"], launch["batch"], launch["seq"], launch["variant"])
    assert spec == (model, chunks, 1, seq, variant), spec
    prov = launch["provenance"]
    row = dict(status=outcome["status"], attempt=path.name, host=prov["host"],
               fixture_sha256=prov["fixtures_sha256"][model])
    if row["status"] == "cuda_oom":
        assert outcome["oom_ranks"], f"Missing OOM evidence in {path}"
        row["oom_ranks"] = outcome["oom_ranks"]
        row["oom_example"] = read(path / f"oom-{outcome['oom_ranks'][0]}.json")
        assert "cuda out of memory" in row["oom_example"]["error"].lower()
    if row["status"] != "success":
        return row

    ranks = [read(path / f"rank-{i}.json") for i in range(8)]
    for i, rank in enumerate(ranks):
        assert rank["rank"] == i and rank["variant"] == variant and rank["model"] == model
        assert rank["chunks"] == chunks and rank["rows"] == 2 * seq // chunks
        assert rank["vocab"] == 128256 and rank["width"] == (4096 if model == "100b" else 7168)
        assert rank["gradient_divide_factor"] == 1.0 and rank["force_sum_reduction_for_comms"]
        assert rank["smoke_collectives"] == {"gather": 1, "scatter": 1}
        assert len(rank["staged_chunks"]) == chunks
        assert all(x == (variant == "cuda13") for x in rank["staged_chunks"])
        assert math.isfinite(rank["total_loss"]) and rank["bf16_direct"] == (variant == "cuda13")
    assert all(r["batch"] == 1 and r["seq"] == seq and r["physical_tokens"] == seq for r in ranks)
    assert all(r["hidden_shape"] == [2, 1, seq, ranks[0]["width"]] for r in ranks)
    activations = [read(path / f"activation-{i}.json") for i in range(8)]
    assert all(a["ce_route"] == activations[0]["ce_route"] for a in activations)
    row["ce_route"] = activations[0]["ce_route"]
    if variant == "cuda13":
        assert row["ce_route"] == "cuda13-consuming"
        applied = []
        for a in activations:
            plan = a["plans"][0]
            applied.append(dict(dw="catalog" if all(plan["tuned_calls_beta0_beta1"]) else "generic",
                                projection=a["counts"]["projection"], dhidden=a["counts"]["dhidden"],
                                linear_fallback=a["linear_route"]["descriptor_fallback"]))
        assert all(item == applied[0] for item in applied)
        row["routes"] = applied[0]
    row["main_valid"] = ranks[0]["valid_tokens"]
    row["mtp_valid"] = ranks[0]["valid_mtp_tokens"]
    row["components"] = ranks[0]["components"]
    assert all(r["total_loss"] == ranks[0]["total_loss"] and r["components"] == row["components"] for r in ranks)
    steps = list(zip(*(r["measured"] for r in ranks), strict=True))
    assert len(steps) == launch["samples"] and all(s["ms"] > 0 and math.isfinite(s["ms"]) and s["total_gib"] > 0 for step in steps for s in step)
    wall = [max(step, key=lambda r:r["ms"]) for step in steps]
    row.update(ms=median(r["ms"] for r in wall), total_gib=max(r["total_gib"] for step in steps for r in step),
               submit_ms=median(r["submit_ms"] for r in wall), wait_ms=median(r["wait_ms"] for r in wall),
               steps_ms=[r["ms"] for r in wall], loss=ranks[0]["total_loss"], samples=len(wall))
    return row


def aggregate(root):
    rows = []
    for model, chunks in CASES:
        for seq in LENGTHS:
            directory = root / f"{model}-c{chunks}-b1-s{seq}"
            a, b = [cell(directory / v, model, chunks, seq, v) for v in ("baseline", "cuda13")]
            row = dict(model=model, chunks=chunks, sequence=seq, batch=1, folded_rows_per_chunk=2 * seq // chunks,
                       baseline=a, tuned=b)
            if "host" in a and "host" in b:
                assert a["host"] == b["host"] and a["fixture_sha256"] == b["fixture_sha256"]
            if a["status"] == b["status"] == "success":
                assert (a["main_valid"],a["mtp_valid"]) == (b["main_valid"],b["mtp_valid"])
                row["component_deltas"] = {key:b["components"][key] - a["components"][key] for key in a["components"]}
                row.update(throughput_gain_pct=100 * (a["ms"] / b["ms"] - 1),
                           savings_gib=a["total_gib"] - b["total_gib"], loss_delta=b["loss"]-a["loss"])
            rows.append(row)
    return rows


def display(x):
    if x["status"] == "success":
        return f"{x['ms']:.2f} / {x['total_gib']:.3f}"
    if x["status"] == "cuda_oom":
        return "CUDA OOM"
    return x["status"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    rows = aggregate(args.results)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    for model, chunks in CASES:
        print(f"\n### {model.upper()} C{chunks}\n")
        print("| Local sequence | Baseline A + compile, ms / GiB | Tuned CE + cuBLASLt, ms / GiB | Head throughput gain |")
        print("|---:|---:|---:|---:|")
        for r in rows:
            if (model,chunks) == (r["model"],r["chunks"]):
                gain = f"{r['throughput_gain_pct']:+.1f}%" if "throughput_gain_pct" in r else "—"
                print(f"| {r['sequence']:,} | {display(r['baseline'])} | {display(r['tuned'])} | {gain} |")
