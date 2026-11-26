"""Summarize fresh full-head timing, memory and per-rank precision evidence."""
import argparse
import json
from pathlib import Path
from statistics import median


def read(run):
    if (run / "exit-code").read_text().strip() != "0":
        raise ValueError(f"Incomplete run {run}")
    launch = json.loads((run / "launch.json").read_text())
    ranks = [json.loads((run / f"rank-{i}.json").read_text()) for i in range(8)]
    active = [json.loads((run / f"activation-{i}.json").read_text()) for i in range(8)]
    for i, (row, receipt) in enumerate(zip(ranks, active, strict=True)):
        assert row["rank"] == i and row["smoke_collectives"] == {"gather": 1, "scatter": 1}
        assert row["bf16_direct"] and len(row["staged_chunks"]) == row["chunks"] and all(row["staged_chunks"])
        assert row["force_sum_reduction_for_comms"] and row["gradient_divide_factor"] == 1
        assert receipt["operations"] == launch["spec"]["operations"]
        assert receipt["ce"] == launch["settings"]["GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256"]
        if "linear" in receipt["operations"]:
            # The scheduled measurement must actually use the nominated GEMM on every chunk.
            spec = json.loads((Path(__file__).resolve().parent / "linear-plans.json").read_text())
            chosen_shapes = set()
            for key in spec:
                problem = json.loads(key)[2]
                m, n, k = problem[:3]
                chosen_shapes.add((m, k, n) if problem[10] else (n, k, m))
            args = launch["cmd"]
            invocations = 3 + int(args[args.index("--warmup") + 1]) + int(args[args.index("--samples") + 1])
            used = invocations * row["chunks"]
            z, h, v = row["rows"], row["width"], row["vocab"]
            expected = dict(projection=used if (z,h,v) in chosen_shapes else 0,
                            dhidden=used if (z,v,h) in chosen_shapes else 0)
            assert receipt["counts"] == expected and sum(expected.values()) > 0
            assert receipt["linear_route"] == {"candidate": sum(expected.values()), "fallback": 0}
        if "dw" in receipt["operations"]:
            assert all(p["pair"] != "native-generic" for p in receipt["plans"][0]["plans"])
    samples = list(zip(*(r["measured"] for r in ranks), strict=True))
    slowest = [max(s, key=lambda measurement: measurement["ms"]) for s in samples]
    return dict(run=run.parent.name, case=run.name, operations=active[0]["operations"],
                time_ms=median(s["ms"] for s in slowest), submit_ms=median(s["submit_ms"] for s in slowest),
                wait_ms=median(s["wait_ms"] for s in slowest),
                total_gib=max(row["total_gib"] for step in samples for row in step),
                max_rank_times=[s["ms"] for s in slowest]), ranks, active


def compare(ranks, ref, active, ref_active):
    differences = {}
    for i, (r, base) in enumerate(zip(ranks, ref, strict=True)):
        assert r["initialization"] == base["initialization"] and r["numerics"]["reference_cache"] == base["numerics"]["reference_cache"]
        for name in ("total", "loss_ce", "loss_mtp", "dhidden", "dweight_shard"):
            for metric in ("relative_l2", "max_abs"):
                diff = r["numerics"][name][metric] - base["numerics"][name][metric]
                if diff > 0:
                    differences.setdefault(name+"_"+metric, []).append([i, diff])
    return dict(dhidden_bitwise=all(r["dhidden_sha256"] == b["dhidden_sha256"] for r,b in zip(ranks,ref)),
                dw_bitwise=all(r["dweight_shard_sha256"] == b["dweight_shard_sha256"] for r,b in zip(active,ref_active)),
                losses_equal=all(r["components"] == b["components"] and r["total_loss"] == b["total_loss"] for r,b in zip(ranks,ref)),
                fp64_increases=differences)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args=p.parse_args()
    rows=[]
    for case in sorted({d.name for d in args.root.glob("*/*") if (d / "exit-code").exists()}):
        paths=sorted(args.root.glob("*/"+case))
        done=[p for p in paths if (p / "exit-code").exists() and (p / "exit-code").read_text().strip()=="0"]
        results=[read(p) for p in done]
        if not results:
            continue
        base, ref, ref_active=next(rr for rr in results if not rr[0]["operations"])
        for cell,ranks,active in results:
            cell.update(compare(ranks,ref,active,ref_active))
            cell["time_reduction_vs_first_control_pct"]=100*(1-cell["time_ms"]/base["time_ms"])
            rows.append(cell)
    print(json.dumps(rows, indent=2))
