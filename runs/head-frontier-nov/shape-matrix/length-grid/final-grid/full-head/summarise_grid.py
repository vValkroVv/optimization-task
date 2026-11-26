"""Summarize rank-max wall time and native GIFT memory for each full-head pair."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


WORLD = 8
TERMINAL = {"success", "cuda_oom"}


def read(path):
    return json.loads(path.read_text())


def benchmark(attempt, request):
    candidate = request["variant"] == "cuda13"
    batches = []
    ranks = []
    activations = []
    for rank in range(WORLD):
        result = read(attempt / f"rank-{rank}.json")
        activation = read(attempt / f"activation-{rank}.json")
        for key in ("model", "batch", "seq", "chunks", "variant"):
            if result[key] != request[key]:
                raise ValueError(f"Rank {rank} has different {key} in {attempt}")
        expected_shape = [2, 1, request["seq"], 4096 if request["model"] == "100b" else 7168]
        if (result["rank"] != rank or result["hidden_shape"] != expected_shape
                or result["rows"] != 2 * request["seq"] // request["chunks"]
                or result["vocab"] != 128256 or result["physical_tokens"] != request["seq"]
                or result["staged_chunks"] != [candidate] * request["chunks"]
                or result["bf16_direct"] != candidate
                or result["gradient_divide_factor"] != 1.0 or not result["force_sum_reduction_for_comms"]
                or result["smoke_collectives"] != {"gather": 1, "scatter": 1}
                or len(result["measured"]) != request["samples"]):
            raise ValueError(f"Rank {rank} replay shape, path or sample count differs in {attempt}")
        route = "cuda13-consuming" if candidate else "compiled-aten-per-head"
        if activation["ce_route"] != route or activation.get("manifests", {}) != request["manifests"]:
            raise ValueError(f"Rank {rank} CE route or manifest selection differs in {attempt}")
        for step in result["measured"]:
            if (step["ms"] <= 0 or step["total_gib"] <= 0
                    or any(not math.isfinite(step[k]) or step[k] < 0
                           for k in ("ms", "submit_ms", "wait_ms", "total_gib"))):
                raise ValueError(f"Invalid measured timing or memory on rank {rank} in {attempt}")
        ranks.append(result)
        activations.append(activation)
        batches.append({key: result[key] for key in ("valid_tokens", "valid_mtp_tokens", "denominator")})
    if any(batch != batches[0] for batch in batches[1:]):
        raise ValueError(f"Ranks replayed different target shapes in {attempt}")
    if any(r["total_loss"] != ranks[0]["total_loss"] or r["components"] != ranks[0]["components"] for r in ranks):
        raise ValueError(f"Ranks produced different loss values in {attempt}")
    steps = list(zip(*(result["measured"] for result in ranks), strict=True))
    slowest_ranks = [max(range(WORLD), key=lambda rank: step[rank]["ms"]) for step in steps]
    wall = [step[rank] for step, rank in zip(steps, slowest_ranks, strict=True)]
    peak, peak_sample, peak_rank = max(
        (metrics["total_gib"], sample, rank)
        for sample, step in enumerate(steps, start=1) for rank, metrics in enumerate(step)
    )
    return dict(
        ms=median(step["ms"] for step in wall), steps_ms=[step["ms"] for step in wall],
        max_wall_ranks=slowest_ranks, submit_ms=median(step["submit_ms"] for step in wall),
        wait_ms=median(step["wait_ms"] for step in wall), total_gib=peak,
        max_memory_sample=peak_sample, max_memory_rank=peak_rank,
        allocator_retry_count_process=[r["allocator_retry_count_process"] for r in ranks],
        loss=ranks[0]["total_loss"], components=ranks[0]["components"], **batches[0],
        ce_route=activations[0]["ce_route"],
        routes=[{key: a[key] for key in ("counts", "plans", "linear_route", "linear_descriptors")}
                for a in activations] if candidate else [],
    )


def summarize(root, requests):
    rows = {}
    for request in requests:
        case, seq, pair, variant = (request[key] for key in ("case", "seq", "pair", "variant"))
        row = rows.setdefault((case, seq), {
            **{key: request[key] for key in ("case", "model", "chunks", "batch")},
            "sequence": seq, "folded_rows_per_chunk": 2 * seq // request["chunks"], "pairs": {},
        })
        entry = row["pairs"].setdefault(pair, {"repeat": pair, "order": []})
        entry["order"].append(variant)
        arm = root / f"{case}-b1-s{seq}" / f"pair-{pair:02d}" / variant
        attempts = []
        for path in sorted(arm.glob("attempt-*")):
            invocation = read(path / "invocation.json")
            if any(invocation[key] != value for key, value in request.items()):
                raise ValueError(f"Attempt used different configuration: {path}")
            status = read(path / "status.json") if (path / "status.json").exists() else {"status": "pending"}
            status["path"] = str(path.relative_to(root))
            if status["status"] == "cuda_oom":
                status["oom"] = [read(path / f"oom-{rank}.json") for rank in status["oom_ranks"]]
            attempts.append(status)
        current = attempts[-1] if attempts else {"status": "pending"}
        result = {"status": current["status"], "attempts": attempts, "manifests": request["manifests"]}
        if current["status"] == "success":
            result.update(benchmark(root / current["path"], request))
        entry[variant] = result

    for row in rows.values():
        pairs = list(row["pairs"].values())
        row["pairs"] = pairs
        row["status"] = "complete"
        for pair in pairs:
            baseline, tuned = pair["baseline"], pair["cuda13"]
            if not all(arm["status"] in TERMINAL for arm in (baseline, tuned)):
                row["status"] = "incomplete"
            if all(arm["status"] == "success" for arm in (baseline, tuned)):
                keys = ("valid_tokens", "valid_mtp_tokens", "denominator")
                if any(baseline[k] != tuned[k] for k in keys):
                    raise ValueError(f"Paired arms replayed different target dimensions in {row['case']}")
                pair["throughput_gain_pct"] = 100 * (baseline["ms"] / tuned["ms"] - 1)
                pair["savings_gib"] = baseline["total_gib"] - tuned["total_gib"]
        for variant, name in (("baseline", "baseline"), ("cuda13", "tuned")):
            arms = [pair[variant] for pair in pairs]
            statuses = {arm["status"] for arm in arms}
            if statuses == {"success"}:
                medians = [arm["ms"] for arm in arms]
                row[name] = {"status": "success", "ms": median(medians),
                             "total_gib": max(arm["total_gib"] for arm in arms),
                             "repeat_spread_pct": 100 * (max(medians) / min(medians) - 1)}
            else:
                row[name] = {"status": statuses.pop() if len(statuses) == 1 else "mixed"}
        gains = [pair["throughput_gain_pct"] for pair in pairs if "throughput_gain_pct" in pair]
        if len(gains) == len(pairs):
            row["throughput_gain_pct"] = median(gains)
            row["savings_gib"] = row["baseline"]["total_gib"] - row["tuned"]["total_gib"]
    return {"schema": 1, "rows": list(rows.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.series, read(args.series / "schedule.json")), indent=2))
