"""Run the isolated head grid with the final per-cell plan policy."""

import argparse
import fcntl
import json
from pathlib import Path
import re

from manifests import dw_plans, expected_routes, pin, read_pinned
from run_grid import HERE, legacy, provenance, run_arm, schedule
from summarise_grid import read, summarize


def load_policy(path):
    policy = read(path)
    if policy["format"] != 1:
        raise ValueError("Expected final policy format 1")

    def resolve(entry):
        return pin(path.parent / entry["path"], entry["sha256"])

    defaults = policy["defaults"]
    linear_default = resolve(defaults["linear"])
    dw_catalogs = {int(chunks): resolve(entry) for chunks, entry in defaults["dw"].items()}
    linear_plans = read_pinned(linear_default)
    overrides = {}
    for model, by_chunks in policy["cells"].items():
        for chunks, by_seq in by_chunks.items():
            for seq, override in by_seq.items():
                key = (model, int(chunks), int(seq))
                manifest = resolve(override["linear"])
                expected = dict(linear_plans)
                del expected[override["remove_exact_descriptor"]]
                if read_pinned(manifest) != expected:
                    raise ValueError(f"Cell {key} changed more than its selected linear descriptor")
                overrides[key] = manifest
    return dw_catalogs, linear_default, overrides


def requests_for(args, dw_catalogs, linear_default, overrides):
    dw = {chunks: dw_plans(read_pinned(entry)) for chunks, entry in dw_catalogs.items()}
    paths = (linear_default, *overrides.values())
    linear = {entry["path"]: read_pinned(entry) for entry in paths}
    requests = list(schedule(args.cases, args.lengths))
    for req in requests:
        req.update(manifests={}, warmup=10, samples=10)
        if req["variant"] == "baseline":
            continue
        model, chunks, seq = (req[key] for key in ("model", "chunks", "seq"))
        choice = overrides.get((model, chunks, seq), linear_default)
        route = expected_routes(model, chunks, seq, dw[chunks], linear[choice["path"]])
        req["manifests"] = {"linear": choice, "dw": dw_catalogs[chunks],
                            "ce": {"path": str(legacy.CE_LIBRARY), "sha256": legacy.CE_SHA256}}
        req["expected_routes"] = {"linear": route["linear"],
                                  "dw_case": route["dw"].get("case") if route["dw"] else None}
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=HERE / "final-policy.json")
    parser.add_argument("--series", default="final-policy")
    parser.add_argument("--cases", nargs="+", choices=legacy.CASES, default=tuple(legacy.CASES))
    parser.add_argument("--lengths", nargs="+", type=int, choices=legacy.LENGTHS, default=legacy.LENGTHS)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.series):
        parser.error("--series must be a single directory name")
    policy_path = args.policy.resolve(strict=True)
    dw, linear, overrides = load_policy(policy_path)
    requests = requests_for(args, dw, linear, overrides)
    if args.plan_only:
        print(json.dumps(requests, indent=2))
        return

    root = HERE / "results" / args.series
    with (HERE / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = {"linear": linear, **{f"dw_c{c}": entry for c, entry in dw.items()}}
        files.update({f"override_{key}": entry for key, entry in overrides.items()})
        sources = provenance({request["model"] for request in requests}, files)
        helpers = (Path(__file__).resolve(), policy_path, HERE / "run_final_detached.sh")
        sources["files_sha256"].update({str(p): legacy.sha256(p) for p in helpers})
        for req in requests:
            source = legacy.BASELINE_SOURCE if req["variant"] == "baseline" else legacy.CANDIDATE_SOURCE
            req["loss_source_sha256"] = sources["files_sha256"][str(source / "torchtitan/components/loss.py")]
        root.mkdir(parents=True, exist_ok=True)
        schedule_file = root / "schedule.json"
        if schedule_file.is_file() and read(schedule_file) != requests:
            raise RuntimeError(f"{schedule_file} contains a different final grid")
        legacy.write_json(schedule_file, requests)
        try:
            for req in requests:
                run_arm(root, req, sources, timeout=1200)
        finally:
            result = summarize(root, requests)
            legacy.write_json(HERE / f"{args.series}-summary.json", result)
        if any(row["status"] != "complete" for row in result["rows"]):
            raise RuntimeError("Some requested pairs did not complete")


if __name__ == "__main__":
    main()
