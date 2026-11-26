"""Launch fresh, counterbalanced eight-rank isolated-head pairs."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from manifests import ENV_FILES, dw_plans, expected_routes, pin, read_pinned
from summarise_grid import summarize


HERE = Path(__file__).resolve().parent
GRID = HERE.parents[1]
sys.path.insert(1, str(GRID))
import launch as legacy


REPEATS = (1, 2)
BASELINE_REVISION = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"


def schedule(cases, lengths):
    for index, seq in enumerate(legacy.LENGTHS):
        if seq not in lengths:
            continue
        for case_index, (case, (model, chunks)) in enumerate(legacy.CASES.items()):
            if case not in cases:
                continue
            for repeat in REPEATS:
                baseline_first = (index + case_index + repeat) % 2 == 1
                order = legacy.VARIANTS if baseline_first else legacy.VARIANTS[::-1]
                for position, variant in enumerate(order, start=1):
                    yield dict(case=case, model=model, chunks=chunks, seq=seq, batch=1,
                               pair=repeat, position=position, variant=variant)


def provenance(models, manifests):
    import torch

    result = legacy.provenance(models)
    root = legacy.ROOT
    files = (
        HERE / "run_grid.py", HERE / "rank_replay.py", HERE / "manifests.py",
        HERE / "summarise_grid.py", HERE / "run_detached.sh", root / "tuning/head_run.py",
        root / "tuning/proj/lt_ops.py", root / "tuning/proj/lt_bridge.so",
        *(source / "torchtitan/tools/utils.py" for source in (legacy.BASELINE_SOURCE, legacy.CANDIDATE_SOURCE)),
    )
    result["files_sha256"].update({str(path): legacy.sha256(path) for path in files})
    result["files_sha256"].update({entry["path"]: entry["sha256"] for entry in manifests.values()})
    revision = subprocess.check_output(["git", "-C", str(legacy.BASELINE_SOURCE), "rev-parse", "HEAD"],
                                       text=True, timeout=15).strip()
    if revision != BASELINE_REVISION:
        raise RuntimeError(f"Baseline A source revision changed: {revision}")
    result.update(baseline_revision=revision, nccl_version=torch.cuda.nccl.version())
    return result


def verify_outcome(attempt, code, request):
    status, oom_ranks, reason = legacy.outcome(attempt, code, request)
    if status not in legacy.TERMINAL:
        return status, oom_ranks, reason
    ranks = range(legacy.WORLD) if status == "success" else oom_ranks
    expected_gpus = legacy.read_json(attempt / "invocation.json")["settings"]["CUDA_VISIBLE_DEVICES"].split(",")
    for rank in ranks:
        record = legacy.read_json(attempt / f"resolved-{rank}.json")
        expected = {key: request[key] for key in ("model", "chunks", "seq", "batch", "variant", "manifests")}
        expected.update(rank=rank, local_rank=rank, world_size=legacy.WORLD, cuda_devices=expected_gpus)
        if record != expected:
            return "error", oom_ranks, f"Rank {rank} resolved different settings or manifest paths"
        if status == "success":
            source = legacy.BASELINE_SOURCE if request["variant"] == "baseline" else legacy.CANDIDATE_SOURCE
            expected_file = source / "torchtitan/components/loss.py"
            result = legacy.read_json(attempt / f"rank-{rank}.json")
            activation = legacy.read_json(attempt / f"activation-{rank}.json")
            if (Path(result["source"]).resolve() != expected_file.resolve()
                    or result["source_hash"] != request["loss_source_sha256"]
                    or activation.get("manifests", {}) != request["manifests"]):
                return "error", [], f"Rank {rank} executed a different source or set of plans"
    return status, oom_ranks, reason


def run_arm(series, request, sources, timeout):
    case, model, seq, chunks, variant = (request[key] for key in ("case", "model", "seq", "chunks", "variant"))
    arm = series / f"{case}-b1-s{seq}" / f"pair-{request['pair']:02d}" / variant
    if legacy.complete(arm, request, sources):
        return
    gpus = legacy.gpu_preflight()
    arm.mkdir(parents=True, exist_ok=True)
    next_attempt = max((int(path.name.split("-")[-1]) for path in arm.glob("attempt-*")), default=0) + 1
    attempt = arm / f"attempt-{next_attempt:03d}"
    attempt.mkdir()
    cmd = [
        sys.executable, "-Bu", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={legacy.WORLD}", "--max-restarts=0", "--log-dir", str(attempt / "elastic"),
        str(HERE / "rank_replay.py"), "--model", model, "--batch", "1", "--seq", str(seq),
        "--chunks", str(chunks), "--variant", variant,
        "--fixture", str(legacy.ROOT / "inputs" / f"{model}.pt"), "--out", str(attempt),
        "--warmup", str(request["warmup"]), "--samples", str(request["samples"]),
    ]
    settings = legacy.settings_for(model, variant, gpus)
    cache = HERE / "cache"
    settings.update(TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"),
                    CUDA_CACHE_PATH=str(cache / "cuda"))
    for name in ("LD_LIBRARY_PATH", "CUDA_HOME"):
        settings[name] = os.environ[name]
    unset = {
        "PYTORCH_CUDA_ALLOC_CONF", "GIFT_EXACT_FOLDED_CE_LIBRARY",
        "GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256", "HEAD_REPLAY_DHIDDEN_DIGEST", "CUDA_LAUNCH_BLOCKING",
        *(key for variable in ENV_FILES.values() for key in (variable, variable + "_SHA256")),
    }
    for name, variable in ENV_FILES.items():
        if variant != "baseline":
            selected = request["manifests"][name]
            settings[variable] = selected["path"]
            settings[variable + "_SHA256"] = selected["sha256"]
    env = {key: value for key, value in os.environ.items() if key not in unset}
    env.update(settings)
    legacy.write_json(attempt / "invocation.json", {
        **request, "cmd": cmd, "settings": settings, "unset": sorted(unset - settings.keys()),
        "provenance": sources, "gpu_preflight": gpus, "idle_memory_mib": legacy.IDLE_MEMORY_MIB,
        "timeout_seconds": timeout,
    })
    started = legacy.now()
    print(started, "start", attempt.relative_to(HERE), flush=True)
    code = None
    try:
        with (attempt / "run.log").open("w") as log:
            code, stopped = legacy.run_process(cmd, env, log, timeout)
        if stopped is None:
            status, oom_ranks, reason = verify_outcome(attempt, code, request)
        else:
            status, oom_ranks, reason = stopped, [], f"Replay {stopped} before a completed result"
    except (OSError, ValueError, KeyError) as error:
        status, oom_ranks, reason = "error", [], str(error)
    legacy.write_json(attempt / "status.json", dict(status=status, returncode=code, started_at=started,
                      ended_at=legacy.now(), oom_ranks=oom_ranks, reason=reason))
    print(legacy.now(), status, attempt.relative_to(HERE), flush=True)
    if status not in legacy.TERMINAL:
        raise RuntimeError(f"{reason}. Inspect {attempt / 'run.log'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="final")
    parser.add_argument("--cases", nargs="+", choices=legacy.CASES, default=tuple(legacy.CASES))
    parser.add_argument("--lengths", nargs="+", type=int, choices=legacy.LENGTHS, default=legacy.LENGTHS)
    for name in ("dw-c2", "dw-c8", "linear"):
        parser.add_argument(f"--{name}", nargs=2, metavar=("PATH", "SHA256"), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--plan-only", action="store_true", help="Validate the catalog inputs without CUDA or torchrun")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.series):
        parser.error("--series must be a single directory name")
    if min(args.warmup, args.samples, args.timeout) <= 0:
        parser.error("warmup, samples and timeout must be positive")
    manifests = {"dw_c2": pin(*args.dw_c2), "dw_c8": pin(*args.dw_c8), "linear": pin(*args.linear)}
    if manifests["dw_c2"]["path"] == manifests["dw_c8"]["path"]:
        parser.error("The C2 and C8 catalogs must have distinct paths")
    dw = {chunks: dw_plans(read_pinned(manifests[f"dw_c{chunks}"])) for chunks in (2, 8)}
    linear = read_pinned(manifests["linear"])
    requests = list(schedule(args.cases, args.lengths))
    for request in requests:
        chunks = request["chunks"]
        selected = {}
        if request["variant"] != "baseline":
            selected = {"dw": manifests[f"dw_c{chunks}"], "linear": manifests["linear"],
                        "ce": {"path": str(legacy.CE_LIBRARY), "sha256": legacy.CE_SHA256}}
            routes = expected_routes(request["model"], chunks, request["seq"], dw[chunks], linear)
            request["expected_routes"] = {"dw_case": routes["dw"].get("case") if routes["dw"] else None,
                                          "linear": routes["linear"]}
        request.update(manifests=selected, warmup=args.warmup, samples=args.samples)
    root = HERE / "results" / args.series
    if args.plan_only:
        print(json.dumps(requests, indent=2))
        return

    with (HERE / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = provenance({request["model"] for request in requests}, manifests)
        for request in requests:
            source = legacy.BASELINE_SOURCE if request["variant"] == "baseline" else legacy.CANDIDATE_SOURCE
            request["loss_source_sha256"] = sources["files_sha256"][str(source / "torchtitan/components/loss.py")]
        root.mkdir(parents=True, exist_ok=True)
        schedule_file = root / "schedule.json"
        if schedule_file.exists() and legacy.read_json(schedule_file) != requests:
            raise RuntimeError(f"{schedule_file} contains a different grid")
        legacy.write_json(schedule_file, requests)
        try:
            for request in requests:
                run_arm(root, request, sources, args.timeout)
        finally:
            summary = summarize(root, requests)
            legacy.write_json(HERE / f"{args.series}-summary.json", summary)
        if any(row["status"] != "complete" for row in summary["rows"]):
            raise RuntimeError("Some requested pairs did not complete")


if __name__ == "__main__":
    main()
