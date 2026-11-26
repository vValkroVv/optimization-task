"""Repeat pinned final-grid head arms in a dedicated profile directory."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

PHASE = Path(__file__).resolve().parent
FULL_HEAD = PHASE.parents[1] / "full-head"
sys.path.insert(0, str(FULL_HEAD))
from run_final_grid import load_policy, requests_for
from run_grid import legacy, verify_outcome


def launch(request, series, samples, profile_sample, swap):
    seq = request["seq"]
    variant = request["variant"]
    run = f"100b-c8-b1-s{seq}/pair-{request['pair']:02d}/{variant}/attempt-001"
    ref_series = "corrected-100b-c8-128k" if seq == 131072 else "final"
    reference = FULL_HEAD / "results" / ref_series / run / "invocation.json"
    prior = legacy.read_json(reference)
    for key, value in request.items():
        if prior[key] != value:
            raise RuntimeError(f"Final-policy request differs from pinned invocation at {key}: {reference}")
    attempt = series / run
    attempt.mkdir(parents=True)
    gpus = legacy.gpu_preflight()
    for field in ("uuid", "driver"):
        if [gpu[field] for gpu in gpus] != [gpu[field] for gpu in prior["gpu_preflight"]]:
            raise RuntimeError(f"GPU topology no longer matches the pinned grid at {field}")
    for key in ("LD_LIBRARY_PATH", "CUDA_HOME"):
        if os.environ[key] != prior["settings"][key]:
            raise RuntimeError(f"Loaded {key} differs from the pinned launch")

    by_index = {gpu["index"]: gpu["uuid"] for gpu in gpus}
    order = sorted(by_index)
    if swap:
        first, second = swap
        order[first], order[second] = order[second], order[first]

    commands = list(prior["cmd"])
    if commands[0] != sys.executable:
        raise RuntimeError(f"Expected {commands[0]}, got {sys.executable}")
    commands[commands.index(str(FULL_HEAD / "rank_replay.py"))] = str(PHASE / "profile_rank.py")
    commands[commands.index("--log-dir") + 1] = str(attempt / "elastic")
    commands[commands.index("--out") + 1] = str(attempt)
    commands[commands.index("--samples") + 1] = str(samples)
    cache = PHASE / "cache"
    settings = dict(prior["settings"],
                    CUDA_VISIBLE_DEVICES=",".join(by_index[index] for index in order),
                    TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                    TRITON_CACHE_DIR=str(cache / "triton"),
                    CUDA_CACHE_PATH=str(cache / "cuda"),
                    C8_PHASE_PROFILE_SAMPLE=str(profile_sample))
    env = {key: value for key, value in os.environ.items() if key not in prior["unset"]}
    env.update(settings)
    diagnostic_files = [Path(__file__), PHASE / "profile_rank.py", PHASE / "run_profile.sh"]
    invocation = {
        **prior, "cmd": commands, "settings": settings, "gpu_preflight": gpus, "samples": samples,
        "diagnostic_sources": {str(path): legacy.sha256(path) for path in diagnostic_files},
        "reference": str(reference), "reference_sha256": legacy.sha256(reference),
        "physical_gpu_index_by_logical_rank": order,
    }
    legacy.write_json(attempt / "invocation.json", invocation)
    started = legacy.now()
    print(started, "start", attempt.relative_to(PHASE), flush=True)
    with (attempt / "run.log").open("w") as log:
        code, stopped = legacy.run_process(commands, env, log, prior["timeout_seconds"])
    if stopped is None:
        status, oom_ranks, reason = verify_outcome(attempt, code, invocation)
    else:
        status, oom_ranks, reason = stopped, [], f"Replay {stopped}"
    artifacts = [f"timing-{rank}.json" for rank in range(legacy.WORLD)]
    if profile_sample:
        artifacts.extend(f"trace-{rank}.json" for rank in range(legacy.WORLD))
    if status == "success" and any(not (attempt / name).is_file() for name in artifacts):
        status, reason = "error", "Missing timing or profiler artifact"
    ended = legacy.now()
    legacy.write_json(attempt / "status.json", dict(status=status, returncode=code, started_at=started,
                       ended_at=ended, oom_ranks=oom_ranks, reason=reason,
                       trace_sha256={name: legacy.sha256(attempt / name) for name in artifacts
                                     if name.startswith("trace-") and (attempt / name).is_file()}))
    print(ended, status, attempt.relative_to(PHASE), flush=True)
    if status != "success":
        raise RuntimeError(f"{status}: {reason}. Inspect {attempt / 'run.log'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", required=True)
    parser.add_argument("--length", nargs="+", type=int, choices=(4096, 65536, 131072, 262144), required=True)
    parser.add_argument("--pair", type=int, choices=(1, 2), required=True)
    parser.add_argument("--variant", choices=("baseline", "cuda13"))
    parser.add_argument("--swap-physical", nargs=2, type=int, choices=range(8), metavar=("FIRST", "SECOND"))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--profile-sample", type=int, default=6)
    args = parser.parse_args()
    if args.samples < 3 or not 1 < args.profile_sample < args.samples:
        parser.error("The trace requires unprofiled samples on both sides")
    if not args.series.replace("-", "").isalnum():
        parser.error("series must contain letters, numbers or hyphens")
    series = PHASE / "runs" / args.series
    dw, linear, overrides = load_policy(FULL_HEAD / "final-policy.json")
    case = SimpleNamespace(cases=["100b-c8"], lengths=args.length)
    requests = [req for req in requests_for(case, dw, linear, overrides)
                if req["pair"] == args.pair and (args.variant is None or args.variant == req["variant"])]
    with (PHASE / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        series.mkdir(parents=True, exist_ok=False)
        cache = PHASE / "cache"
        if not cache.exists():
            shutil.copytree(FULL_HEAD / "cache", cache)
        legacy.write_json(series / "schedule.json", [dict(req, samples=args.samples) for req in requests])
        for req in requests:
            launch(req, series, args.samples, args.profile_sample, args.swap_physical)


if __name__ == "__main__":
    main()
