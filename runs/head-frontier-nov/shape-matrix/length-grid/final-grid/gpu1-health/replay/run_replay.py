"""Replay the corrected 128k head once on the replacement eight-H100 node."""

import os
from pathlib import Path
import shutil
import socket
import sys
import time
from statistics import median


HERE = Path(__file__).resolve().parent
FINAL_GRID = HERE.parents[1]
FULL_HEAD = FINAL_GRID / "full-head"
PHASE = FINAL_GRID / "c8-diagnosis/phase"
HEAD_ARM = "100b-c8-b1-s131072/pair-02/cuda13/attempt-001"
JOB_ID = "7c9acdc0-339d-480d-b7ea-9d3a43448e3a"
sys.path.insert(0, str(FULL_HEAD))

from run_grid import legacy, verify_outcome
from summarise_grid import benchmark


def verify_inputs(prior):
    import torch

    provenance = prior["provenance"]
    runtime = dict(host=socket.gethostname(), python=sys.version, torch=str(torch.__version__),
                   torch_git_version=torch.version.git_version, torch_cuda=torch.version.cuda,
                   nccl_version=list(torch.cuda.nccl.version()))
    if runtime["host"] != f"lm-mpi-job-{JOB_ID}-mpimaster-0":
        raise RuntimeError(f"Unexpected replay node: {runtime['host']}")
    for key in runtime.keys() - {"host"}:
        if runtime[key] != provenance[key]:
            raise RuntimeError(f"Runtime differs from the pinned launch at {key}: {runtime[key]}")
    for key in ("LD_LIBRARY_PATH", "CUDA_HOME"):
        if os.environ[key] != prior["settings"][key]:
            raise RuntimeError(f"Loaded {key} differs from the pinned launch")
    files = {**provenance["files_sha256"], **prior["diagnostic_sources"],
             prior["reference"]: prior["reference_sha256"],
             str(legacy.ROOT / "inputs/100b.pt"): provenance["fixtures_sha256"]["100b"]}
    for name, expected in files.items():
        if legacy.sha256(Path(name)) != expected:
            raise RuntimeError(f"SHA-256 differs from the pinned launch: {name}")
    return {**provenance, **runtime}


def summarize(attempt, invocation):
    results = benchmark(attempt, invocation)
    samples = invocation["samples"]
    times = []
    for gpu in invocation["gpu_preflight"]:
        rank = gpu["index"]
        device = legacy.read_json(attempt / f"physical-device-{rank}.json")
        if device["logical_rank"] != rank or device["visible_uuid"] != gpu["uuid"]:
            raise RuntimeError(f"Rank {rank} UUID receipt differs from preflight: {device}")
        timings = legacy.read_json(attempt / f"timing-{rank}.json")
        measured = legacy.read_json(attempt / f"rank-{rank}.json")["measured"]
        if len(timings) != samples:
            raise RuntimeError(f"Rank {rank} has {len(timings)} timestamps, expected {samples}")
        previous = 0
        for timing, wall in zip(timings, measured, strict=True):
            start, submitted, stop = (timing[f"{key}_utc_ns"] for key in ("start", "submitted", "stop"))
            if (timing["profiled"] or not previous < start <= submitted <= stop
                    or abs((stop - start) / 1e6 - wall["ms"]) > 0.001):
                raise RuntimeError(f"Rank {rank} timing disagrees with the measured call: {timing}")
            previous = stop
        times.append(timings)
    results.update(first_five_rank_max_median_ms=median(results["steps_ms"][:5]),
                   last_six_rank_max_median_ms=median(results["steps_ms"][-6:]),
                   measured_start_utc_ns=min(timing[0]["start_utc_ns"] for timing in times),
                   measured_stop_utc_ns=max(timing[-1]["stop_utc_ns"] for timing in times))
    legacy.write_json(attempt / "summary.json", results)


def main():
    prior_path = PHASE / "runs/profile-128k-swap-late" / HEAD_ARM / "invocation.json"
    prior = legacy.read_json(prior_path)
    expected = dict(model="100b", batch=1, seq=131072, chunks=8, variant="cuda13", warmup=10, samples=32)
    if any(prior[key] != value for key, value in expected.items()):
        raise RuntimeError(f"Expected the corrected 10+32 arm: {prior_path}")
    prior_status = legacy.read_json(prior_path.with_name("status.json"))
    if prior_status["status"] != "success":
        raise RuntimeError(f"Expected a successful reference run: {prior_status}")
    provenance = verify_inputs(prior)
    gpus = legacy.gpu_preflight()
    order = list(range(legacy.WORLD))
    if [gpu["index"] for gpu in gpus] != order or len({gpu["driver"] for gpu in gpus}) != 1:
        raise RuntimeError(f"Expected eight consecutive GPU indices and one host driver version: {gpus}")
    cache = HERE / "cache"
    if not cache.exists():
        shutil.copytree(PHASE / "cache", cache)
    # Recheck availability immediately before creating the only attempt.
    gpus = legacy.gpu_preflight()
    attempt = HERE / "attempt-001"
    attempt.mkdir()
    cmd = list(prior["cmd"])
    if cmd[0] != sys.executable:
        raise RuntimeError(f"Expected {cmd[0]}, got {sys.executable}")
    cmd[cmd.index("--log-dir") + 1] = str(attempt / "elastic")
    cmd[cmd.index("--out") + 1] = str(attempt)
    settings = dict(prior["settings"], C8_PHASE_PROFILE_SAMPLE="0",
                    CUDA_VISIBLE_DEVICES=",".join(gpu["uuid"] for gpu in gpus),
                    TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                    TRITON_CACHE_DIR=str(cache / "triton"), CUDA_CACHE_PATH=str(cache / "cuda"))
    env = {key: value for key, value in os.environ.items() if key not in prior["unset"]}
    env.update(settings)
    invocation = dict(prior, cmd=cmd, settings=settings, gpu_preflight=gpus, provenance=provenance,
                      job_id=JOB_ID, physical_gpu_index_by_logical_rank=order,
                      extended_reference=str(prior_path), extended_reference_sha256=legacy.sha256(prior_path),
                      replay_source_sha256=legacy.sha256(Path(__file__)))
    legacy.write_json(attempt / "invocation.json", invocation)
    started_utc_ns = time.time_ns()
    print(legacy.now(), "start", attempt, flush=True)
    code, status, oom_ranks, reason = None, "error", [], None
    try:
        with (attempt / "run.log").open("w") as log:
            code, stopped = legacy.run_process(cmd, env, log, prior["timeout_seconds"])
        if stopped is not None:
            status, reason = stopped, f"Replay {stopped}"
        else:
            status, oom_ranks, reason = verify_outcome(attempt, code, invocation)
        if status == "success":
            summarize(attempt, invocation)
    except Exception as error:
        status, reason = "error", str(error)
        raise
    finally:
        legacy.write_json(attempt / "status.json", dict(status=status, returncode=code, reason=reason,
                          oom_ranks=oom_ranks, started_utc_ns=started_utc_ns, ended_utc_ns=time.time_ns()))
        print(legacy.now(), status, attempt, flush=True)
    if status != "success":
        raise RuntimeError(f"{status}: {reason}. Inspect {attempt / 'run.log'}")


if __name__ == "__main__":
    main()
