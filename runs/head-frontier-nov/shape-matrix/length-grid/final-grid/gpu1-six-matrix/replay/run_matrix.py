"""Run the final-policy B1 length grid on physical GPUs 0–5 of gpu1."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


HERE = Path(__file__).resolve().parent
FINAL_GRID = HERE.parents[1]
FULL_HEAD = FINAL_GRID / "full-head"
SIX = FINAL_GRID / "gpu1-six/replay"
sys.path[:0] = [str(FULL_HEAD), str(SIX)]
from run_six import INDICES, JOB_ID, preflight
from run_final_grid import load_policy, requests_for
from run_grid import ENV_FILES, legacy, provenance, verify_outcome
import summarise_grid


TERMINAL = legacy.TERMINAL
POLICY = FULL_HEAD / "final-policy.json"
TIMEOUT_SECONDS = 1200
WORLD = len(INDICES)


def run_inputs():
    configs = argparse.Namespace(cases=tuple(legacy.CASES), lengths=legacy.LENGTHS)
    dw, linear, overrides = load_policy(POLICY)
    files = {"linear": linear, **{f"dw_c{chunks}": entry for chunks, entry in dw.items()}}
    files.update({f"override_{key}": entry for key, entry in overrides.items()})
    return requests_for(configs, dw, linear, overrides), files


def pin_sources(files):
    sources = provenance({"100b", "700b"}, files)
    prior_path = (FULL_HEAD / "results/final/700b-c8-b1-s4096/pair-01/cuda13/attempt-001/invocation.json")
    prior = legacy.read_json(prior_path)
    if sources["host"] != f"lm-mpi-job-{JOB_ID}-mpimaster-0":
        raise RuntimeError(f"Unexpected replay node: {sources['host']}")
    for key in ("python", "torch", "torch_git_version", "torch_cuda", "baseline_revision", "fixtures_sha256"):
        if sources[key] != prior["provenance"][key]:
            raise RuntimeError(f"Prior full-grid runtime or fixture differs at {key}")
    if list(sources["nccl_version"]) != prior["provenance"]["nccl_version"]:
        raise RuntimeError("The current NCCL version differs from the prior full grid")
    for key in ("CUDA_HOME", "LD_LIBRARY_PATH"):
        if os.environ[key] != prior["settings"][key]:
            raise RuntimeError(f"The loaded image environment differs at {key}")
    for path, sha256 in prior["provenance"]["files_sha256"].items():
        if path in sources["files_sha256"] and sources["files_sha256"][path] != sha256:
            raise RuntimeError(f"Source differs from the prior full grid: {path}")

    helpers = (HERE / "run_matrix.py", HERE / "rank_matrix.py", HERE / "run_detached.sh",
               POLICY, FULL_HEAD / "run_final_grid.py", SIX / "run_six.py", SIX / "rank_six.py",
               FINAL_GRID / "gpu1-health/replay/run_replay.py",
               FINAL_GRID / "c8-diagnosis/phase/profile_rank.py")
    sources["files_sha256"].update({str(path): legacy.sha256(path) for path in helpers})
    sources["cache_seed"] = str(FULL_HEAD / "cache")
    sources["previous_full_grid_invocation"] = str(prior_path)
    return sources


def verify_receipts(attempt, invocation, started, code):
    status, ranks, reason = verify_outcome(attempt, code, invocation)
    if status == "cuda_oom":
        for rank in ranks:
            receipt = legacy.read_json(attempt / f"oom-{rank}.json")
            if not started <= receipt["process_arrived_utc_ns"] <= receipt["occurred_utc_ns"] <= time.time_ns():
                raise RuntimeError(f"Rank {rank} has an invalid OOM timestamp")
            if receipt["phase"] not in ("fixture", "head_setup", "collective_smoke", "warmup",
                                        "allocator_prime", "measurement", "final_backward"):
                raise RuntimeError(f"Rank {rank} reported allocation failure outside the head: {receipt}")
            if receipt["phase"] in ("warmup", "measurement"):
                count = invocation["warmup" if receipt["phase"] == "warmup" else "samples"]
                if not 1 <= receipt["iteration"] <= count:
                    raise RuntimeError(f"Rank {rank} has an invalid OOM iteration")
    if status != "success":
        return status, ranks, reason

    summarise_grid.benchmark(attempt, invocation)
    for name in ("activation", "contract", "physical-device", "rank", "resolved", "timing"):
        if {path.name for path in attempt.glob(f"{name}-*.json")} != {f"{name}-{rank}.json" for rank in INDICES}:
            raise RuntimeError(f"Unexpected {name} receipt set in {attempt}")
    for gpu in invocation["gpu_preflight"]:
        rank = gpu["index"]
        device = legacy.read_json(attempt / f"physical-device-{rank}.json")
        if device["logical_rank"] != rank or device["visible_uuid"] != gpu["uuid"]:
            raise RuntimeError(f"Rank {rank} mapped to an unexpected GPU: {device}")
        contract = legacy.read_json(attempt / f"contract-{rank}.json")
        timings = legacy.read_json(attempt / f"timing-{rank}.json")
        rank_result = legacy.read_json(attempt / f"rank-{rank}.json")
        measured = rank_result["measured"]
        arrivals = contract["sample_barrier_arrived_utc_ns"]
        warmup = contract["warmup_host_utc_ns"]
        previous = contract["process_arrived_utc_ns"]
        if not started <= previous or len(warmup) != invocation["warmup"]:
            raise RuntimeError(f"Invalid rank {rank} process arrival or warmup count")
        for entry, exit_ in warmup:
            if not previous < entry <= exit_:
                raise RuntimeError(f"Invalid rank {rank} warmup timestamps")
            previous = exit_
        if len(arrivals) != invocation["samples"] or len(timings) != invocation["samples"]:
            raise RuntimeError(f"Invalid rank {rank} measurement receipt count")
        for arrived, timing, wall in zip(arrivals, timings, measured, strict=True):
            start, submitted, stop = (timing[f"{key}_utc_ns"] for key in ("start", "submitted", "stop"))
            if (timing["profiled"] or not previous < arrived <= start <= submitted <= stop
                    or abs((stop - start) / 1e6 - wall["ms"]) > 0.001):
                raise RuntimeError(f"Rank {rank} timestamps disagree with its measured call: {timing}")
            previous = stop
        width = rank_result["width"]
        vocab = 128256
        elements = vocab * width
        expected_collectives = [
            dict(kind=kind, full_elements=elements, local_elements=elements // WORLD,
                 world=WORLD, dtype=dtype)
            for kind, dtype in (("all-gather", "torch.bfloat16"), ("sum-reduce-scatter", "torch.float32"))
        ]
        if (contract["collectives"] != expected_collectives or contract["weight_shape"] != [vocab, width]
                or contract["weight_grad_offset"] != [rank * (vocab // WORLD), 0]
                or bool(contract["z_loss_coefficient"]) != (invocation["model"] == "100b")):
            raise RuntimeError(f"Rank {rank} has a different native head contract: {contract}")
    return status, ranks, reason


def run_arm(root, request, sources, cache):
    case, model, seq, chunks, variant = (request[key] for key in ("case", "model", "seq", "chunks", "variant"))
    arm = root / f"{case}-b1-s{seq}" / f"pair-{request['pair']:02d}" / variant
    if legacy.complete(arm, request, sources):
        return
    preflight_started = time.time_ns()
    gpus = preflight()
    preflight_ended = time.time_ns()
    arm.mkdir(parents=True, exist_ok=True)
    next_attempt = max((int(path.name.split("-")[-1]) for path in arm.glob("attempt-*")), default=0) + 1
    attempt = arm / f"attempt-{next_attempt:03d}"
    attempt.mkdir()
    cmd = [
        sys.executable, "-Bu", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={WORLD}", "--max-restarts=0", "--log-dir", str(attempt / "elastic"),
        str(HERE / "rank_matrix.py"), "--model", model, "--batch", "1", "--seq", str(seq),
        "--chunks", str(chunks), "--variant", variant, "--fixture", str(legacy.ROOT / "inputs" / f"{model}.pt"),
        "--out", str(attempt), "--warmup", str(request["warmup"]), "--samples", str(request["samples"]),
    ]
    settings = legacy.settings_for(model, variant, gpus)
    settings.update(
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"),
        CUDA_CACHE_PATH=str(cache / "cuda"), C8_PHASE_PROFILE_SAMPLE="0",
        LD_LIBRARY_PATH=os.environ["LD_LIBRARY_PATH"], CUDA_HOME=os.environ["CUDA_HOME"],
    )
    unset = {
        "PYTORCH_CUDA_ALLOC_CONF", "GIFT_EXACT_FOLDED_CE_LIBRARY", "GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256",
        "HEAD_REPLAY_DHIDDEN_DIGEST", "CUDA_LAUNCH_BLOCKING",
        *(key for variable in ENV_FILES.values() for key in (variable, variable + "_SHA256")),
    }
    if variant == "cuda13":
        for name, variable in ENV_FILES.items():
            selected = request["manifests"][name]
            settings[variable], settings[variable + "_SHA256"] = selected["path"], selected["sha256"]
    env = {key: value for key, value in os.environ.items() if key not in unset}
    env.update(settings)
    invocation = dict(
        request, cmd=cmd, settings=settings, unset=sorted(unset - settings.keys()), provenance=sources,
        job_id=JOB_ID, gpu_preflight=gpus, physical_gpu_index_by_logical_rank=INDICES, world_size=WORLD,
        gpu_preflight_started_utc_ns=preflight_started, gpu_preflight_ended_utc_ns=preflight_ended,
        idle_memory_mib=legacy.IDLE_MEMORY_MIB, timeout_seconds=TIMEOUT_SECONDS,
    )
    legacy.write_json(attempt / "invocation.json", invocation)
    started = time.time_ns()
    print(legacy.now(), "start", attempt.relative_to(HERE), flush=True)
    code, status, oom_ranks, reason = None, "error", [], None
    try:
        with (attempt / "run.log").open("w") as log:
            code, stopped = legacy.run_process(cmd, env, log, TIMEOUT_SECONDS)
        if stopped is None:
            status, oom_ranks, reason = verify_receipts(attempt, invocation, started, code)
        else:
            status, reason = stopped, f"Replay {stopped}"
    except Exception as error:
        status, reason = "error", str(error)
        raise
    finally:
        legacy.write_json(attempt / "status.json", dict(status=status, returncode=code, reason=reason,
                          oom_ranks=oom_ranks, started_utc_ns=started, ended_utc_ns=time.time_ns()))
        print(legacy.now(), status, attempt.relative_to(HERE), reason or "", flush=True)
    if status not in TERMINAL:
        raise RuntimeError(f"{status}: {reason}. Inspect {attempt / 'run.log'}")


def summarize(root, requests):
    result = summarise_grid.summarize(root, requests)
    result.update(world_size=WORLD, warmup=10, measured_samples=10)
    for row in result["rows"]:
        for pair in row["pairs"]:
            baseline, candidate = pair["baseline"], pair["cuda13"]
            if baseline["status"] == candidate["status"] == "success":
                if baseline["components"].keys() != candidate["components"].keys():
                    raise RuntimeError(f"Loss component names differ for {row['case']} at {row['sequence']}")
                a = [baseline["loss"], *baseline["components"].values()]
                b = [candidate["loss"], *(candidate["components"][key] for key in baseline["components"])]
                if not all(math.isclose(x, y, rel_tol=1e-6) for x, y in zip(a, b, strict=True)):
                    raise RuntimeError(f"Loss differs for {row['case']} at {row['sequence']}: {a}, {b}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    requests, files = run_inputs()
    if args.plan_only:
        print(json.dumps(requests, indent=2))
        return

    legacy.WORLD = summarise_grid.WORLD = WORLD
    with (HERE / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = pin_sources(files)
        for req in requests:
            source = legacy.BASELINE_SOURCE if req["variant"] == "baseline" else legacy.CANDIDATE_SOURCE
            req["loss_source_sha256"] = sources["files_sha256"][str(source / "torchtitan/components/loss.py")]
        root = HERE / "results"
        root.mkdir(exist_ok=True)
        schedule_path = root / "schedule.json"
        if schedule_path.is_file() and legacy.read_json(schedule_path) != requests:
            raise RuntimeError(f"{schedule_path} contains a different grid")
        legacy.write_json(schedule_path, requests)
        cache = HERE / "cache"
        if not cache.exists():
            temporary = HERE / f"cache-copy-{os.getpid()}"
            shutil.copytree(sources["cache_seed"], temporary)
            temporary.rename(cache)
        try:
            for request in requests:
                run_arm(root, request, sources, cache)
        finally:
            result = summarize(root, requests)
            legacy.write_json(HERE / "summary.json", result)
        if any(row["status"] != "complete" for row in result["rows"]):
            raise RuntimeError("Some requested pairs did not complete")


if __name__ == "__main__":
    main()
