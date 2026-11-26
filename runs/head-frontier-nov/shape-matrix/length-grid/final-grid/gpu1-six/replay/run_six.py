"""Run the corrected 100B C8 128k cell in ABBA order on physical GPUs 0–5."""

import csv
import math
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
FINAL_GRID = HERE.parents[1]
PHASE = FINAL_GRID / "c8-diagnosis/phase"
CASE = "100b-c8-b1-s131072"
BASE = FINAL_GRID / "full-head/results/corrected-100b-c8-128k"
REFERENCES = {
    "baseline": BASE / CASE / "pair-02/baseline/attempt-001/invocation.json",
    "cuda13": PHASE / "runs/profile-128k-swap-late" / CASE / "pair-02/cuda13/attempt-001/invocation.json",
}
INDICES = list(range(6))
WARMUP = 10
SAMPLES = 32

sys.path.insert(0, str(FINAL_GRID / "gpu1-health/replay"))
from run_replay import JOB_ID, legacy, summarize, verify_inputs, verify_outcome
import summarise_grid


def preflight():
    selection = "--id=" + ",".join(map(str, INDICES))

    def query(field):
        return subprocess.check_output(
            ["nvidia-smi", selection, field, "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        ).strip()

    inventory = query("--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,utilization.gpu")
    apps = query("--query-compute-apps=gpu_uuid,pid")
    gpus = [
        dict(index=int(index), uuid=uuid, name=name, total_mib=int(total),
             used_mib=int(used), driver=driver, utilization_pct=int(utilization))
        for index, uuid, name, total, used, driver, utilization
        in csv.reader(inventory.splitlines(), skipinitialspace=True)
    ]
    gpus.sort(key=lambda gpu: gpu["index"])
    if ([gpu["index"] for gpu in gpus] != INDICES
            or len({gpu["uuid"] for gpu in gpus}) != len(INDICES)
            or len({gpu["driver"] for gpu in gpus}) != 1
            or any("H100" not in gpu["name"] or gpu["total_mib"] < 78 * 1024 for gpu in gpus)):
        raise RuntimeError(f"Unexpected selected GPUs: {inventory!r}")
    if apps or any(gpu["used_mib"] > legacy.IDLE_MEMORY_MIB or gpu["utilization_pct"] for gpu in gpus):
        raise RuntimeError(f"Selected GPUs are busy: {inventory!r}; {apps!r}")
    return gpus


def run_arm(attempt, prior, provenance, cache, position):
    gpus = preflight()
    attempt.mkdir()
    cmd = list(prior["cmd"])
    if cmd[0] != sys.executable:
        raise RuntimeError(f"Expected {cmd[0]}, got {sys.executable}")
    cmd[cmd.index("--nproc-per-node=8")] = f"--nproc-per-node={len(INDICES)}"
    cmd[cmd.index("--log-dir") + 1] = str(attempt / "elastic")
    cmd[cmd.index("--log-dir") + 2] = str(HERE / "rank_six.py")
    cmd[cmd.index("--out") + 1] = str(attempt)
    cmd[cmd.index("--samples") + 1] = str(SAMPLES)
    settings = dict(prior["settings"], C8_PHASE_PROFILE_SAMPLE="0",
                    CUDA_VISIBLE_DEVICES=",".join(gpu["uuid"] for gpu in gpus),
                    TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                    TRITON_CACHE_DIR=str(cache / "triton"), CUDA_CACHE_PATH=str(cache / "cuda"))
    env = {key: value for key, value in os.environ.items() if key not in prior["unset"]}
    env.update(settings)
    invocation = dict(prior, position=position, pair=1 if position <= 2 else 2,
                      samples=SAMPLES, cmd=cmd, settings=settings, gpu_preflight=gpus,
                      provenance=provenance, job_id=JOB_ID, world_size=len(INDICES),
                      physical_gpu_index_by_logical_rank=INDICES,
                      prior_invocation=str(REFERENCES[prior["variant"]]),
                      prior_invocation_sha256=legacy.sha256(REFERENCES[prior["variant"]]),
                      six_rank_sources={name: legacy.sha256(HERE / name)
                                        for name in ("run_six.py", "rank_six.py", "run_detached.sh")})
    legacy.write_json(attempt / "invocation.json", invocation)
    started = time.time_ns()
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
            assert_complete(attempt)
            summarize(attempt, invocation)
    except Exception as error:
        status, reason = "error", str(error)
        raise
    finally:
        legacy.write_json(attempt / "status.json", dict(status=status, returncode=code, reason=reason,
                          oom_ranks=oom_ranks, started_utc_ns=started, ended_utc_ns=time.time_ns()))
        print(legacy.now(), status, attempt, flush=True)
    if status != "success":
        raise RuntimeError(f"{status}: {reason}. Inspect {attempt / 'run.log'}")


def assert_complete(arm):
    for file_type in ("activation", "contract", "physical-device", "rank", "resolved", "timing"):
        expected_names = {f"{file_type}-{rank}.json" for rank in INDICES}
        if {path.name for path in arm.glob(f"{file_type}-*.json")} != expected_names:
            raise RuntimeError(f"Unexpected {file_type} receipts in {arm}")
    for rank in INDICES:
        c = legacy.read_json(arm / f"contract-{rank}.json")
        timing = legacy.read_json(arm / f"timing-{rank}.json")
        last_stop = 0
        if len(c["warmup_host_utc_ns"]) != WARMUP:
            raise RuntimeError(f"Missing warmup timestamps for rank {rank}")
        for start, stop in c["warmup_host_utc_ns"]:
            if not last_stop < start <= stop < timing[0]["start_utc_ns"]:
                raise RuntimeError(f"Invalid warmup timestamps for rank {rank}")
            last_stop = stop


def compare_arms(a, b):
    a_summary, b_summary = [legacy.read_json(arm / "summary.json") for arm in (a, b)]
    for key in ("valid_tokens", "valid_mtp_tokens", "denominator"):
        if a_summary[key] != b_summary[key]:
            raise RuntimeError(f"Mismatched fixture field {key} in {a} and {b}")
    for key in ("loss", "components"):
        a_values = [a_summary[key]] if key == "loss" else a_summary[key].values()
        b_values = [b_summary[key]] if key == "loss" else b_summary[key].values()
        if not all(math.isclose(x, y, rel_tol=1e-6) for x, y in zip(a_values, b_values, strict=True)):
            raise RuntimeError(f"Mismatched {key} in {a} and {b}")
    gradients = {}
    for rank in INDICES:
        x, y = [legacy.read_json(arm / f"contract-{rank}.json") for arm in (a, b)]
        if any(x[key] != y[key] for key in x.keys() - {"gradients", "warmup_host_utc_ns"}):
            raise RuntimeError(f"Rank {rank} has different shape, collective or loss contracts")
        for name in x["gradients"]:
            u, v = (record["gradients"][name] for record in (x, y))
            if any(u[key] != v[key] for key in ("shape", "dtype", "sample_stride")):
                raise RuntimeError(f"Rank {rank} has different {name} gradient contracts")
            rel_norm = abs(u["l2"] - v["l2"]) / u["l2"]
            rel_samples = math.sqrt(sum((p - q) ** 2 for p, q in zip(u["sampled"], v["sampled"], strict=True))
                                    / sum(p ** 2 for p in u["sampled"]))
            if max(rel_norm, rel_samples) > 0.02:
                raise RuntimeError(f"Rank {rank} {name} gradient differs: {rel_norm}, {rel_samples}")
            gradients.setdefault(name, []).append(dict(relative_l2_norm_delta=rel_norm,
                                                      sampled_relative_l2_error=rel_samples))
    return dict(baseline=a.name, cuda13=b.name, gradient_comparison_by_rank=gradients,
                baseline_ms=a_summary["ms"], cuda13_ms=b_summary["ms"],
                speedup_pct=100 * (a_summary["ms"] / b_summary["ms"] - 1),
                latency_reduction_pct=100 * (1 - b_summary["ms"] / a_summary["ms"]),
                baseline_total_gib=a_summary["total_gib"], cuda13_total_gib=b_summary["total_gib"])


def main():
    refs = {variant: legacy.read_json(path) for variant, path in REFERENCES.items()}
    for variant, prior in refs.items():
        expected = dict(model="100b", batch=1, seq=131072, chunks=8, variant=variant, warmup=WARMUP)
        if any(prior[key] != value for key, value in expected.items()):
            raise RuntimeError(f"Unexpected pinned invocation: {REFERENCES[variant]}")
        if legacy.read_json(REFERENCES[variant].with_name("status.json"))["status"] != "success":
            raise RuntimeError(f"Unsuccessful pinned invocation: {REFERENCES[variant]}")
    provenance = verify_inputs(refs["cuda13"])
    for variant, prior in refs.items():
        loss_source = Path(prior["settings"]["PYTHONPATH"].split(os.pathsep)[0]) / "torchtitan/components/loss.py"
        if provenance["files_sha256"][str(loss_source)] != prior["loss_source_sha256"]:
            raise RuntimeError(f"Unexpected source SHA-256 for {variant}")
    cache = HERE / "cache"
    if not cache.exists():
        shutil.copytree(PHASE / "cache", cache)
    legacy.WORLD = summarise_grid.WORLD = len(INDICES)
    arms = HERE / "arms"
    arms.mkdir(exist_ok=True)
    for idx, variant in enumerate(("baseline", "cuda13", "cuda13", "baseline"), start=1):
        run_arm(arms / f"{idx:02d}-{variant}", refs[variant], provenance, cache, idx)
        if idx in (2, 4):
            baseline = arms / f"{1 if idx == 2 else 4:02d}-baseline"
            candidate = arms / f"{idx if idx == 2 else 3:02d}-cuda13"
            legacy.write_json(HERE / f"pair-{idx // 2:02d}.json", compare_arms(baseline, candidate))
    pairs = [legacy.read_json(HERE / f"pair-{idx:02d}.json") for idx in (1, 2)]
    legacy.write_json(HERE / "summary.json",
                      dict(pairs=pairs, speedup_pct_median_of_pairs=median(p["speedup_pct"] for p in pairs)))


if __name__ == "__main__":
    main()
