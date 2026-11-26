"""Run fresh eight-rank Baseline A and CUDA 13 replays on the B1 length grid."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


GRID = Path(__file__).resolve().parent
ROOT = GRID.parent
TASK = ROOT.parents[2]
WORLD = 8
CASES = {"100b-c2": ("100b", 2), "100b-c8": ("100b", 8), "700b-c8": ("700b", 8)}
LENGTHS = tuple(4096 * 2**power for power in range(7))
VARIANTS = ("baseline", "cuda13")
CE_SHA256 = "d3343cd518a05a6d5bea58b0493fa33373e040f1ce3dd5864b14c03033528929"
CE_LIBRARY = ROOT / "tuning/ce/centered-tuned.so"
CATALOGS = (
    ROOT / "tuning/linear-plans.json",
    ROOT / "tuning/lt/results/calibrated-catalog.json",
)
BASELINE_SOURCE = TASK.parent / "a10-cute-001/gift"
CANDIDATE_SOURCE = ROOT / "integrated/gift"
IDLE_MEMORY_MIB = 256
TERMINAL = {"success", "cuda_oom"}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plan(cases, lengths):
    for length_index, seq in enumerate(LENGTHS):
        if seq not in lengths:
            continue
        for case_index, case in enumerate(CASES):
            if case not in cases:
                continue
            order = VARIANTS if (length_index + case_index) % 2 == 0 else VARIANTS[::-1]
            for variant in order:
                yield case, seq, variant


def nvidia_smi(query):
    return subprocess.check_output(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    ).strip()


def gpu_preflight():
    inventory = nvidia_smi("--query-gpu=index,uuid,name,memory.total,memory.used,driver_version")
    gpu_rows = csv.reader(inventory.splitlines(), skipinitialspace=True)
    gpus = []
    for index, uuid, name, total, used, driver in gpu_rows:
        gpus.append(
            dict(index=int(index), uuid=uuid, name=name, total_mib=int(total),
                 used_mib=int(used), driver=driver)
        )
    if len(gpus) != WORLD or len({gpu["uuid"] for gpu in gpus}) != WORLD:
        raise RuntimeError(f"Expected {WORLD} distinct GPUs, got {inventory!r}")
    if any("H100" not in gpu["name"] or gpu["total_mib"] < 78 * 1024 for gpu in gpus):
        raise RuntimeError(f"Expected the eight H100 80GB GPUs, got {inventory!r}")
    gpus.sort(key=lambda gpu: gpu["index"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        selection = set(visible.split(","))
        if selection not in ({str(gpu["index"]) for gpu in gpus}, {gpu["uuid"] for gpu in gpus}):
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES does not expose all checked GPUs: {visible!r}")
    apps = nvidia_smi("--query-compute-apps=gpu_uuid,pid")
    if apps or any(gpu["used_mib"] > IDLE_MEMORY_MIB for gpu in gpus):
        raise RuntimeError(f"GPU preflight found memory in use or compute processes: {inventory!r}; {apps!r}")
    return gpus


def provenance(models):
    import torch

    source_files = [
        Path(__file__).resolve(), GRID / "replay.py",
        ROOT / "matrix_fixture.py", ROOT / "tuning/linear_policy.py",
        ROOT / "tuning/lt/lt_writer.py", *CATALOGS, CE_LIBRARY,
        BASELINE_SOURCE / "torchtitan/components/loss.py",
        CANDIDATE_SOURCE / "torchtitan/components/loss.py",
        CANDIDATE_SOURCE / "torchtitan/distributed/gfsdp/_staged_head_dw.py",
        CANDIDATE_SOURCE / "torchtitan/ops/exact_folded_ce.py",
    ]
    historical_replay = ROOT / "tuning/head_bench.py"
    source_files.append(historical_replay if historical_replay.is_file() else ROOT / "head_bench.py")
    for helper in (GRID / "fixture.py",):
        if helper.is_file():
            source_files.append(helper)
    digests = {str(path): sha256(path) for path in source_files}
    if digests[str(CE_LIBRARY)] != CE_SHA256:
        raise RuntimeError(f"Unexpected SHA-256 for {CE_LIBRARY}: {digests[str(CE_LIBRARY)]}")
    fixtures = {model: sha256(ROOT / "inputs" / f"{model}.pt") for model in sorted(models)}
    return {
        "host": socket.gethostname(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "baseline_expected_revision": "e9b7cdbb",
        "files_sha256": digests,
        "fixtures_sha256": fixtures,
    }


def settings_for(model, variant, gpus):
    baseline = variant == "baseline"
    paths = [BASELINE_SOURCE if baseline else CANDIDATE_SOURCE, ROOT]
    if baseline:
        paths.extend(ROOT / f"src/{vendor}" for vendor in ("liger/src", "fla", "apple"))
    paths.extend((TASK / ".worktrees/final-all-fix/gift", TASK.parent / "lm10-c1/gift"))
    settings = {
        "PYTHONPATH": os.pathsep.join(map(str, paths)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCHINDUCTOR_CUDAGRAPHS": "0",
        "TORCHINDUCTOR_CACHE_DIR": str(ROOT / "cache/head-inductor"),
        "TRITON_CACHE_DIR": str(ROOT / "cache/head-triton"),
        "CUDACHERRY_DISABLE_JIT": "1",
        "GIFT_DIRECT_BF16_DHIDDEN": str(int(not baseline)),
        "GIFT_FUSED_HIDDEN_Z_LOSS": str(int(not baseline and model == "100b")),
        "GIFT_STABLE_CE_FOLDED_BACKWARD": str(int(not baseline)),
        "GIFT_STAGED_HEAD_DW": "0",
        "GIFT_EXPERIMENT_LM_HEAD_MATRIX": "" if baseline else "generic",
        "GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS": "",
        "CUDA_VISIBLE_DEVICES": ",".join(gpu["uuid"] for gpu in gpus),
    }
    if not baseline:
        settings["GIFT_EXACT_FOLDED_CE_LIBRARY"] = str(CE_LIBRARY)
        settings["GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256"] = CE_SHA256
    return settings


def complete(arm_dir, request, sources):
    for attempt in sorted(arm_dir.glob("attempt-*"), reverse=True):
        status_file = attempt / "status.json"
        if not status_file.is_file() or read_json(status_file)["status"] not in TERMINAL:
            continue
        saved = read_json(attempt / "invocation.json")
        if any(saved[key] != value for key, value in request.items()):
            raise RuntimeError(f"Completed attempt has different measurement settings: {attempt}. Use a new --series.")
        previous = saved["provenance"]
        model = request["model"]
        keys = ("files_sha256", "torch", "torch_git_version", "torch_cuda", "python", "host")
        if (any(previous[key] != sources[key] for key in keys)
                or previous["fixtures_sha256"][model] != sources["fixtures_sha256"][model]):
            raise RuntimeError(f"Completed attempt has different provenance: {attempt}. Use a new --series.")
        return True
    return False


def outcome(attempt, code, request):
    oom_ranks = []
    for rank in range(WORLD):
        marker = attempt / f"oom-{rank}.json"
        if not marker.is_file():
            continue
        receipt = read_json(marker)
        message = receipt.get("error", receipt.get("message", ""))
        if receipt.get("rank", rank) != rank or "cuda out of memory" not in message.lower():
            return "error", [], f"Invalid CUDA OOM receipt {marker.name}"
        oom_ranks.append(rank)
    if code != 0:
        if oom_ranks:
            return "cuda_oom", oom_ranks, "A replay rank raised a CUDA allocation out-of-memory exception"
        return "error", [], f"torchrun exit {code} without a CUDA OOM receipt"
    if oom_ranks:
        return "error", oom_ranks, "torchrun completed with CUDA OOM receipts"
    for rank in range(WORLD):
        result = attempt / f"rank-{rank}.json"
        activation = attempt / f"activation-{rank}.json"
        if not result.is_file() or not activation.is_file():
            return "error", [], f"Rank {rank} did not save both receipts"
        row = read_json(result)
        if row["batch"] != request["batch"] or row["seq"] != request["seq"]:
            return "error", [], f"Rank {rank} returned a different batch or sequence length"
        read_json(activation)
    return "success", [], None


def run_process(cmd, env, log, timeout):
    with subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    ) as process:
        try:
            return process.wait(timeout=timeout), None
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            # TorchElastic owns the rank processes and handles SIGTERM for cleanup.
            process.terminate()
            try:
                code = process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            return code, "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"


def run_arm(args, case, seq, variant, sources):
    model, chunks = CASES[case]
    request = dict(case=case, model=model, batch=1, seq=seq, chunks=chunks, variant=variant,
                   warmup=args.warmup, samples=args.samples)
    arm_dir = GRID / "results" / args.series / f"{case}-b1-s{seq}" / variant
    if complete(arm_dir, request, sources):
        return
    gpus = gpu_preflight()
    arm_dir.mkdir(parents=True, exist_ok=True)
    next_attempt = max((int(path.name.split("-")[-1]) for path in arm_dir.glob("attempt-*")), default=0) + 1
    attempt = arm_dir / f"attempt-{next_attempt:03d}"
    attempt.mkdir()
    cmd = [
        sys.executable, "-Bu", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={WORLD}", "--max-restarts=0", "--log-dir", str(attempt / "elastic"),
        str(GRID / "replay.py"), "--model", model, "--batch", "1", "--seq", str(seq),
        "--chunks", str(chunks), "--variant", variant,
        "--fixture", str(ROOT / "inputs" / f"{model}.pt"), "--out", str(attempt),
        "--warmup", str(args.warmup), "--samples", str(args.samples),
    ]
    settings = settings_for(model, variant, gpus)
    unset = (
        "PYTORCH_CUDA_ALLOC_CONF", "GIFT_EXACT_FOLDED_CE_LIBRARY",
        "GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256", "HEAD_REPLAY_DHIDDEN_DIGEST", "CUDA_LAUNCH_BLOCKING",
    )
    env = {key: value for key, value in os.environ.items() if key not in unset}
    env.update(settings)
    write_json(attempt / "invocation.json", {
        **request, "cmd": cmd, "settings": settings, "unset": sorted(set(unset) - settings.keys()),
        "provenance": sources, "gpu_preflight": gpus, "idle_memory_mib": IDLE_MEMORY_MIB,
        "timeout_seconds": args.timeout,
    })
    started_at = now()
    print(started_at, "start", attempt, flush=True)
    code = None
    try:
        with (attempt / "run.log").open("w") as log:
            code, stopped = run_process(cmd, env, log, args.timeout)
        if stopped is None:
            status, oom_ranks, reason = outcome(attempt, code, request)
        else:
            status, oom_ranks, reason = stopped, [], f"Replay {stopped} before a completed result"
    except (OSError, ValueError, KeyError) as error:
        status, oom_ranks, reason = "error", [], str(error)
    ended_at = now()
    write_json(attempt / "status.json", dict(
        status=status, returncode=code, started_at=started_at, ended_at=ended_at,
        oom_ranks=oom_ranks, reason=reason,
    ))
    print(ended_at, status, attempt, flush=True)
    if status not in TERMINAL:
        raise RuntimeError(f"{reason}. Inspect {attempt / 'run.log'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-mode", choices=("single",), default="single")
    parser.add_argument("--series", default="sweep")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=tuple(CASES))
    parser.add_argument("--lengths", nargs="+", type=int, choices=LENGTHS, default=LENGTHS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1200, help="Per-variant deadline in seconds")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.series):
        parser.error("--series must be a single directory name")
    if min(args.warmup, args.samples, args.timeout) <= 0:
        parser.error("--warmup, --samples and --timeout must be positive")
    with (GRID / ".launch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A length-grid launcher is already running") from error
        sources = provenance({CASES[case][0] for case in args.cases})
        for case, seq, variant in plan(args.cases, args.lengths):
            run_arm(args, case, seq, variant, sources)


if __name__ == "__main__":
    main()
