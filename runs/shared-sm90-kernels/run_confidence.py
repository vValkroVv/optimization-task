"""Run a fixed, counterbalanced set of independent Hopper component pairs."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAIRS = 16
WARMUP = 100
SAMPLES = 256
SEED = 2048
GPU_BY_CASE = {"norm-residual": 6, "sac-residual": 6, "dense-fp8": 7, "sac-plain": 7}
SWITCHES = [
    "GIFT_CUTE_NORM_FUSE_RESIDUAL",
    "GIFT_CUTE_NORM_SAC_STATS_ONLY",
    "GIFT_FP8_QUANTIZE_WEIGHT_TAIL",
]


def output(args):
    return subprocess.check_output(args, text=True).strip()


def nvidia(gpu, query):
    return output(["nvidia-smi", f"--id={gpu}", query, "--format=csv,noheader,nounits"])


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def prepare(raw, prior_path):
    prior = json.loads(prior_path.read_text())
    revision = output(["git", "-C", REPO, "rev-parse", "HEAD"])
    if revision != prior["revision"] or output(
        ["git", "-C", REPO, "status", "--porcelain", "--", "gift"]
    ):
        raise RuntimeError("baseline training source differs from the validated run")
    rng = random.Random(SEED)
    by_case = {result["case"]: result for result in prior["cases"]}
    cases = {}
    for case, gpu in GPU_BY_CASE.items():
        result = by_case[case]
        fixture = Path(result["fixture"])
        config = Path(result["resolved_config"])
        if (
            sha(fixture) != result["fixture_sha256"]
            or sha(config) != result["resolved_config_sha256"]
        ):
            raise RuntimeError(f"fixture or configuration changed for {case}")
        orders = ["AB", "BA"] * (PAIRS // 2)
        rng.shuffle(orders)
        cases[case] = {
            "gpu": gpu,
            "uuid": nvidia(gpu, "--query-gpu=uuid"),
            "orders": orders,
            "fixture": str(fixture),
            "fixture_sha256": result["fixture_sha256"],
            "resolved_config": str(config),
            "resolved_config_sha256": result["resolved_config_sha256"],
        }
    raw.mkdir()
    for case in cases:
        (raw / case).mkdir()
    write_json(
        raw / "plan.json",
        {
            "pairs": PAIRS,
            "warmup": WARMUP,
            "samples": SAMPLES,
            "seed": SEED,
            "revision": revision,
            "prior_report": str(prior_path),
            "prior_report_sha256": sha(prior_path),
            "harness_sha256": {
                name: sha(HERE / name)
                for name in ("run_confidence.py", "measure.py", "norm.py", "dense.py")
            },
            "cases": cases,
        },
    )


def run_arm(raw, plan, case, pair, position, arm):
    config = plan["cases"][case]
    gpu = config["gpu"]
    directory = raw / case
    base = directory / f"{pair:02d}-{position}-{arm}"
    receipt = base.with_suffix(".json")
    log = base.with_suffix(".log")
    if receipt.exists():
        return
    apps = nvidia(gpu, "--query-compute-apps=pid")
    if apps:
        raise RuntimeError(f"GPU {gpu} already has compute processes: {apps}")
    telemetry = "--query-gpu=temperature.gpu,clocks.sm,memory.used,utilization.gpu"
    cache = directory / f"cache-{arm}"
    env = os.environ.copy()
    for switch in SWITCHES:
        env.pop(switch, None)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": config["uuid"],
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TORCH_LOGS": "recompiles,graph_breaks",
            "TORCHINDUCTOR_MIX_ORDER_REDUCTION": "0",
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TILELANG_CACHE_DIR": str(cache / "tilelang"),
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    (
                        str(REPO / "gift"),
                        str(REPO / "gift/contrib/flash-linear-attention"),
                        env.get("PYTHONPATH"),
                    ),
                )
            ),
        }
    )
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=30",
        "300",
        sys.executable,
        "-B",
        "-u",
        str(HERE / "measure.py"),
        "--case",
        case,
        "--arm",
        arm,
        "--fixture",
        config["fixture"],
        "--resolved-config",
        config["resolved_config"],
        "--output",
        str(receipt),
        "--warmup",
        str(plan["warmup"]),
        "--samples",
        str(plan["samples"]),
    ]
    if pair == 1:
        command.append("--profile")
    print(f"GPU {gpu} pair {pair:02d}/{plan['pairs']} {case} {arm}", flush=True)
    with log.open("x") as handle:
        before = nvidia(gpu, telemetry)
        print(
            json.dumps({"command": command, "gpu_before": before}),
            file=handle,
            flush=True,
        )
        subprocess.run(
            command,
            env=env,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
        print(
            json.dumps({"gpu_after": nvidia(gpu, telemetry)}), file=handle, flush=True
        )


def run_lane(raw, gpu):
    plan = json.loads((raw / "plan.json").read_text())
    for name, expected in plan["harness_sha256"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError(f"planned harness bytes changed: {name}")
    cases = {
        case: config for case, config in plan["cases"].items() if config["gpu"] == gpu
    }
    with (raw / f"gpu-{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for pair in range(1, plan["pairs"] + 1):
            for case, config in cases.items():
                for position, arm in enumerate(config["orders"][pair - 1], 1):
                    run_arm(raw, plan, case, pair, position, arm)
        write_json(
            raw / f"gpu-{gpu}-complete.json", {"gpu": gpu, "pairs": plan["pairs"]}
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--prepare-from", type=Path)
    sub.add_argument("--gpu", type=int, choices=sorted(set(GPU_BY_CASE.values())))
    args = parser.parse_args()
    if args.prepare_from:
        prepare(args.raw.resolve(), args.prepare_from.resolve())
    else:
        run_lane(args.raw.resolve(), args.gpu)


if __name__ == "__main__":
    main()
