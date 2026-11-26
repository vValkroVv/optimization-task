"""Run an ordered, resumable sequence of fresh eight-rank replays."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT.parents[2]
CONTROL_CE = "17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8"


def run(spec, warmup, samples):
    model, chunks = spec["model"], spec["chunks"]
    operations = spec["operations"]
    out = ROOT / "tuning" / "results" / spec["run"] / f"{model}-c{chunks}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "exit-code").is_file() and (out / "exit-code").read_text() == "0":
        return
    gpu_pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if gpu_pids.strip():
        raise RuntimeError(f"GPUs occupied before {out}: {gpu_pids}")
    so = ROOT / (spec["ce_path"] if "ce" in operations else "native-ce/centered-folded-v1.so")
    digest = hashlib.sha256(so.read_bytes()).hexdigest()
    if digest != (spec["ce_sha"] if "ce" in operations else CONTROL_CE):
        raise RuntimeError(f"Unexpected CE binary {so} {digest}")
    historical = model == "100b" and "dw" not in operations
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    settings = dict(
        PYTHONPATH=os.pathsep.join(map(str, (ROOT / "integrated/gift", ROOT, TASK / ".worktrees/final-all-fix/gift",
                                             TASK.parent / "lm10-c1/gift"))),
        HEAD_REPLAY_DHIDDEN_DIGEST="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
        TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_CUDAGRAPHS="0",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / "cache/head-inductor"), TRITON_CACHE_DIR=str(ROOT / "cache/head-triton"),
        CUDACHERRY_DISABLE_JIT="1", GIFT_DIRECT_BF16_DHIDDEN="1", GIFT_STAGED_HEAD_DW="0",
        GIFT_FUSED_HIDDEN_Z_LOSS=str(int(model == "100b")), GIFT_STABLE_CE_FOLDED_BACKWARD="1",
        GIFT_EXPERIMENT_LM_HEAD_MATRIX="new" if historical else "generic",
        GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS=str(chunks) if historical else "",
        GIFT_EXACT_FOLDED_CE_LIBRARY=str(so), GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256=digest,
    )
    cmd = [sys.executable, "-Bu", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
           "--max-restarts=0", str(ROOT / "tuning/head_run.py"), "--model", model, "--chunks", str(chunks),
           "--fixture", str(ROOT / "inputs" / f"{model}.pt"), "--reference", str(ROOT / "output/reference"),
           "--variant", "cuda13-tuned" if historical else "cuda13", "--out", str(out),
           "--warmup", str(warmup), "--samples", str(samples), "--operations", *operations]
    if {"projection", "dhidden", "linear"}.intersection(operations):
        cmd.extend(["--choices", str(ROOT / spec["choices"])])
    if "dw" in operations:
        cmd.extend(["--plan", str(ROOT / spec["plan"])])
    settings.update(spec.get("settings", {}))
    env.update(settings)
    (out / "launch.json").write_text(json.dumps(dict(spec=spec, cmd=cmd, settings=settings), indent=2))
    print(datetime.now(timezone.utc), "start", out, flush=True)
    with (out / "run.log").open("w") as log:
        code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (out / "exit-code").write_text(str(code))
    if code:
        raise RuntimeError(f"Exit {code} for {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    for spec in json.loads(args.schedule.read_text()):
        run(spec, args.warmup, args.samples)
