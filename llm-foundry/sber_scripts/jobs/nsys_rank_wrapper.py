import os
import sys
from pathlib import Path
from sber_scripts.jobs.examples.nsys_train_profiling import USER_LOGIN

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = REPO_ROOT / "scripts" / "train" / "train.py"

job_id = os.environ.get("SLURM_JOB_ID", "nojob")
node_rank = os.environ.get("NODE_RANK", "nonode")
rank = os.environ.get("RANK", "norank")
local_rank = os.environ.get("LOCAL_RANK", "nolocal")

# only rank=0 from each node
if local_rank != "0":
    os.execvp(sys.executable, [sys.executable, str(TRAIN_PY), *sys.argv[1:]])

outdir = Path(f"/mnt/nfs/07-share/{USER_LOGIN}/nsys/{job_id}")
outdir.mkdir(parents=True, exist_ok=True)


cmd = [
    "nsys", "profile",
    "--trace=cuda,cudnn,cublas,nvtx",
    "--sample=none",
    "--cpuctxsw=none",
    "--capture-range=cudaProfilerApi",
    "--capture-range-end=stop",
    "--wait=all",
    "--gpu-metrics-device=all",
    "--force-overwrite=true",
    "-o", str(outdir / f"nsys_node{node_rank}_rank{rank}_local{local_rank}"),
    sys.executable,
    str(TRAIN_PY),
    *sys.argv[1:],
]

os.execvp(cmd[0], cmd)
