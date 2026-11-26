"""Run each eight-rank head variant in a fresh process, saving failures and receipts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TASK = ROOT.parents[2]
HOME_DIR = TASK.parent
CASES = [("100b", 2), ("100b", 8), ("700b", 8)]
DEFAULT = ["baseline", "cuda13", "folded", "folded-lt", "standalone-fla", "standalone-liger",
           "standalone-cute", "standalone-liger-fp32", "standalone-cute-fp32", "fla-linear-lt",
           "apple", "liger-linear", "fla-linear", "liger-linear-default", "fla-linear-default"]
NATIVE_SHA = "17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8"


def now():
    return datetime.now(timezone.utc).isoformat()


def run(args, model, chunks, arm):
    out = ROOT/"output"/args.series/f"{model}-c{chunks}-{arm}"
    if (out/"exit-code").exists() and (out/"exit-code").read_text().strip() == "0" and len(list(out.glob("rank-*.json"))) == 8:
        return True
    out.mkdir(parents=True, exist_ok=True)
    baseline, historical = arm in ("baseline", "baseline-fla", "baseline-dw"), arm == "folded"
    direct = arm in ("folded-lt", "cuda13", "cuda13-tuned", "fla-linear-lt", "apple-direct") or arm.startswith("standalone-")
    source = (HOME_DIR/"a10-cute-001/gift" if baseline else
              TASK/".worktrees/lm-head-mtp-fix/gift" if historical else ROOT/"integrated/gift")
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("GIFT_EXACT_FOLDED_CE_LIBRARY", None)
    env.pop("GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256", None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
        OMP_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_CUDAGRAPHS="0",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/"cache/head-inductor"), TRITON_CACHE_DIR=str(ROOT/"cache/head-triton"),
        CUDACHERRY_DISABLE_JIT="1", GIFT_DIRECT_BF16_DHIDDEN=str(int(not baseline)),
        GIFT_FUSED_HIDDEN_Z_LOSS=str(int(not baseline and model=="100b")),
        GIFT_STABLE_CE_FOLDED_BACKWARD=str(int(direct or historical)), GIFT_STAGED_HEAD_DW=str(int(historical)),
        GIFT_EXPERIMENT_LM_HEAD_MATRIX="new" if arm=="cuda13-tuned" else "generic" if direct else "",
        GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS=str(chunks) if arm=="cuda13-tuned" else "",
        PYTHONPATH=os.pathsep.join(map(str, (source, ROOT, ROOT/"src/liger/src", ROOT/"src/fla", ROOT/"src/apple",
                                               TASK/".worktrees/final-all-fix/gift", HOME_DIR/"lm10-c1/gift"))))
    if arm.startswith("cuda13"):
        env.update(GIFT_EXACT_FOLDED_CE_LIBRARY=str(ROOT/"native-ce/centered-folded-v1.so"),
                   GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256=NATIVE_SHA)
    if arm == "baseline-dw":
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT/"baseline-dw/gift"), env["PYTHONPATH"]))
        env["GIFT_EXPERIMENT_LM_HEAD_MATRIX"] = "new" if model == "100b" else "generic"
        env["GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS"] = str(chunks) if model == "100b" else ""
    cmd = [sys.executable,"-Bu","-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=8",
           "--max-restarts=0",str(ROOT/"head_bench.py"),"--variant",arm,"--model",model,"--chunks",str(chunks),
           "--fixture",str(ROOT/"inputs"/f"{model}.pt"),"--out",str(out),
           "--warmup",str(args.warmup),"--samples",str(args.samples)]
    if args.oracle:
        cmd += ["--reference",str(ROOT/"output/reference")]
    (out/"invocation.json").write_text(json.dumps(cmd,indent=2))
    print(now(),"starting",out.name,flush=True)
    with (out/"run.log").open("w") as log:
        res = subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/"exit-code").write_text(str(res.returncode))
    print(now(),"exit",res.returncode,out.name,flush=True)
    return res.returncode == 0


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--series",default="head")
    p.add_argument("--arms",nargs="+",default=DEFAULT)
    p.add_argument("--cases",nargs="+",default=["100b-c2","100b-c8","700b-c8"])
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--samples",type=int,default=10)
    p.add_argument("--oracle",action="store_true")
    p.add_argument("--wait",type=int,default=0)
    args=p.parse_args()
    deadline=time.monotonic()+args.wait
    for model,chunks in CASES:
        if f"{model}-c{chunks}" not in args.cases:
            continue
        failures=[]
        for arm in args.arms:
            if arm=="cuda13-tuned" and model=="700b":
                continue
            while subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip():
                if time.monotonic()>deadline:
                    raise RuntimeError("GPUs occupied")
                time.sleep(5)
            if not run(args,model,chunks,arm):
                failures.append(arm)
                if len(failures)>=2 and len(failures)==args.arms.index(arm)+1:
                    raise RuntimeError(f"two failures before useful measurement: {failures}")
