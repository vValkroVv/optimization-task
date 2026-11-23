"""Run balanced fresh eight-rank process groups for a fixed-count head comparison."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


AREA = Path(__file__).resolve().parent
CASES = ("baseline", "regular", "alias")


def plan():
    orders = list(itertools.permutations(CASES))
    batch = []
    for block in range(6):
        for chunks in ((2, 8) if block % 2 == 0 else (8, 2)):
            order = orders[(block + (3 if chunks == 8 else 0)) % len(orders)]
            batch.extend((f"b{block}-c{chunks}-{arm}", chunks, arm) for arm in order)
    return batch


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-arm", choices=CASES)
    parser.add_argument("--pilot-chunks", type=int, choices=(2,8), default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(exist_ok=args.resume)
    if args.pilot:
        arms = (args.pilot_arm,) if args.pilot_arm else CASES
        batch = [(f"pilot-c{args.pilot_chunks}-{arm}",args.pilot_chunks,arm) for arm in arms]
    else:
        batch = plan()
    schedule = [name for name,_,_ in batch]
    schedule_path = root/"schedule.json"
    if args.resume:
        assert json.loads(schedule_path.read_text()) == schedule
    else:
        schedule_path.write_text(json.dumps(schedule,indent=2)+"\n")
    cards = subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True
    ).splitlines()
    devices = [c.partition(",")[2].strip() for c in cards]
    assert len(set(devices)) == len(devices) == 8 and [int(c.partition(",")[0]) for c in cards]==list(range(8))
    if not args.resume:
        (root/"topology.txt").write_text(subprocess.check_output(["nvidia-smi","topo","-m"],text=True))
        (root/"devices.txt").write_text("\n".join(cards)+"\n")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=",".join(devices),PYTHONDONTWRITEBYTECODE="1",
             TORCHINDUCTOR_COMPILE_THREADS="1",OMP_NUM_THREADS="2",CUDACHERRY_DISABLE_JIT="1",
             TORCHINDUCTOR_CACHE_DIR=str(AREA/"cache8/inductor"),TRITON_CACHE_DIR=str(AREA/"cache8/triton"),
             FUSED_CE_BINARY="centered.so")
    env.pop("PYTORCH_CUDA_ALLOC_CONF",None)
    for name,chunks,arm in batch:
        dst=root/name
        if args.resume and (dst/"exit-code").is_file() and (dst/"exit-code").read_text().strip()=="0":
            assert len(list(dst.glob("rank-*.json"))) == 8
            continue
        if dst.exists():
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            dst.rename(root/(name+f".attempt-{stamp}"))
        dst.mkdir()
        active=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid",
                                       "--format=csv,noheader"],text=True).strip()
        if active:
            raise RuntimeError(f"Cards in use before {name}: {active}")
        env["PYTORCH_ALLOC_CONF"]="expandable_segments:"+str(arm!="baseline")
        repetitions="2" if args.pilot else "10"
        cmd=[sys.executable,"-Bu","-m","torch.distributed.run","--standalone",
             "--nproc-per-node=8",str(AREA/"replay8.py"),"--chunks",str(chunks),"--arm",arm,
             "--out",str(dst),"--warmup",repetitions,"--samples",repetitions]
        print(datetime.now(timezone.utc).isoformat(),name,flush=True)
        (dst/"invocation.json").write_text(json.dumps({"cmd":cmd,"CUDA_VISIBLE_DEVICES":env["CUDA_VISIBLE_DEVICES"],
                                                      "PYTORCH_ALLOC_CONF":env["PYTORCH_ALLOC_CONF"]},indent=2)+"\n")
        with (dst/"run.log").open("w") as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        (dst/"exit-code").write_text(str(result.returncode)+"\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed: see {dst/'run.log'}")
        assert len(list(dst.glob("rank-*.json"))) == 8
    print("Completed",len(batch),"process groups.",flush=True)


if __name__=="__main__":
    run()
