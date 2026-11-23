"""Measure Baseline A and the final CE on isolated eight-rank EP8 head replays."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


AREA = Path(__file__).resolve().parent
SCHEDULE = ((2, "baseline"), (2, "final"), (8, "final"), (8, "baseline"),
            (32, "baseline"), (32, "final"))


def nvidia(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(exist_ok=args.resume)
    devices = nvidia("--query-gpu=index,uuid,name", "--format=csv,noheader").splitlines()
    cards = [tuple(part.strip() for part in line.split(",")) for line in devices]
    assert [int(index) for index, _, _ in cards] == list(range(8)), cards
    assert len({uuid for _, uuid, _ in cards}) == 8 and all("H100" in card[2] for card in cards)
    manifest = {
        "host": socket.gethostname(),
        "devices": cards,
        "schedule": [f"c{chunks}-{arm}" for chunks, arm in SCHEDULE],
        "topology": nvidia("topo", "-m"),
    }
    manifest_path = root / "manifest.json"
    if args.resume:
        assert json.loads(manifest_path.read_text())["devices"] == [list(card) for card in cards]
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    cache = AREA.parent / "gather-repeat/cache8"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(card[1] for card in cards),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        TORCHINDUCTOR_CUDAGRAPHS="0",
        TORCHINDUCTOR_MIX_ORDER_REDUCTION="0",
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    for chunks, arm in SCHEDULE:
        case = root / f"c{chunks}-{arm}"
        marker = case / "exit-code"
        if args.resume and marker.is_file() and marker.read_text().strip() == "0":
            assert len(list(case.glob("rank-*.json"))) == 8
            continue
        active = nvidia("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")
        if active:
            raise RuntimeError(f"GPUs in use before {case.name}: {active}")
        if case.exists() and args.resume:
            suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            case.rename(case.with_name(case.name + ".attempt-" + suffix))
        case.mkdir()
        cmd = [
            sys.executable, "-Bu", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc-per-node=8", "--max-restarts=0", str(AREA / "replay_pair.py"),
            "--chunks", str(chunks), "--arm", "regular" if arm == "final" else "baseline",
            "--out", str(case), "--warmup", "10", "--samples", "10",
        ]
        (case / "invocation.json").write_text(json.dumps(cmd, indent=2) + "\n")
        print(datetime.now(timezone.utc).isoformat(), "starting", case.name, flush=True)
        with (case / "run.log").open("w") as log:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600)
        marker.write_text(f"{completed.returncode}\n")
        if completed.returncode:
            raise RuntimeError(f"{case.name} failed, see {case / 'run.log'}")
        assert len(list(case.glob("rank-*.json"))) == 8
        print(datetime.now(timezone.utc).isoformat(), "completed", case.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
