"""Run the pinned isolated-head timer on six ranks and record UTC per sample."""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType


ARRIVAL_UTC_NS = time.time_ns()
FINAL_GRID = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(FINAL_GRID / "gpu1-six/replay"))
from rank_six import PHASE, TIMER_SHA256, TUNING, WORLD, record_collective

sys.path.insert(0, str(PHASE))
import profile_rank
from replay import write_json
import torch


current_phase = {"phase": "setup"}


def phase(name, index=None):
    current_phase.update(phase=name, index=index)
    return time.time_ns()


def write_contract(args, rank, rec, value, h, z, head, local_dw, collectives, warmup, measured_arrivals):
    from torchtitan.distributed.gfsdp._tensor import RaggedShard

    vocab, width = rec["weight"].shape
    placement = head.weight.grad.placements
    if len(placement) != 1 or not isinstance(placement[0], RaggedShard):
        raise RuntimeError(f"Unexpected GFSDP gradient placements: {placement}")
    (shard,) = placement
    local_numel = local_dw.numel()
    if (vocab % WORLD or head.weight.grad.shape != (vocab, width)
            or shard.local_numel != local_numel or shard.global_offset != rank * local_numel
            or shard.shard_numels != (local_numel,) * WORLD):
        raise RuntimeError(f"Unexpected dW shard layout: {shard}")
    if value.shape or value.dtype != torch.float32 or not math.isfinite(value.item()):
        raise RuntimeError(f"Unexpected loss output: {value.shape}, {value.dtype}")
    if h.grad.shape != h.shape or (z is not None and (z.grad.shape != z.shape or z.grad.dtype != z.dtype)):
        raise RuntimeError("The head backward returned an unexpected input-gradient shape or dtype")
    write_json(args.out / f"contract-{rank}.json", dict(
        process_arrived_utc_ns=ARRIVAL_UTC_NS, loss_dtype=str(value.dtype),
        denominator=rec["denominator"], head_weights=rec["head_weights"],
        z_loss_coefficient=1.0e-7 if z is not None else None,
        weight_shape=[vocab, width], weight_grad_offset=[shard.global_offset // width, 0],
        hidden_grad_shape=list(h.grad.shape), hidden_grad_dtype=str(h.grad.dtype),
        z_hidden_grad_shape=list(z.grad.shape) if z is not None else None,
        z_hidden_grad_dtype=str(z.grad.dtype) if z is not None else None,
        weight_grad_shape=list(local_dw.shape), weight_grad_dtype=str(local_dw.dtype),
        collectives=collectives, warmup_host_utc_ns=warmup,
        sample_barrier_arrived_utc_ns=measured_arrivals,
    ))


def install_timer():
    source = TUNING / "head_bench.py"
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != TIMER_SHA256:
        raise RuntimeError(f"Unexpected timer SHA-256: {source}")
    program = original.decode("utf-8")
    replacements = {
        'assert rank == int(os.environ["RANK"]) and world == 8':
            'assert rank == int(os.environ["RANK"]) and world == WORLD',
        '    h = rec["hidden"].cuda()': '    phase("head_setup")\n    h = rec["hidden"].cuda()',
        '    staged, calls = ': '    phase("collective_smoke")\n    collectives = []\n    staged, calls = ',
        '            calls["gather"] += 1':
            '            collectives.append(record_collective("all-gather", out, inp, kw["group"]))\n'
            '            calls["gather"] += 1',
        '            calls["scatter"] += 1':
            '            collectives.append(record_collective("sum-reduce-scatter", inp, out, kw["group"]))\n'
            '            calls["scatter"] += 1',
        '    for _ in range(args.warmup):\n        invoke()':
            '    warmup = []\n'
            '    for index in range(1, args.warmup + 1):\n'
            '        warmup_start = phase("warmup", index)\n'
            '        invoke()\n'
            '        warmup.append([warmup_start, time.time_ns()])\n'
            '    phase("allocator_prime")',
        '    for _ in range(args.samples):\n        dist.barrier()':
            '    measured_arrivals = []\n'
            '    for index in range(1, args.samples + 1):\n'
            '        measured_arrivals.append(phase("measurement", index))\n'
            '        dist.barrier()',
        '    value = invoke(reset=True)': '    phase("final_backward")\n    value = invoke(reset=True)',
        '    ce_comp = ':
            '    phase("verification")\n'
            '    write_contract(args, rank, rec, value, h, z, head, weight_grad, collectives, warmup, measured_arrivals)\n'
            '    ce_comp = ',
    }
    for before, after in replacements.items():
        if program.count(before) != 1:
            raise RuntimeError(f"Established timer marker differs: {before!r}")
        program = program.replace(before, after, 1)
    timer = ModuleType("head_bench")
    timer.__file__ = str(source)
    timer.__dict__.update(WORLD=WORLD, record_collective=record_collective, write_contract=write_contract, phase=phase)
    exec(compile(program, str(source), "exec"), timer.__dict__)
    sys.modules["head_bench"] = timer


def main():
    rank = int(os.environ["RANK"])
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    install_timer()
    try:
        profile_rank.main()
    except torch.OutOfMemoryError as error:
        if "cuda out of memory" in str(error).lower():
            path = out / f"oom-{rank}.json"
            base = json.loads(path.read_text()) if path.exists() else {}
            stage = current_phase["phase"] if base.get("phase") == "head-replay" else base.get("phase", "setup")
            write_json(path, dict(rank=rank, phase=stage, iteration=current_phase.get("index"),
                                  error=str(error), occurred_utc_ns=time.time_ns(),
                                  process_arrived_utc_ns=ARRIVAL_UTC_NS))
        raise


if __name__ == "__main__":
    main()
