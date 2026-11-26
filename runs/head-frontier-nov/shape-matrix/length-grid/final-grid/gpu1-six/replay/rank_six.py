"""Adapt the pinned diagnostic timer to the six-rank GFSDP group."""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType

import torch
import torch.distributed as dist


HERE = Path(__file__).resolve().parent
FINAL_GRID = HERE.parents[1]
PHASE = FINAL_GRID / "c8-diagnosis/phase"
TUNING = FINAL_GRID.parents[1] / "tuning"
WORLD = 6
TIMER_SHA256 = "484bee2f2affee0999268804f314219d6e69829a75bbd38ab4d83607eff220ed"


def record_collective(kind, full, local, group):
    assert dist.get_backend(group) == "nccl"
    assert dist.get_process_group_ranks(group) == list(range(WORLD))
    assert full.dtype == local.dtype and full.numel() == WORLD * local.numel()
    return dict(kind=kind, full_elements=full.numel(), local_elements=local.numel(),
                world=WORLD, dtype=str(full.dtype))


def gradient_receipt(grad, expected_shape, dtype):
    if grad.shape != expected_shape or grad.dtype != dtype or not grad.is_contiguous():
        raise RuntimeError(f"Unexpected gradient: {grad.shape}, {grad.dtype}, {grad.stride()}")
    values = grad.detach().view(-1)
    stride = max(1, (values.numel() - 1) // 2047)
    indices = torch.arange(0, values.numel(), stride, device=grad.device)
    norm = torch.linalg.vector_norm(values, dtype=torch.float32).item()
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"Invalid gradient L2 norm {norm}")
    return dict(shape=list(grad.shape), dtype=str(grad.dtype), l2=norm,
                sampled=values[indices].float().cpu().tolist(), sample_stride=stride)


def write_contract(args, rank, rec, value, h, z, head, local_weight_grad, collectives, warmup):
    from torchtitan.distributed.gfsdp._tensor import RaggedShard

    if value.shape or value.dtype != torch.float32 or not math.isfinite(value.item()):
        raise RuntimeError(f"Unexpected head loss output: {value.shape}, {value.dtype}")
    v, width = rec["weight"].shape
    placement = head.weight.grad.placements
    if len(placement) != 1 or not isinstance(placement[0], RaggedShard):
        raise RuntimeError(f"Unexpected GFSDP gradient placements {placement}")
    if head.weight.grad.shape != (v, width):
        raise RuntimeError(f"Unexpected global dW shape {head.weight.grad.shape}")
    local_shape = (v // WORLD, width)
    local_numel = local_weight_grad.numel()
    (shard,) = placement
    if (v % WORLD or shard.local_numel != local_numel or shard.global_offset != rank * local_numel
            or shard.shard_numels != (local_numel,) * WORLD):
        raise RuntimeError(f"Unexpected dW shard layout {shard}")
    result = dict(
        loss_shape=list(value.shape), loss_dtype=str(value.dtype), denominator=rec["denominator"],
        head_weights=rec["head_weights"], z_loss_coefficient=1.0e-7,
        weight_shape=[v, width], weight_grad_offset=[shard.global_offset // width, 0], collectives=collectives,
        gradients=dict(hidden=gradient_receipt(h.grad, h.shape, torch.bfloat16),
                       z_hidden=gradient_receipt(z.grad, z.shape, z.dtype),
                       weight=gradient_receipt(local_weight_grad, local_shape, torch.float32)),
        warmup_host_utc_ns=warmup,
    )
    (args.out / f"contract-{rank}.json").write_text(json.dumps(result, allow_nan=False) + "\n")


def install_timer():
    source = TUNING / "head_bench.py"
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != TIMER_SHA256:
        raise RuntimeError(f"Unexpected timer SHA-256: {source}")
    program = original.decode("utf-8")
    replacements = {
        'assert rank == int(os.environ["RANK"]) and world == 8':
            'assert rank == int(os.environ["RANK"]) and world == WORLD',
        '    staged, calls = ': '    collectives = []\n    staged, calls = ',
        '            calls["gather"] += 1':
            '            collectives.append(record_collective("all-gather", out, inp, kw["group"]))\n'
            '            calls["gather"] += 1',
        '            calls["scatter"] += 1':
            '            collectives.append(record_collective("sum-reduce-scatter", inp, out, kw["group"]))\n'
            '            calls["scatter"] += 1',
        '    for _ in range(args.warmup):\n        invoke()':
            '    warmup = []\n'
            '    for _ in range(args.warmup):\n'
            '        warmup_start = time.time_ns()\n'
            '        invoke()\n'
            '        warmup.append([warmup_start, time.time_ns()])',
        '    ce_comp = ':
            '    write_contract(args, rank, rec, value, h, z, head, weight_grad, collectives, warmup)\n'
            '    ce_comp = ',
    }
    for before, after in replacements.items():
        if program.count(before) != 1:
            raise RuntimeError(f"Established diagnostic timer marker differs: {before!r}")
        program = program.replace(before, after, 1)

    timer = ModuleType("head_bench")
    timer.__file__ = str(source)
    timer.__dict__.update(WORLD=WORLD, record_collective=record_collective, write_contract=write_contract)
    exec(compile(program, str(source), "exec"), timer.__dict__)
    sys.modules["head_bench"] = timer


if __name__ == "__main__":
    sys.path.insert(0, str(PHASE))
    from profile_rank import main

    install_timer()
    main()
