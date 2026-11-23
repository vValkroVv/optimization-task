#!/usr/bin/env python3
"""Replay the split MTP transition on eight GPUs with actual GFSDP and MTPLoss."""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


WORLD = 8
PARENT, SOURCE, TRUNK, NORM = "mtp.layers.0", "mtp.layers.0.block.feed_forward", "layers.1", "norm"
OWNERS = (PARENT, SOURCE, TRUNK, NORM)
CASE_NAMES = ("first", "accumulate_repeated", "accumulate_short", "reset_first")
SOURCES = (
    "tests/unit_tests/test_mtp.py", "torchtitan/components/loss.py",
    "torchtitan/distributed/gfsdp/_collectives.py", "torchtitan/distributed/gfsdp/_param_group.py",
    "torchtitan/distributed/gfsdp/_state.py", "torchtitan/distributed/gfsdp/row_owner_embedding.py",
    "torchtitan/models/gift_moe/_base/parallelize.py",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tensor_bytes(tensor, torch):
    data = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": hashlib.sha256(data).hexdigest()}


def source_hashes(gift):
    return {name: hashlib.sha256((gift / name).read_bytes()).hexdigest() for name in SOURCES}


def fixture(gift):
    sys.path[:0] = [str(gift / "contrib/flash-linear-attention"), str(gift / "contrib/flash-qla"), str(gift)]
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from tests.unit_tests.test_mtp import _make_tiny_decoder
    from torchtitan.components.loss import IGNORE_INDEX, MTPLoss
    from torchtitan.config import ActivationCheckpointConfig, CompileConfig
    from torchtitan.distributed.activation_checkpoint import apply_ac
    from torchtitan.distributed.compile import apply_compile
    from torchtitan.distributed.ffn_overlap import enable_ffn_overlap
    from torchtitan.distributed.gfsdp import _param_group
    from torchtitan.models.common.ffn import FeedForward
    from torchtitan.models.gift_moe._base import parallelize

    require(Path(parallelize.__file__).resolve() == gift / SOURCES[-1], "Unexpected GIFT import path")
    return SimpleNamespace(
        torch=torch, dist=dist, mesh=init_device_mesh, decoder=_make_tiny_decoder, loss=MTPLoss,
        ignore=IGNORE_INDEX, ac=apply_ac, ac_config=ActivationCheckpointConfig, compile=apply_compile,
        compile_config=CompileConfig, split=enable_ffn_overlap, ff=FeedForward, pg=_param_group,
        apply_gfsdp=parallelize.apply_gfsdp,
    )


def build(api, dense, edp, transition):
    torch = api.torch
    torch._dynamo.reset()
    torch.manual_seed(2048)
    model = api.decoder(1, n_layers=2).to("cuda")
    mtp = model.mtp.layers["0"]
    widened_ffn = api.ff.Config(dim=mtp.block.feed_forward.w1.in_features, hidden_dim=256)
    mtp.block.feed_forward = widened_ffn.build().to("cuda")
    model.init_weights(buffer_device=torch.device("cuda"))
    api.split(model.layers["0"])
    api.split(mtp)
    api.ac(model, api.ac_config(mode="selective", selective_save=[]), model_compile_enabled=True)
    api.compile(model, api.compile_config(enable=True, backend="aot_eager"))
    # Both arms use the explicit EP8 scheduling branch and the same dense units.
    api.apply_gfsdp(model, dense, param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                    pp_enabled=False, ep_degree=WORLD, edp_mesh=edp, mtp_transition_prefetch=transition)
    model.train()
    model.lazy_init()
    model._skip_lm_head = True
    loss = api.loss.Config(num_chunks=4, loss_weights=[0.3]).build()
    loss.set_lm_head(model.lm_head)
    modules = {name: model.get_submodule(name) for name in (*OWNERS, "lm_head")}
    groups = {}
    for name, module in modules.items():
        (group,) = module._get_fsdp_state()._fsdp_param_groups
        require(group.mesh_info.shard_process_group is dense.get_group() and group.mesh_info.shard_mesh_size == WORLD,
                f"{name}: replay requires the eight-rank dense group")
        groups[name] = group
    return model, loss, modules, groups


def alias_layout(modules, groups):
    donor, trunk, norm = (groups[name] for name in (SOURCE, TRUNK, NORM))
    layout = {"post_backward_target": donor._post_backward_prefetch_target is trunk}
    for name, source, target in (
        ("data", donor.data_buffer_ctx, trunk.data_buffer_ctx),
        ("grad", donor.grad_buffer_ctx, norm.grad_buffer_ctx),
    ):
        contained = (target._ranges.keys() <= source._ranges.keys() and all(
            source._ranges[dtype][0] <= start < end <= source._ranges[dtype][1]
            for dtype, (start, end) in target._ranges.items()
        ))
        layout[name] = {
            "active": target._storage_ctx is source._storage_ctx and target.overlaps(source) and contained,
            "source_bytes": {str(dtype): bounds for dtype, bounds in source._ranges.items()},
            "target_bytes": {str(dtype): bounds for dtype, bounds in target._ranges.items()},
        }
    head_state = modules["lm_head"]._get_fsdp_state()
    states = {modules[name]._get_fsdp_state(): name for name in OWNERS}
    layout["head_prefetch"] = [states.get(state, "other") for state in head_state._states_to_backward_prefetch]
    return layout


def inputs(torch, rank, case, ignore):
    index = 0 if case == "reset_first" else CASE_NAMES.index(case)
    generator = torch.Generator().manual_seed(3100 + 13 * rank + index)
    shape = (1, 4) if index == 2 else (2, 8)
    ids = torch.randint(0, 16, shape, generator=generator)
    if index == 1:
        ids[:, 1::2] = ids[:, :1]
    labels = torch.randint(0, 16, shape, generator=generator)
    labels[:, -1] = ignore
    return ids.cuda(), labels.cuda()


def replay(api, dense, edp, rank, transition):
    torch, dist, pg = api.torch, api.dist, api.pg
    model, loss, modules, groups = build(api, dense, edp, transition)
    layout = alias_layout(modules, groups)
    watched = {groups[name]: name for name in OWNERS}
    reduce_owners = {id(param): name for name, group in groups.items() for param in group.fsdp_params}
    # RaggedShardDTensor.to_local() rejects grad-enabled callers. Read its
    # backing shard just as the existing GFSDP numeric replay does.
    masters = {name: {p._param_fqn: tensor_bytes(p.sharded_param._local_tensor, torch) for p in group.fsdp_params}
               for name, group in groups.items() if name in OWNERS}
    production_ag, production_rs = pg.all_gather, pg.reduce_scatter
    source_post_backward = groups[SOURCE].post_backward
    phase, from_donor, wire, timeline = "", False, [], []

    def donor_post_backward(*args, **kwargs):
        nonlocal from_donor
        from_donor = True
        try:
            return source_post_backward(*args, **kwargs)
        finally:
            from_donor = False

    def observe_ag(group, *args, **kwargs):
        if group not in watched:
            return production_ag(group, *args, **kwargs)
        exchange = dist.all_gather_into_tensor

        def capture(output_tensor, input_tensor, **fields):
            require(not fields.get("async_op") and input_tensor.dtype == output_tensor.dtype == torch.bfloat16,
                    "Expected stream-ordered BF16 gather on the selected scope")
            sent = input_tensor.detach().clone()
            result = exchange(output_tensor, input_tensor, **fields)
            wire.append((watched[group], "gather_bf16", sent, output_tensor.detach().clone()))
            timeline.append([phase, "ag", watched[group], group._training_state.name, from_donor])
            return result

        with patch.object(dist, "all_gather_into_tensor", capture):
            return production_ag(group, *args, **kwargs)

    def observe_rs(params, *args, **kwargs):
        name = reduce_owners.get(id(params[0]))
        if name not in (*OWNERS, "lm_head"):
            return production_rs(params, *args, **kwargs)
        exchange = dist.reduce_scatter_tensor

        def capture(output, input, **fields):
            require(not fields.get("async_op") and input.dtype == output.dtype == torch.float32,
                    "Expected stream-ordered FP32 reduction on the selected scope")
            if name in OWNERS:
                sent = input.detach().clone()
            result = exchange(output, input, **fields)
            if name in OWNERS:
                wire.append((name, "reduce_scatter_fp32", sent, output.detach().clone()))
            timeline.append([phase, "rs", name, "POST_BACKWARD", from_donor])
            return result

        with patch.object(dist, "reduce_scatter_tensor", capture):
            return production_rs(params, *args, **kwargs)

    cases = {}
    with (patch.object(pg, "all_gather", observe_ag), patch.object(pg, "reduce_scatter", observe_rs),
          patch.object(groups[SOURCE], "post_backward", donor_post_backward)):
        for name in CASE_NAMES:
            if name in ("first", "reset_first"):
                model.zero_grad(set_to_none=True)
            ids, labels = inputs(torch, rank, name, api.ignore)
            phase = "forward"
            preds = model(ids)
            phase = "head_chunks"
            value = loss(preds, labels, (labels != api.ignore).sum())
            phase = "decoder_backward"
            value.backward()
            torch.cuda.synchronize()
            collectives = {owner: {"gather_bf16": [], "reduce_scatter_fp32": []} for owner in OWNERS}
            for owner, op, sent, received in wire:
                collectives[owner][op].append([tensor_bytes(sent, torch), tensor_bytes(received, torch)])
            gradients = {}
            for owner in OWNERS:
                gradients[owner] = {}
                for p in groups[owner].fsdp_params:
                    grad = p.sharded_param.grad
                    require(grad is not None and grad.dtype == torch.float32, f"Missing local FP32 grad for {p._param_fqn}")
                    gradients[owner][p._param_fqn] = tensor_bytes(grad._local_tensor, torch)
            cases[name] = {"timeline": timeline.copy(), "bytes": {
                "ids": tensor_bytes(ids, torch), "labels": tensor_bytes(labels, torch),
                "main_output": tensor_bytes(preds[0], torch), "mtp_output": tensor_bytes(preds[1], torch),
                "loss": tensor_bytes(value, torch), "collectives": collectives, "fp32_gradients": gradients,
            }}
            wire.clear()
            timeline.clear()
    return {"layout": layout, "masters": masters, "cases": cases}


def mismatches(control, candidate):
    failures = []
    for transition, arm in ((False, control), (True, candidate)):
        layout = arm["layout"]
        if (layout["data"]["active"], layout["grad"]["active"], layout["post_backward_target"]) != (transition,) * 3:
            failures.append(f"transition={transition}: alias or donor target inactive/miswired")
        if layout["head_prefetch"] != ([PARENT, SOURCE] if transition else [PARENT]):
            failures.append(f"transition={transition}: unexpected head prefetch configuration")
        if arm["cases"]["first"]["bytes"] != arm["cases"]["reset_first"]["bytes"]:
            failures.append(f"transition={transition}: reset did not reproduce first case")
    if control["masters"] != candidate["masters"]:
        failures.append("initial local master bytes differ")
    for case in CASE_NAMES:
        a, b = control["cases"][case], candidate["cases"][case]
        failures.extend(f"{case}: {key} bytes differ" for key in a["bytes"] if a["bytes"][key] != b["bytes"][key])
        for transition, arm in ((False, a), (True, b)):
            if any(not records for ops in arm["bytes"]["collectives"].values() for records in ops.values()):
                failures.append(f"{case}: transition={transition}: missing selected gather/reduce bytes")
        head = ["head_chunks", "rs", "lm_head", "POST_BACKWARD", False]
        parent = ["head_chunks", "ag", PARENT, "PRE_BACKWARD", False]
        ffn = ["head_chunks", "ag", SOURCE, "PRE_BACKWARD", False]
        handoff = ["decoder_backward", "ag", TRUNK, "PRE_BACKWARD", True]
        release = ["decoder_backward", "rs", SOURCE, "POST_BACKWARD", True]
        history = b["timeline"]
        if not (all(event in history for event in (parent, ffn, head, release, handoff))
                and history.index(parent) < history.index(ffn) < history.index(head)
                and history.index(release) < history.index(handoff)):
            failures.append(f"{case}: candidate missed the head-prefetch or donor-release gather")
        if ffn in a["timeline"] or handoff in a["timeline"]:
            failures.append(f"{case}: control used a candidate gather route")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to replace {args.output}")
    gift = (args.source_root / "gift").resolve(strict=True)
    source = source_hashes(gift)
    api = fixture(gift)
    torch, dist = api.torch, api.dist
    rank = int(os.environ["LOCAL_RANK"])
    require(int(os.environ["WORLD_SIZE"]) == int(os.environ["LOCAL_WORLD_SIZE"]) == torch.cuda.device_count() == WORLD
            and int(os.environ["RANK"]) == rank, "Expected one node with eight visible GPUs")
    require(torch.__version__ == "2.15.0.dev20260822+cu130"
            and torch.version.git_version == "8b353065ce260336c5fdfa74d754e8bd9b91b062"
            and tuple(torch.cuda.nccl.version())[:3] == (2, 30, 7), "Unexpected Torch/NCCL replay runtime")
    torch.cuda.set_device(rank)
    device = torch.cuda.get_device_properties(rank)
    require("H100" in device.name and device.multi_processor_count == 132, "Expected an H100 with 132 SMs")
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        dense = api.mesh("cuda", (WORLD,), mesh_dim_names=("fsdp",))
        edp = api.mesh("cuda", (WORLD, 1), mesh_dim_names=("ep", "edp"))["edp"]
        if rank == 0:
            print("MTP transition bitwise: split control", flush=True)
        control = replay(api, dense, edp, rank, False)
        dist.barrier()
        if rank == 0:
            print("MTP transition bitwise: split plus transition", flush=True)
        candidate = replay(api, dense, edp, rank, True)
        require(source == source_hashes(gift), "Source changed during replay")
        rows = [None] * WORLD
        row = {"rank": rank, "device": str(device.uuid), "source": source, "control": control,
               "candidate": candidate, "mismatches": mismatches(control, candidate)}
        dist.all_gather_object(rows, row)
        require(len({row["device"] for row in rows}) == WORLD and all(row["source"] == source for row in rows),
                "Inconsistent sources or GPUs across ranks")
        exact = not any(row["mismatches"] for row in rows)
        if rank == 0:
            report = {
                "schema_version": 1, "exact": exact, "source_root": str(gift), "source": source,
                "torch": torch.__version__, "torch_git": torch.version.git_version,
                "cuda": torch.version.cuda, "nccl": list(torch.cuda.nccl.version()),
                "scope": "Tiny two-layer BF16/FP32 dense Decoder; EP8 scheduling branch, SAC/AOT and four MTPLoss chunks",
                "ranks": rows,
            }
            temp = args.output.with_suffix(f".tmp-{os.getpid()}")
            temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            temp.replace(args.output)
        dist.barrier()
        return int(not exact)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
