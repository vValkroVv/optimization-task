#!/usr/bin/env python3
"""Observe compiled attention routes in one untimed native optimizer update."""

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


BRIDGE = Path(__file__).resolve().parents[1] / "flash-qla/measure.py"
WORLD_SIZE = 8
ACCUMULATION = 32
TOKEN_SHAPE = (2, 4096)
PROFILE_MICROBATCH = 2
SWITCHES = (
    "GIFT_GDN_DIRECT_FLA_BACKWARD",
    "GIFT_GDN_FUSE_SHORT_CONV_BWD",
    "GIFT_GDN_FUSE_OUTPUT_NORM_FP8",
    "GIFT_MLA_FUSED_ROPE",
    "GIFT_GDN_DIRECT_OUTPUT_NORM_BWD",
)
BRIDGE_ENV = {
    "TASK_FLASHQLA_BACKEND": "fla",
    "TASK_FLASHQLA_CAPTURE_MODE": "disabled",
    "TASK_FLASHQLA_ROUTING_METADATA": "original",
    "TASK_FLASHQLA_CROSS_ENTROPY": "original",
}


def expected_environment(switches):
    return BRIDGE_ENV | {
        flag: str(int(enabled))
        for flag, enabled in zip(SWITCHES, switches, strict=True)
    }


def expected_routes(switches):
    scan, conv, fp8, mla, norm = switches
    return {
        "cpu": {
            "Torch-Compiled Region": True,
            "CompiledFunctionBackward": True,
            "torchtitan::gdn_chunk": True,
            "torchtitan::qkv_conv1d_unpack": True,
            "torchtitan::gdn_fla_backward": scan,
            "ChunkGatedDeltaRuleFunctionBackward": not scan,
            "torchtitan::gdn_short_conv_backward": conv,
            "torchtitan::gated_rmsnorm": not fp8,
            "torchtitan::gdn_output_norm_quantize": fp8,
            "torchtitan::gdn_output_norm_backward": norm,
        },
        "cuda": {
            "chunk_fwd_kernel_o": True,
            # FLA dispatches dqkwg to another backend on supported devices.
            # Both scan routes always launch the native gate-gradient kernel.
            "gdn_gate_bwd_kernel": True,
            "causal_conv1d_bwd_kernel": not conv,
            "_qkv_bwd_pack_kernel": not conv,
            "_gdn_short_conv_backward_kernel": conv,
            "_gdn_output_norm_quantize_128_kernel": fp8,
            "_gdn_output_norm_backward_kernel": norm,
            "_gdn_output_norm_weight_grad_kernel": norm,
            "layer_norm_gated_bwd_kernel": not norm,
            "_mla_rope_pack": mla,
            "_mla_rope_unpack": mla,
        },
    }


def summarize_profile(events, switches):
    names = {"cpu": Counter(), "cuda": Counter()}
    for event in events:
        kind = event.device_type.name.lower()
        if kind in names:
            names[kind][event.name] += 1
    # Compiled frame IDs and Triton-generated names can carry extra text.
    return {
        kind: {
            marker: {
                name: count for name, count in names[kind].items() if marker in name
            }
            for marker in markers
        }
        for kind, markers in expected_routes(switches).items()
    }


def assert_routes(observed, switches, rank):
    mismatches = []
    for kind, markers in expected_routes(switches).items():
        for marker, expected in markers.items():
            names = observed[kind][marker]
            if bool(names) != expected:
                mismatches.append(
                    f"{kind} {marker} expected active={expected}, saw {names}"
                )
    if mismatches:
        raise RuntimeError(
            f"rank {rank} attention dispatch mismatch: {', '.join(mismatches)}"
        )


def related_cuda_events(events):
    markers = (
        "chunk",
        "dqkwg",
        "dhu",
        "gdn",
        "gated",
        "fla",
        "rope",
        "causal_conv",
        "qkv",
    )
    counts = Counter(
        event.name[:256]
        for event in events
        if event.device_type.name == "CUDA"
        and any(marker in event.name.lower() for marker in markers)
    )
    return dict(counts.most_common(32))


def expected_groups(rank, ep):
    first = rank // ep * ep
    return {
        "fsdp": list(range(WORLD_SIZE)),
        "ep": list(range(first, first + ep)),
        "efsdp": list(range(rank % ep, WORLD_SIZE, ep)),
    }


def expected_degrees(ep):
    return {
        "world_size": WORLD_SIZE,
        "dp_shard": 8,
        "dp_replicate": 1,
        "tp": 1,
        "pp": 1,
        "cp": 1,
        "ep": ep,
        "full_dtensor": False,
    }


def mesh_receipt(trainer, rank, ep):
    import torch.distributed as dist

    dims = trainer.parallel_dims
    degrees = {name: getattr(dims, name) for name in expected_degrees(ep)}
    if degrees != expected_degrees(ep):
        raise RuntimeError(f"rank {rank} has unexpected parallel dimensions: {degrees}")
    groups = {}
    for name, expected in expected_groups(rank, ep).items():
        mesh = dims.get_mesh(name)
        members = mesh.mesh.flatten().tolist()
        pg_members = dist.get_process_group_ranks(mesh.get_group())
        if members != expected or pg_members != expected:
            raise RuntimeError(
                f"rank {rank} {name} expected {expected}, found mesh {members}, group {pg_members}"
            )
        groups[name] = members
    return {"degrees": degrees, "groups": groups}


def model_receipt(trainer, switches):
    from torchtitan.models.common.attention.linear.gdn import GatedDeltaNet
    from torchtitan.models.common.attention.softmax.mla import MLA
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    scan, conv, fp8, mla, norm = switches
    gdn = []
    mla_layers = []
    for part in trainer.model_parts:
        for name, module in part.named_modules():
            if isinstance(module, GatedDeltaNet):
                fuse_norm = module._output_norm_fp8
                actual = (
                    module._direct_fla_backward,
                    module._fuse_short_conv_backward,
                    fuse_norm is not None,
                    getattr(fuse_norm, "keywords", {}).get("direct_backward", False),
                )
                if actual != (scan, conv, fp8, norm) or not isinstance(
                    module.o_proj, FP8Linear
                ):
                    raise RuntimeError(
                        f"unexpected GDN and projection selection at {name}: {actual}"
                    )
                gdn.append(name)
            elif isinstance(module, MLA):
                if module._fused_rope != mla:
                    raise RuntimeError(
                        f"unexpected MLA selection at {name}: {module._fused_rope}"
                    )
                mla_layers.append(name)
    if not gdn or not mla_layers:
        raise RuntimeError("native model does not expose both GDN and MLA blocks")
    return {"gdn_fp8_projections": gdn, "mla_layers": mla_layers}


def receipt_path(measurement, rank):
    return measurement / f"linear-activation-rank-{rank:02d}.json"


def write_json(path, payload):
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_receipts(measurement: Path, ep: int, switches: tuple[bool, ...]) -> dict:
    """Require one complete native update, actual mesh, and sampled route evidence."""
    profiled_ranks = list(range(0, WORLD_SIZE, ep))
    ranks = {}
    for rank in range(WORLD_SIZE):
        path = receipt_path(measurement, rank)
        receipt = json.loads(path.read_text())
        expected = {
            "rank": rank,
            "native_update": 1,
            "environment": expected_environment(switches),
            "token_shape": list(TOKEN_SHAPE),
            "microbatches": ACCUMULATION,
            "mesh": {
                "degrees": expected_degrees(ep),
                "groups": expected_groups(rank, ep),
            },
        }
        for field, value in expected.items():
            if receipt[field] != value:
                raise RuntimeError(f"{path}: unexpected {field}: {receipt[field]!r}")
        if rank in profiled_ranks:
            if receipt["profile_microbatch"] != PROFILE_MICROBATCH:
                raise RuntimeError(f"{path}: wrong profiled microbatch")
            assert_routes(receipt["profile"], switches, rank)
        elif receipt["profile"] is not None:
            raise RuntimeError(f"{path}: unexpected profiled rank")
        ranks[str(rank)] = receipt
    return {"profiled_ranks": profiled_ranks, "ranks": ranks}


def run_with_probe(native_main, train_argv, adapter, ep, switches, rank):
    import torch

    if (
        adapter.options["expected_steps"] != 1
        or adapter.options["warmup_updates"] != 0
        or adapter.options["window_updates"] != 0
        or adapter.options["profile_update"] is not None
    ):
        raise RuntimeError("linear-fixes activation requires one untimed native update")

    measurement = Path(adapter.options["output_dir"])
    receipt = {
        "rank": rank,
        "environment": expected_environment(switches),
        "profile": None,
    }
    original = adapter.trainer_class.forward_backward_step
    microbatches = 0

    def forward_backward(trainer, *, input_dict, labels, global_valid_tokens):
        nonlocal microbatches
        microbatches += 1
        if microbatches == 1:
            shape = tuple(input_dict["input"].shape)
            if (
                shape != TOKEN_SHAPE
                or tuple(labels.shape) != TOKEN_SHAPE
                or trainer.gradient_accumulation_steps != ACCUMULATION
                or trainer.config.training.steps != 1
                or trainer.step != 1
                or not trainer.config.compile.enable
                or not {"model", "loss"}.issubset(trainer.config.compile.components)
            ):
                raise RuntimeError(
                    f"rank {rank} activation has unexpected native settings or input {shape}"
                )
            receipt["token_shape"] = shape
            receipt["mesh"] = mesh_receipt(trainer, rank, ep)
            receipt["model"] = model_receipt(trainer, switches)

        kwargs = dict(
            input_dict=input_dict,
            labels=labels,
            global_valid_tokens=global_valid_tokens,
        )
        if microbatches != PROFILE_MICROBATCH or rank % ep:
            return original(trainer, **kwargs)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            loss = original(trainer, **kwargs)
            torch.cuda.synchronize()
        events = profile.events()
        receipt["profile"] = summarize_profile(events, switches)
        receipt["profile_microbatch"] = microbatches
        try:
            assert_routes(receipt["profile"], switches, rank)
        except RuntimeError:
            receipt["related_cuda_events"] = related_cuda_events(events)
            write_json(
                measurement / f"linear-activation-failure-rank-{rank:02d}.json", receipt
            )
            raise
        return loss

    with patch.object(adapter.trainer_class, "forward_backward_step", forward_backward):
        result = native_main(train_argv, adapter)
    if microbatches != ACCUMULATION:
        raise RuntimeError(
            f"rank {rank} completed {microbatches} activation microbatches"
        )
    receipt.update(native_update=1, microbatches=microbatches)
    write_json(receipt_path(measurement, rank), receipt)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-ep", required=True, type=int, choices=(4, 8))
    parser.add_argument("--expected-switches", required=True)
    args, native_args = parser.parse_known_args(argv)
    bits = args.expected_switches
    if len(bits) != len(SWITCHES) or not set(bits) <= {"0", "1"}:
        parser.error("--expected-switches must contain five 0/1 digits")
    switches = tuple(value == "1" for value in bits)
    if switches[4] and not switches[2]:
        parser.error("direct norm backward requires fused FP8 forward")

    ep = args.expected_ep
    rank = int(os.environ["RANK"])
    for name, value in expected_environment(switches).items():
        actual = os.environ.get(name)
        if actual != value:
            raise RuntimeError(f"rank {rank} expected {name}={value!r}, got {actual!r}")

    spec = importlib.util.spec_from_file_location(
        "_linear_fixes_native_measure", BRIDGE
    )
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    original_bootstrap = bridge._bootstrap_and_run

    def bootstrap(train_argv, adapter, native_main):
        def observed_main(config_argv, native_adapter):
            return run_with_probe(
                native_main, config_argv, native_adapter, ep, switches, rank
            )

        return original_bootstrap(train_argv, adapter, observed_main)

    with patch.object(bridge, "_bootstrap_and_run", bootstrap):
        return bridge.main(native_args)


if __name__ == "__main__":
    main()
