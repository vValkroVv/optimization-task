#!/usr/bin/env python3
"""Localize native MTP A/A and split VJPs before optimizer amplification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import ffn_components as native


def logical_grad_snapshot(model, tensors, out):
    """Copy activation VJPs before gathering five selected logical gradients."""
    result = {
        "output": out.detach().cpu().clone(),
        "input/hidden": tensors[0].grad.detach().cpu().clone(),
        "input/embedding": tensors[1].grad.detach().cpu().clone(),
    }
    for name, parameter in model.named_parameters():
        name = native.canonical(name)
        if not (
            name.startswith("mtp.layers.0.final_norm.")
            or name.endswith("block.feed_forward.w2.weight")
            or name.endswith("block.attention.wq.weight")
        ):
            continue
        value = parameter.grad
        if value is None:
            raise RuntimeError(f"missing gradient: {name}")
        result["grad/" + name] = native.cpu_tensor(value)
    return result


def attention_cache_evidence(cache_directory):
    evidence = []
    for path in Path(cache_directory).rglob("*.py"):
        source = path.read_text()
        ops = [
            op for op in (
                "_scaled_dot_product_cudnn_attention",
                "_scaled_dot_product_flash_attention",
                "_scaled_dot_product_efficient_attention",
            )
            if "torch.ops.aten." + op + ".default" in source
            or "torch.ops.aten." + op + "_backward.default" in source
        ]
        if ops:
            evidence.append({
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "attention_ops": ops,
            })
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-cache-root", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
    os.environ["TORCHINDUCTOR_MIX_ORDER_REDUCTION"] = "0"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    rank = int(os.environ["RANK"])
    caches = native.configure_compiler_caches(
        args.output, rank, args.compiler_cache_root
    )
    import torch
    import torch.distributed as dist
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.flash_qla.backend import install_backend

    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise RuntimeError("requires eight authorized GPU ranks")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    install_backend("external_gate")
    cfg = native.resolved_config("mtp")
    dims = ParallelDims.from_config(cfg.parallelism, world_size=8)
    dims.build_mesh()
    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "scope": "unchanged production native MTP, 3 independent microbatch VJPs, no optimizer; five selected logical gradients gathered only after activation/VJP CPU copies",
        "seeds": [0, 0, 31],
        "compiler_cache": caches,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_helper_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "results": {},
    }
    baseline = []
    initial = None
    try:
        for label, split in (("A", False), ("AA", False), ("B", True)):
            torch._dynamo.reset()
            cfg = native.resolved_config("mtp", split)
            cfg.dump_folder = str(args.output)
            model = native.build(cfg, "mtp", dims, device)
            weights = native.snapshot(model)
            if initial is None:
                initial = weights
                record["contract"] = native.contract(cfg, model, "mtp")
                record["resolved_config"] = cfg.to_dict()
            else:
                differences = native.errors(initial, weights)
                if any(x["unequal"] for x in differences.values()):
                    raise RuntimeError("initial logical FP32 parameters differ")
            del weights
            comparisons = []
            for index, seed in enumerate((0, 0, 31)):
                model.zero_grad(set_to_none=True)
                tensors = native.inputs(device, rank, seed, "mtp")
                out = native.microbatch(model, "mtp", tensors)
                torch.cuda.synchronize(device)
                current = logical_grad_snapshot(model, tensors, out)
                if label == "A":
                    if index == 1:
                        record["repeat_A"] = native.errors(baseline[0], current)
                    baseline.append(current)
                else:
                    comparisons.append(native.errors(baseline[index], current))
                del current, out, tensors
                print(f"rank={rank} label={label} microbatch={index} complete", flush=True)
            record["results"][label] = comparisons
            record["attention_cache_evidence"] = attention_cache_evidence(
                caches["directories"]["TORCHINDUCTOR_CACHE_DIR"]
            )
            (args.output / f"rank{rank}.json").write_text(
                json.dumps(record, indent=2, default=str) + "\n"
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
