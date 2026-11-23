"""Compare compiled GIFT MLA RoPE implementations on an isolated CUDA device."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import torch
import torch._inductor.config
import triton

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.config.schema.models.common.attention.softmax.gate import (
    SigmoidGateConfig,
)
from torchtitan.config.schema.models.common.norms.gated_norm import GatedRMSNormConfig
from torchtitan.distributed.activation_checkpoint import _apply_op_sac
from torchtitan.models.common.attention.softmax import mla_rope_pack
from torchtitan.models.common.attention.softmax.mla import MLA
from torchtitan.models.common.embeddings.rope.rope import ComplexRoPE
from torchtitan.ops.float8.cuda_gemm import FP8Linear

ROOT = Path(__file__).resolve().parents[2]
SAC_SAVE = "experts_up"
MARKERS = ("_mla_rope_pack", "_mla_rope_unpack")
CROP_REVISION = "fe5950b56de5b2cec7ff4cd063f7338c98200c45"
CROP_SHA256 = "1a5dc6ea665934ec696027ad5c211ad8e8bb1011d400a06284875ed97bf9bf9a"
MIB = 1024 * 1024


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_inputs(fixture_root, seqlen):
    loader = load_module(
        "mla_component_fixtures",
        ROOT / "runs/production-data-abcd/component_fixtures.py",
    )
    ranks = range(seqlen // 4096)
    records, receipts = zip(
        *(loader.load_fixture(fixture_root, rank) for rank in ranks)
    )
    rows = [record["mtp"] for record in records]
    if any(row["attention_masks"] is not None for row in rows):
        raise ValueError(
            "the component experiment requires the captured causal SDPA input"
        )

    def tensor(name):
        values = [row[name] for row in rows]
        value = values[0] if len(values) == 1 else torch.cat(values, dim=1)
        return value.to("cuda", torch.bfloat16)

    hidden, grad = tensor("hidden"), tensor("upstream")
    positions = torch.arange(seqlen, dtype=torch.int32, device="cuda").repeat(2, 1)
    if seqlen == 4096 and rows[0]["positions"] is not None:
        captured = rows[0]["positions"].to(positions)
        if not torch.equal(captured.expand_as(positions), positions):
            raise ValueError(
                "the fixture positions differ from the expected causal arange"
            )
    with torch.device("cuda"):
        cache = ComplexRoPE(
            ComplexRoPE.Config(dim=64, max_seq_len=seqlen, theta=100_000.0)
        ).cache
    receipt_keys = (
        "rank",
        "world_size",
        "source_head",
        "capture_sha256",
        "tensors_sha256",
        "input_ids_int64_sha256",
    )
    provenance = [{key: receipt[key] for key in receipt_keys} for receipt in receipts]
    return (
        (hidden.requires_grad_(), cache, None, positions),
        grad,
        provenance,
        loader.digest,
    )


def build_mla(heads, seed, name, state):
    q_lora = 0 if heads == 32 else 1536
    norm = GatedRMSNormConfig(
        eps=1e-6,
        zero_centered=True,
        gate_rank=16,
        gating_weight=2.0,
        norm_backend="cute",
    )
    fuse_rope = name in ("candidate", "crop")
    with (
        torch.device("cuda"),
        patch.dict(os.environ, {"GIFT_MLA_FUSED_ROPE": str(int(fuse_rope))}),
    ):
        module = MLA.Config(
            dim=4096,
            n_heads=heads,
            q_lora_rank=q_lora,
            kv_norm=norm,
            q_norm=norm._copy() if q_lora else None,
            gate=SigmoidGateConfig(),
            wo=FP8Linear.Config(bias=False),
            use_mla_scaling_factor=True,
        ).build()
    assert module._fused_rope == fuse_rope
    module = module.to(dtype=torch.bfloat16)
    if state is None:
        # Initialize in FP32 and round the sampled matrices to the installed dtype.
        torch.manual_seed(seed)
        with torch.no_grad(), torch.device("cuda"):
            for parameter in module.parameters():
                if parameter.ndim > 1:
                    value = torch.empty(parameter.shape, dtype=torch.float32)
                    parameter.copy_(torch.nn.init.xavier_normal_(value))
                else:
                    parameter.zero_()
    else:
        module.load_state_dict(state, strict=True)
    return module


def make_run(model, name, crop, inputs, incoming):
    wrapped = _apply_op_sac(
        model,
        ActivationCheckpointConfig(selective_save=[SAC_SAVE]),
        save=frozenset({SAC_SAVE}),
    )
    compiled = torch.compile(wrapped, backend="inductor", fullgraph=True, dynamic=False)
    parameters = tuple(model.parameters())
    labels = ("output", "input", *(name for name, _ in model.named_parameters()))

    def run():
        output = compiled(*inputs)
        grads = torch.autograd.grad(output, (inputs[0], *parameters), incoming)
        return output, *grads

    def kernel_context():
        if name == "crop":
            return patch.object(mla_rope_pack, "MLARoPEPack", crop.MLARoPEPack)
        return nullcontext()

    return run, kernel_context, labels


def allocate(run):
    torch.cuda.synchronize()
    # Isolate allocator reservation from the previous variant's cached blocks.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    result = run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    reserved_peak = torch.cuda.max_memory_reserved()
    del result
    return {
        "before_mib": before / MIB,
        "peak_mib": peak / MIB,
        "delta_mib": (peak - before) / MIB,
        "before_reserved_mib": reserved_before / MIB,
        "peak_reserved_mib": reserved_peak / MIB,
        "delta_reserved_mib": (reserved_peak - reserved_before) / MIB,
    }


def timed(run):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    result = run()
    end.record()
    submitted = time.perf_counter()
    end.synchronize()
    completed = time.perf_counter()
    del result
    return {
        "cuda_ms": start.elapsed_time(end),
        "wall_ms": (completed - t0) * 1e3,
        "host_submission_ms": (submitted - t0) * 1e3,
        "host_wait_ms": (completed - submitted) * 1e3,
    }


def check_pair(actual, expected):
    # Chunk the FP32 reduction to cap host memory for full projection gradients.
    squares = ref_squares = maximum = 0.0
    count = 0
    finite = True
    for a, b in zip(actual.flatten().split(1 << 20), expected.flatten().split(1 << 20)):
        a, b = a.float(), b.float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            finite = False
            break
        d = a - b
        squares += d.square().sum().item()
        ref_squares += b.square().sum().item()
        maximum = max(maximum, d.abs().max().item())
        count += d.count_nonzero().item()
    return {
        "finite": finite,
        "elements": actual.numel(),
        "mismatches": count if finite else None,
        "max_abs": maximum if finite else None,
        "rmse": (squares / actual.numel()) ** 0.5 if finite else None,
        "relative_l2": (squares / max(ref_squares, 1e-30)) ** 0.5 if finite else None,
    }


def benchmark(runs, warmup, samples):
    for _ in range(warmup):
        for run, context, _ in runs.values():
            with context():
                run()
        torch.cuda.synchronize()
    names = list(runs)
    timings = {name: [] for name in names}
    order = []
    for index in range(samples):
        cycle = names if (index // len(names)) % 2 == 0 else names[::-1]
        offset = index % len(names)
        sequence = cycle[offset:] + cycle[:offset]
        order.append(sequence)
        for name in sequence:
            run, context, _ = runs[name]
            with context():
                timings[name].append(timed(run))
    memory = {}
    for name, (run, context, _) in runs.items():
        with context():
            memory[name] = allocate(run)
    return {
        "sample_order": order,
        "memory_method": "separate synchronized call after freeing unused allocator cache",
        "variants": {
            name: {
                "memory": memory[name],
                "samples": rows,
                "median_cuda_ms": statistics.median(row["cuda_ms"] for row in rows),
                "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
                "median_host_submission_ms": statistics.median(
                    row["host_submission_ms"] for row in rows
                ),
                "median_host_wait_ms": statistics.median(
                    row["host_wait_ms"] for row in rows
                ),
            }
            for name, rows in timings.items()
        },
    }


def profile(runs, output):
    records = {}
    for name, (run, context, _) in runs.items():
        with context():
            run()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                run()
                torch.cuda.synchronize()
        path = output.with_name(f"{output.stem}.{name}.trace.json")
        prof.export_chrome_trace(str(path))
        kernels = [
            event
            for event in json.loads(path.read_text())["traceEvents"]
            if event.get("cat") == "kernel"
        ]
        markers = {
            marker: {
                "count": sum(marker in event["name"] for event in kernels),
                "us": sum(event["dur"] for event in kernels if marker in event["name"]),
            }
            for marker in MARKERS
        }
        records[name] = {
            "trace": str(path),
            "kernels": len(kernels),
            "rope_kernels": markers,
            "attention_kernels": sorted(
                {
                    event["name"]
                    for event in kernels
                    if any(
                        part in event["name"].lower()
                        for part in ("cudnn", "flash", "attention", "fmha")
                    )
                }
            ),
            "activation_ok": bool(kernels)
            and all(
                bool(row["count"]) == (name in ("candidate", "crop"))
                for row in markers.values()
            ),
        }
    return records


def parity(runs, limit):
    def observe(variant):
        run, context, labels = runs[variant]
        with context():
            result = tuple(tensor.detach().cpu() for tensor in run())
        return dict(zip(labels, result))

    reference = observe("baseline")
    results = {}
    candidate = None
    for name in runs:
        if name == "baseline":
            continue
        print(f"Comparing {name} with baseline", flush=True)
        observed = observe(name)
        results[name] = {
            label: check_pair(observed[label], expected)
            for label, expected in reference.items()
        }
        if name == "candidate" and "crop" in runs:
            candidate = observed
        if name == "crop" and candidate is not None:
            results["candidate_vs_crop"] = {
                label: check_pair(candidate[label], old)
                for label, old in observed.items()
            }
    control = results.get("control", {})
    for name, tensors in results.items():
        for label, row in tensors.items():
            control_error = control.get(label, {}).get("relative_l2") or 0.0
            tolerance = limit if name == "control" else max(limit, 2 * control_error)
            row["relative_l2_limit"] = tolerance
            row["passed"] = row["finite"] and row["relative_l2"] <= tolerance
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--input-fixtures", type=Path, required=True)
    parser.add_argument("--mode", choices=("check", "timing", "profile"), required=True)
    parser.add_argument(
        "--variant",
        choices=("paired", "baseline", "candidate", "crop"),
        default="paired",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-kernel", type=Path)
    parser.add_argument("--seed", type=int, default=19321)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--relative-l2-limit", type=float, default=0.01)
    args = parser.parse_args()
    if args.crop_kernel and args.seqlen != 4096:
        parser.error("--crop-kernel is restricted to S4096")
    if args.variant == "crop" and not args.crop_kernel:
        parser.error("--variant crop requires --crop-kernel")
    if args.mode == "check" and args.variant == "baseline":
        parser.error("use --variant paired to check the identical control")
    if min(args.warmup, args.samples) < 1 or args.relative_l2_limit <= 0:
        parser.error("warmup, samples, and relative L2 limit must be positive")

    torch._inductor.config.triton.cudagraphs = False
    torch.set_num_threads(1)
    heads = 32 if args.seqlen == 4096 else 64
    inputs, incoming, fixtures, digest = load_inputs(args.input_fixtures, args.seqlen)
    crop_metadata = None
    if args.crop_kernel:
        sha256 = digest(args.crop_kernel)
        if sha256 != CROP_SHA256:
            raise ValueError(f"crop kernel does not match {CROP_REVISION}: {sha256}")
        crop_metadata = {
            "path": str(args.crop_kernel.resolve()),
            "sha256": sha256,
            "revision": CROP_REVISION,
        }
    crop = (
        load_module("mla_component_historical_crop", args.crop_kernel)
        if args.crop_kernel
        else None
    )
    if args.variant == "paired":
        names = ["baseline", "control", "candidate"]
        if crop is not None:
            names.append("crop")
    elif args.mode == "check":
        names = ["baseline", "control", args.variant]
    else:
        names = [args.variant]
    state = None
    runs = {}
    for name in names:
        model = build_mla(heads, args.seed, name, state)
        if state is None:
            state = model.state_dict()
        runs[name] = make_run(model, name, crop, inputs, incoming)
    del state
    report = {
        "command": sys.argv,
        "gift_revision": subprocess.check_output(
            ["git", "-C", str(ROOT / "gift"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "current_kernel_sha256": digest(Path(mla_rope_pack.__file__)),
        "harness_sha256": digest(Path(__file__)),
        "crop_kernel": crop_metadata,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "shape": {
            "batch": 2,
            "sequence": args.seqlen,
            "heads": heads,
            "model_dim": 4096,
        },
        "q_lora_rank": 0 if heads == 32 else 1536,
        "dtype": "bfloat16",
        "inputs": "captured MTP hidden/upstream; rank sequences concatenated at 8K; causal arange",
        "weights": "shared seeded Xavier matrices and zero vectors",
        "mode": args.mode,
        "seed": args.seed,
        "fixture_root": str(args.input_fixtures.resolve()),
        "fixture_receipts": fixtures,
        "selective_activation_checkpoint": f"save:{SAC_SAVE}",
        "compile": {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "cudagraphs": False,
        },
        "rope_flag": {name: int(name in ("candidate", "crop")) for name in runs},
    }
    print(f"MLA {args.mode} B2 S{args.seqlen} H{heads}: {', '.join(names)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    if args.mode == "check":
        rows = parity(runs, args.relative_l2_limit)
        failed = any(
            not row["passed"] for tensors in rows.values() for row in tensors.values()
        )
    elif args.mode == "profile":
        rows = profile(runs, args.output)
        failed = any(not row["activation_ok"] for row in rows.values())
    else:
        rows = benchmark(runs, args.warmup, args.samples)
    report["results"] = rows
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if failed:
        raise SystemExit(f"{args.mode} failed; see {args.output}")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
