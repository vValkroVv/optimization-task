"""Inspect or replay the scalar and vector Hopper norm statistics kernels."""

import argparse
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import torch


def load_variant(original, source, module_name="forward_f1_vector"):
    spec = importlib.util.spec_from_file_location(
        f"{original.__package__}.{module_name}", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PLANS = {}
    return module


def compile_f1(module, output):
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_stream

    def build(key, make_op, args):
        if key[0] in ("f1", "f1_vector"):
            return cute.compile(
                make_op(),
                *args,
                make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi --gpu-arch sm_90a --generate-line-info",
            )
        return None

    with patch.object(module, "compile_op", build), patch.object(module, "PLANS", {}):
        _, f1, _, split = module._fwd_plan(
            4096, 16, (torch.float32,) * 3, 1e-5, 1.0, torch.device("cuda:0"), None, 116
        )
    if split != 8 or not f1.__ptx__ or not f1.__cubin__:
        raise RuntimeError("Expected the native eight-split SM90a F1 compilation")
    output.with_suffix(".ptx").write_text(f1.__ptx__)
    output.with_suffix(".cubin").write_bytes(f1.__cubin__)


def profile_kernel(run, expected):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as trace:
        for _ in range(8):
            run(None)
            torch.cuda.synchronize()
    kernels = [k for k in trace.events() if k.device_type.name == "CUDA"]
    if len(kernels) != 8 or any(expected not in k.name for k in kernels):
        raise RuntimeError(f"Expected {expected}, got {[k.name for k in kernels]}")
    return [{"name": k.name, "duration_us": k.time_range.elapsed_us()} for k in kernels]


def measure_pairs(runs, args):
    from dense_weight_scale import measure
    from torchtitan.tools.utils import device_memory_usage

    for run in runs.values():
        for _ in range(args.warmup):
            run(None)
        torch.cuda.synchronize()
    pairs = []
    for index in range(args.pairs):
        order = ("A", "B") if index % 2 == 0 else ("B", "A")
        blocks = {}
        for arm in order:
            block = measure(runs[arm], None, args.samples)
            if block["allocator_retries"]:
                raise RuntimeError(
                    f"Unexpected allocation retry while timing arm {arm}"
                )
            _, block["device_peak_gib"] = device_memory_usage(torch.cuda.memory_stats())
            blocks[arm] = block
        pairs.append({"order": order, "arms": blocks})
    return pairs


@torch.no_grad()
def replay(forward, candidate, args):
    from norm_skip import NATIVE_SHAPE, digest, mismatched_bytes, restore
    from torchtitan.models.common.norms.cute_gated_norm.op import _aligned

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    fixture = torch.load(args.fixture, map_location="cpu", mmap=True, weights_only=True)
    if fixture["torch_revision"] != torch.version.git_version:
        raise RuntimeError(
            "The native fixture and replay use different PyTorch revisions"
        )
    record = fixture["records"]["plain"]
    x, weight, up, down = (
        _aligned(restore(record["tensors"][name], record["layouts"][name]))
        for name in ("x", "weight", "up", "down")
    )
    if x.shape != NATIVE_SHAPE or x.dtype != torch.bfloat16:
        raise RuntimeError("Expected native BF16 batch 2, sequence 4096, hidden 4096")
    hidden, rank = x.shape[-1], up.shape[0]
    x = x.view(-1, hidden)
    budget = forward.resolve_num_sms(device)
    plans = {
        arm: module._fwd_plan(
            hidden,
            rank,
            (up.dtype, weight.dtype, down.dtype),
            record["eps"],
            record["gating_weight"],
            device,
            None,
            budget,
        )
        for arm, module in (("A", forward), ("B", candidate))
    }
    prep, _, final, split = plans["A"]
    if (
        plans["B"][0] is not prep
        or plans["B"][2] is not final
        or plans["B"][3] != split
    ):
        raise RuntimeError("Only the first norm statistics kernel can differ")
    w_eff = torch.empty_like(up, dtype=torch.bfloat16)
    prep(
        up,
        weight,
        down,
        w_eff,
        torch.empty_like(weight, dtype=torch.float32),
        torch.empty_like(down, dtype=torch.bfloat16),
        torch.empty_like(up, dtype=torch.bfloat16),
        torch.empty_like(down, dtype=torch.bfloat16),
    )

    def scratch(rows):
        return forward.workspace(device, [(split, rows, rank), (split, rows)])

    mismatches = {}
    for rows in (65, x.shape[0]):
        partials = scratch(rows)
        for tensor in partials:
            tensor.fill_(float("nan"))
        plans["A"][1](x[:rows], w_eff, *partials)
        expected = tuple(t.clone() for t in partials)
        if not all(t.isfinite().all().item() for t in expected):
            raise RuntimeError(
                f"Incomplete or nonfinite baseline partials at {rows} rows"
            )
        for tensor in partials:
            tensor.fill_(float("nan"))
        plans["B"][1](x[:rows], w_eff, *partials)
        mismatches[rows] = mismatched_bytes(("p_part", "ss_part"), expected, partials)
    del expected
    if any(count for result in mismatches.values() for count in result.values()):
        raise RuntimeError(f"F1 changed native partials: {mismatches}")

    partials = scratch(x.shape[0])

    def make_run(kernel):
        def run(_):
            kernel(x, w_eff, *partials)
            return partials

        return run

    runs = {arm: make_run(plan[1]) for arm, plan in plans.items()}
    pairs = measure_pairs(runs, args)
    profiles = {
        arm: profile_kernel(runs[arm], expected)
        for arm, expected in (("A", "GnFwdStats_"), ("B", "GnFwdStatsVector_"))
    }
    with args.output.open("x") as handle:
        json.dump(
            {
                "hostname": platform.node(),
                "gpu": torch.cuda.get_device_name(device),
                "torch_revision": torch.version.git_version,
                "cutlass_dsl": version("nvidia-cutlass-dsl"),
                "fixture_sha256": digest(args.fixture),
                "runner_sha256": digest(Path(__file__)),
                "baseline_sha256": digest(Path(forward.__file__)),
                "candidate_sha256": digest(args.candidate),
                "shape": list(NATIVE_SHAPE),
                "split_k": split,
                "sm_budget": budget,
                "warmup": args.warmup,
                "samples": args.samples,
                "native_partials_mismatched_bytes": mismatches,
                "profiles": profiles,
                "pairs": pairs,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=8)
    args = parser.parse_args()
    if args.fixture and (
        not args.candidate or min(args.warmup, args.samples, args.pairs) < 1
    ):
        parser.error("A fixture replay needs a candidate and positive counts")

    from torchtitan.models.common.norms.cute_gated_norm import forward

    module = load_variant(forward, args.candidate) if args.candidate else forward
    if args.fixture:
        replay(forward, module, args)
    else:
        compile_f1(module, args.output)


if __name__ == "__main__":
    main()
