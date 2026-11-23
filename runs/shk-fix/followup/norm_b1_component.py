"""Inspect or replay the Hopper norm backward dS and weight-gradient partials."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import torch

from norm_f1_component import load_variant, measure_pairs, profile_kernel
from norm_skip import NATIVE_SHAPE, digest, mismatched_bytes, restore


def compile_b1(module, output):
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_stream

    def build(key, make_op, args):
        if key[0] in ("b1", "b1_gate_layout"):
            return cute.compile(
                make_op(),
                *args,
                make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi --gpu-arch sm_90a --generate-line-info",
            )
        return None

    with patch.object(module, "compile_op", build), patch.object(module, "PLANS", {}):
        b1, _, _, _, chunks, groups, _ = module._bwd_plan(
            4096, 16, 2.0, torch.device("cuda:0"), (torch.float32,) * 3, 116
        )
    if (chunks, groups) != (4, 29) or b1 is None or not b1.__ptx__ or not b1.__cubin__:
        raise RuntimeError(
            "Expected the native four-chunk, 29-group SM90a B1 compilation"
        )
    output.with_suffix(".ptx").write_text(b1.__ptx__)
    output.with_suffix(".cubin").write_bytes(b1.__cubin__)


@torch.no_grad()
def replay(backward, candidate, args):
    from torchtitan.models.common.norms.cute_gated_norm.op import _aligned, _forward

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
    tensors, layouts = record["tensors"], record["layouts"]
    x, weight, up, down = (
        _aligned(restore(tensors[name], layouts[name]))
        for name in ("x", "weight", "up", "down")
    )
    if "dy" in tensors:
        dy = _aligned(restore(tensors["dy"], layouts["dy"]))
        dy_source = "plain"
    else:
        tangent = fixture["first_tangent"]
        dy = _aligned(restore(tangent["dy"], tangent["layout"]))
        dy_source = "first_tangent"
    if any(t.shape != NATIVE_SHAPE or t.dtype != torch.bfloat16 for t in (x, dy)):
        raise RuntimeError("Expected native BF16 batch 2, sequence 4096, hidden 4096")

    gating_weight = float(record["gating_weight"])
    _, rho, p, w_f, gd_bf, gd_t, _ = _forward(
        x, weight, up, down, record["eps"], gating_weight
    )
    hidden, rank = gd_bf.shape
    x, dy = (t.view(-1, hidden) for t in (x, dy))
    rho, p = rho.view(-1), p.view(-1, rank)
    budget = backward.resolve_num_sms(device)
    grad_dtypes = (weight.dtype, up.dtype, down.dtype)
    plans = {
        arm: module._bwd_plan(hidden, rank, gating_weight, device, grad_dtypes, budget)
        for arm, module in (("A", backward), ("B", candidate))
    }
    baseline, variant = plans["A"], plans["B"]
    if (
        baseline[0] is variant[0]
        or any(a is not b for a, b in zip(baseline[1:4], variant[1:4], strict=True))
        or baseline[4:] != variant[4:]
    ):
        raise RuntimeError("Only the B1 kernel may differ between plans")
    chunks, groups, _ = baseline[4:]

    def make_runs(rows):
        partials = backward.workspace(
            device, [(chunks, rows, rank), (chunks, rows), (groups, hidden, rank)]
        )
        common = (x[:rows], dy[:rows], w_f, gd_bf)
        rest = (rho[:rows], p[:rows], *partials)

        def make_run(kernel, inputs):
            def run(_):
                kernel(*inputs)
                return partials

            return run

        return partials, {
            "A": make_run(baseline[0], (*common, gd_t, *rest)),
            "B": make_run(variant[0], (*common, *rest)),
        }

    mismatches = {}
    for rows in (65, x.shape[0]):
        partials, runs = make_runs(rows)
        for tensor in partials:
            tensor.fill_(float("nan"))
        runs["A"](None)
        expected = tuple(t.clone() for t in partials)
        if not all(t.isfinite().all().item() for t in expected):
            raise RuntimeError(
                f"Incomplete or nonfinite baseline partials at {rows} rows"
            )
        for tensor in partials:
            tensor.fill_(float("nan"))
        runs["B"](None)
        mismatches[rows] = mismatched_bytes(
            ("ds_part", "c1_part", "ws_gd"), expected, partials
        )
    del expected

    parity_passed = not any(
        count for case in mismatches.values() for count in case.values()
    )
    pairs, profiles = [], {}
    if parity_passed:
        # The last pair of callables holds the native inputs and preallocated outputs.
        pairs = measure_pairs(runs, args)
        profiles = {
            arm: profile_kernel(runs[arm], expected_name)
            for arm, expected_name in (("A", "GnBwdDs_"), ("B", "GnBwdDsGateLayout_"))
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
                "baseline_sha256": digest(Path(backward.__file__)),
                "candidate_sha256": digest(args.candidate),
                "shape": list(NATIVE_SHAPE),
                "dy_source": dy_source,
                "sm_budget": budget,
                "hidden_chunks": chunks,
                "row_groups": groups,
                "b1_stages": backward.B1_STAGES,
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
    if not parity_passed:
        raise RuntimeError(f"B1 changed native partials: {mismatches}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    if args.fixture and (
        not args.candidate or min(args.pairs, args.warmup, args.samples) < 1
    ):
        parser.error("A fixture replay needs a candidate and positive counts")

    from torchtitan.models.common.norms.cute_gated_norm import backward

    module = (
        load_variant(backward, args.candidate, module_name="backward_b1_gate_layout")
        if args.candidate
        else backward
    )
    if args.fixture:
        replay(backward, module, args)
    else:
        compile_b1(module, args.output)


if __name__ == "__main__":
    main()
