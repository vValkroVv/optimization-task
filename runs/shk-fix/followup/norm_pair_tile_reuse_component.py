"""Compare separate CuTe residual F2 and next F1 against one fused kernel."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import torch

from norm_f1_component import load_variant, measure_pairs
from norm_skip import NATIVE_SHAPE, digest, mismatched_bytes, restore

OUTPUTS = ("residual_output", "rho", "p", "next_p_part", "next_ss_part")


def prepare(prep, record):
    inputs = [
        restore(record["tensors"][name], record["layouts"][name])
        for name in ("up", "weight", "down")
    ]
    up, weight, down = inputs
    w_eff = torch.empty_like(up, dtype=torch.bfloat16)
    w_f = torch.empty_like(weight, dtype=torch.float32)
    gd_bf = torch.empty_like(down, dtype=torch.bfloat16)
    prep(
        *inputs,
        w_eff,
        w_f,
        gd_bf,
        torch.empty_like(up, dtype=torch.bfloat16),
        torch.empty_like(down, dtype=torch.bfloat16),
    )
    return w_eff, w_f, gd_bf


def profile(runs, output, calls):
    profiles = {}
    for arm, expected in (
        ("A", ("GnFwdOut_", "GnFwdStats_")),
        ("B", ("GnFwdOutWithNextStats_",)),
    ):
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as trace:
            for index in range(calls):
                with torch.profiler.record_function(f"norm_pair_tile_call_{index}"):
                    runs[arm](None)
                    torch.cuda.synchronize()
        path = output.with_suffix(f".{arm}.trace.json")
        trace.export_chrome_trace(str(path))
        events = json.loads(path.read_bytes())["traceEvents"]
        kernels = sorted(
            (event for event in events if event.get("cat") == "kernel"),
            key=lambda event: event["ts"],
        )
        annotations = sorted(
            (
                event
                for event in events
                if event.get("cat") == "user_annotation"
                and event["name"].startswith("norm_pair_tile_call_")
            ),
            key=lambda event: event["ts"],
        )
        if len(annotations) != calls or len(kernels) != calls * len(expected):
            raise RuntimeError(
                f"Unexpected profile counts for {arm}: {len(annotations)} calls, "
                f"{len(kernels)} kernels"
            )
        for index, call in enumerate(annotations):
            observed = [
                event["name"]
                for event in kernels
                if call["ts"] <= event["ts"]
                and event["ts"] + event["dur"] <= call["ts"] + call["dur"]
            ]
            if len(observed) != len(expected) or any(
                token not in name
                for token, name in zip(expected, observed, strict=True)
            ):
                raise RuntimeError(
                    f"Unexpected kernels for {arm} call {index}: {observed}"
                )
        profiles[arm] = {
            "trace_sha256": digest(path),
            "kernels": [
                {"name": event["name"], "duration_us": event["dur"]}
                for event in kernels
            ],
        }
    return profiles


@torch.no_grad()
def run(args):
    from torchtitan.models.common.norms.cute_gated_norm import forward

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    budget = forward.resolve_num_sms(device)
    candidate = load_variant(forward, args.candidate, "forward_norm_pair_tile_reuse")
    fixture = torch.load(args.fixture, map_location="cpu", mmap=True, weights_only=True)
    if fixture["torch_revision"] != torch.version.git_version:
        raise RuntimeError("The fixture and replay use different PyTorch revisions")
    first = fixture["records"]["residual"]
    second = fixture["records"]["plain"]
    x, residual = (
        restore(first["tensors"][name], first["layouts"][name])
        for name in ("x", "residual")
    )
    if any(
        t.shape != NATIVE_SHAPE or t.dtype != torch.bfloat16 or not t.is_contiguous()
        for t in (x, residual)
    ):
        raise RuntimeError("Expected the native contiguous BF16 activations")

    hidden = x.shape[-1]
    x = x.view(-1, hidden)
    residual = residual.view(-1, hidden)
    rank = first["tensors"]["up"].shape[0]
    chunks = forward.pick_n_hc(x.shape[0], hidden, budget)
    stages = forward.pick_f2_stages(hidden, chunks, add_residual=True)
    if (hidden, rank, budget, chunks, stages) != (4096, 16, 116, 4, 9):
        raise RuntimeError("The fixture does not select the required native tiling")

    def plan(record, output_chunks, *, add_residual=False):
        return forward._fwd_plan(
            hidden,
            rank,
            tuple(record["tensors"][name].dtype for name in ("up", "weight", "down")),
            record["eps"],
            record["gating_weight"],
            device,
            output_chunks,
            budget,
            add_residual=add_residual,
        )

    prep, f1, f2, splits = plan(first, chunks, add_residual=True)
    next_prep, next_f1, _, next_splits = plan(second, None)
    if splits != 8 or next_splits != splits:
        raise RuntimeError("The two controls must use the native eight-way F1 split")
    w_eff, w_f, gd_bf = prepare(prep, first)
    next_w_eff, _, _ = prepare(next_prep, second)
    fused = candidate.compile_norm_pair(
        hidden, rank, first["eps"], first["gating_weight"], device, budget
    )

    def make_calls(rows):
        source = x[:rows]
        skip = residual[:rows]
        p_part, ss_part, next_p_part, next_ss_part = forward.workspace(
            device,
            [(splits, rows, rank), (splits, rows)] * 2,
        )
        f1(source, w_eff, p_part, ss_part)
        y = torch.empty_like(source)
        rho = torch.empty(rows, device=device, dtype=torch.float32)
        p = torch.empty((rows, rank), device=device, dtype=torch.float32)
        outputs = (y, rho, p, next_p_part, next_ss_part)
        first_args = (source, w_f, gd_bf, p_part, ss_part, y, rho, p, skip)
        next_args = (next_w_eff, next_p_part, next_ss_part)

        def baseline(_):
            f2(*first_args)
            next_f1(y, *next_args)
            return outputs

        def changed(_):
            fused(*first_args, *next_args)
            return outputs

        return {"A": baseline, "B": changed}, outputs

    parity = {}
    for rows in (65, x.shape[0]):
        runs, outputs = make_calls(rows)
        for tensor in outputs:
            tensor.fill_(float("nan"))
        runs["A"](None)
        reference = tuple(t.clone() for t in outputs)
        if not all(t.isfinite().all().item() for t in reference):
            raise RuntimeError(f"Baseline produced incomplete outputs at {rows} rows")
        for tensor in outputs:
            tensor.fill_(float("nan"))
        runs["B"](None)
        parity[rows] = mismatched_bytes(OUTPUTS, reference, outputs)
        if any(parity[rows].values()):
            raise RuntimeError(f"Fused kernel changed outputs at {rows} rows: {parity}")
        del reference

    # Both arms use the same prepared inputs and resident workspaces. This
    # isolates kernel timing and excludes workspace lifetimes across norm calls.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    pairs = measure_pairs(runs, args)
    profiles = profile(runs, args.output, args.profile_calls)
    receipt = {
        "scope": "resident_residual_f2_and_next_f1",
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "torch_revision": torch.version.git_version,
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "fixture_sha256": digest(args.fixture),
        "runner_sha256": digest(Path(__file__)),
        "baseline_sha256": digest(Path(forward.__file__)),
        "candidate_sha256": digest(args.candidate),
        "shape": list(NATIVE_SHAPE),
        "split_k": splits,
        "sm_budget": budget,
        "output_chunks": chunks,
        "input_stages": stages,
        "warmup": args.warmup,
        "samples": args.samples,
        "profile_calls": args.profile_calls,
        "mismatched_bytes": parity,
        "profiles": profiles,
        "pairs": pairs,
    }
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--profile-calls", type=int, default=8)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.pairs, args.profile_calls) < 1:
        parser.error("Replay counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
