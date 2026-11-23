"""Tune BF16-output H100 projection and dHidden on actual LM-head operands."""

import argparse
import ctypes as c
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from frontier_dw_lib import (
    BF16, COMPUTE_FP32, FP32, MAX_WORKSPACE, MIN_ALIGN_A, REDUCTION_MASK,
    TRANS_A, TRANS_B, Heuristic, Int, Layout, LtWriter, Ptr, U64, scratch_for,
)
from replay import CURRENT, FIXTURE

sys.path.insert(0, str(CURRENT / "gift"))
from torchtitan.components.loss import MTPLoss
from torchtitan.experiments.four_fusion.stable_ce import folded_cross_entropy


MIB = 1024**2


class Bf16OutputLt(LtWriter):
    def __init__(self, kind, rows, vocab, width):
        self.kind = kind
        super().__init__(rows, vocab, width)

    def new_layout(self, row_major, rows, vocab, width):
        operation = Ptr()
        self.check(self.desc_create, c.byref(operation), COMPUTE_FP32, FP32)
        if self.kind == "projection":
            self.attribute(self.desc_attr, operation, TRANS_B if row_major else TRANS_A, Int(1))
            if row_major:
                matrices = ((rows, width, width), (vocab, width, width), (rows, vocab, vocab))
            else:
                matrices = ((width, vocab, width), (width, rows, width), (vocab, rows, vocab))
        elif row_major:
            matrices = ((rows, vocab, vocab), (vocab, width, width), (rows, width, width))
        else:
            matrices = ((width, vocab, width), (vocab, rows, vocab), (width, rows, width))
        a, b, cd = (self.matrix(row_major, BF16, *dims) for dims in matrices)
        return Layout(operation, a, b, cd, row_major)

    def candidates(self, order, cap, count=6):
        layout = self.layouts[order]
        preference = Ptr()
        self.check(self.pref_create, c.byref(preference))
        try:
            self.attribute(self.pref_attr, preference, MAX_WORKSPACE, U64(cap))
            # 2 permits FP32 split reduction. Unsplit plans have reduction scheme 0.
            # INPLACE and OUTPUT_TYPE would accumulate partials in BF16.
            self.attribute(self.pref_attr, preference, REDUCTION_MASK, c.c_uint(2))
            for attr in range(MIN_ALIGN_A, MIN_ALIGN_A + 4):
                self.attribute(self.pref_attr, preference, attr, c.c_uint(256))
            results, returned = (Heuristic * count)(), Int()
            self.check(
                self.heuristic, self.handle, layout.operation, layout.a, layout.b, layout.cd,
                layout.cd, preference, count, results, c.byref(returned),
            )
            described = [self.describe(order, entry) for entry in results[:returned.value] if entry.state == 0]
            assert all(entry["config"]["reduction"] in (0, 2) for entry in described), described
            return described
        finally:
            self.check(self.pref_destroy, preference)

    def execute(self, plan, left, weight, output):
        self.apply(plan, left, weight, output, False)


def make_operands(record, chunks):
    tiles = record["h_heads"].chunk(chunks, dim=1)
    targets = record["heads_y"].chunk(chunks, dim=1)
    hidden = tiles[0].contiguous().flatten(0, 1)
    target = targets[0].contiguous()
    logits = torch.nn.functional.linear(hidden, record["weight"]).requires_grad_()
    main, mtp = folded_cross_entropy(logits.reshape(4, -1, logits.size(-1)), target, 2)
    (gradient,) = torch.autograd.grad(main / record["denominator"] + (mtp / record["denominator"]) * 0.3, logits)
    return hidden, gradient.detach(), logits.detach()


def graph(fn):
    fn()
    torch.cuda.synchronize()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        for _ in range(3):
            fn()
    return captured


def event_sample(captured):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    captured.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 3


def eager_samples(fn):
    samples = {"host_submit_us": [], "wall_ms": []}
    for _ in range(4):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        submitted = time.perf_counter()
        torch.cuda.synchronize()
        stopped = time.perf_counter()
        samples["host_submit_us"].append((submitted - start) * 1e6)
        samples["wall_ms"].append((stopped - start) * 1e3)
    return samples


def compare(actual, native, left, right):
    mismatch = torch.zeros((), dtype=torch.int64, device="cuda")
    nonfinite = torch.zeros_like(mismatch)
    max_abs = torch.zeros((), device="cuda")
    for out, expect in zip(actual.split(128), native.split(128), strict=True):
        mismatch += torch.count_nonzero(out.view(torch.int16) != expect.view(torch.int16))
        nonfinite += torch.count_nonzero(~out.isfinite())
        max_abs = torch.maximum(max_abs, (out.float() - expect.float()).abs().max())
    rows = torch.linspace(0, left.shape[0] - 1, 96, device="cuda").long()
    cols = torch.linspace(0, right.shape[1] - 1, 96, device="cuda").long()
    reference = left[rows].double() @ right[:, cols].double()
    rounded = reference.to(torch.bfloat16)
    result = {
        "bits_different": int(mismatch),
        "elements": actual.numel(),
        "nonfinite": int(nonfinite),
        "max_abs_against_native": float(max_abs),
        "oracle_shape": list(reference.shape),
        "oracle": {},
    }
    denom = reference.square().sum().sqrt()
    for name, output in (("native", native), ("candidate", actual)):
        values = output[rows[:, None], cols]
        result["oracle"][name] = {
            "relative_l2": float((values.double() - reference).square().sum().sqrt() / denom),
            "bits_against_rounded_fp64": int(torch.count_nonzero(values.view(torch.int16) != rounded.view(torch.int16))),
        }
    return result


def tune(kind, left, weight, projection, caps, count):
    right = weight.T if kind == "projection" else weight
    rows = left.size(0)
    vocab, width = weight.shape
    assert left.is_contiguous() and weight.is_contiguous()
    expected = projection if kind == "projection" else left @ right
    native = torch.empty_like(expected)
    output = torch.empty_like(expected)
    native_fn = lambda: torch.mm(left, right, out=native)
    native_graph = graph(native_fn)
    control = [event_sample(native_graph) for _ in range(5)]
    writer = Bf16OutputLt(kind, rows, vocab, width)
    configs = {}
    for cap in caps:
        for order in ("col", "row"):
            for config in writer.candidates(order, cap * MIB, count=count):
                configs[(order, config["algo"])] = config
    assert configs, (kind, rows)
    scratch = scratch_for(*configs.values())
    for config in configs.values():
        plan = writer.prepare(config, scratch)
        try:
            captured = graph(lambda: writer.execute(plan, left, weight, output))
            timings = [event_sample(captured) for _ in range(3)]
            config.update(graph_ms=timings, median_graph_ms=statistics.median(timings))
        except RuntimeError as exc:
            config["error"] = str(exc)
    ranked = sorted((c for c in configs.values() if "median_graph_ms" in c), key=lambda c: c["median_graph_ms"])
    assert ranked, configs
    finalists = [ranked[0]]
    zero_workspace = next((c for c in ranked if c["workspace"] == 0), None)
    if zero_workspace is not None and zero_workspace is not ranked[0]:
        finalists.append(zero_workspace)
    for config in finalists:
        plan = writer.prepare(config, scratch)
        selected_fn = lambda: writer.execute(plan, left, weight, output)
        selected_graph = graph(selected_fn)
        a, b = [], []
        for index in range(6):
            arms = ((a, native_graph), (b, selected_graph))
            if index % 2:
                arms = tuple(reversed(arms))
            for series, captured in arms:
                series.append(event_sample(captured))
        config["confirmation"] = {
            "native_graph_ms": a, "candidate_graph_ms": b,
            "native_median_ms": statistics.median(a), "candidate_median_ms": statistics.median(b),
            "native_eager": eager_samples(native_fn), "candidate_eager": eager_samples(selected_fn),
            "numerics": compare(output, expected, left, right),
        }
    writer.close()
    return {"kind": kind, "rows": rows, "native_control_graph_ms": control,
            "cublas_lt_version": writer.version, "candidate_count": len(configs),
            "finalists": finalists, "candidates": list(configs.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--caps-mib", type=int, nargs="+", default=[0, 1, 4, 16, 64])
    parser.add_argument("--heuristics", type=int, default=6)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime_bf16_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    record = torch.load(FIXTURE, map_location="cpu", mmap=True, weights_only=True)["ce"]
    hidden, weight, labels = (record[key].cuda() for key in ("hidden", "weight", "labels"))
    positions = record["positions"].cuda() if record["positions"] is not None else None
    denominator = record["denominator"]
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.cuda()
    assert hidden.dtype == weight.dtype == torch.bfloat16
    record = {
        "weight": weight, "h_heads": hidden.flatten(0, 1),
        "heads_y": torch.cat((labels, MTPLoss._shift_labels(labels, 1, positions))),
        "denominator": denominator,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "source": subprocess.check_output(["git", "-C", str(CURRENT), "rev-parse", "HEAD"], text=True).strip(),
        "fixture": str(FIXTURE),
        "device": torch.cuda.get_device_name(0), "visible": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "native_default_allow_bf16_reduced_precision_reduction": runtime_bf16_reduction,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "workspaces_mib": args.caps_mib,
        "scope": "single warm first fixture chunk; preallocated GEMM output; graph measures three device calls, divided by three; eager submit and synchronized wall separately measured",
        "runs": [],
    }
    for chunks in args.chunks:
        hidden_chunk, grad, projection = make_operands(record, chunks)
        for kind, left in (("projection", hidden_chunk), ("dhidden", grad)):
            result = tune(kind, left, weight, projection, args.caps_mib, args.heuristics)
            result["chunks"] = chunks
            receipt["runs"].append(result)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({k: v for k,v in result.items() if k != "candidates"}), flush=True)
        del hidden_chunk, grad, projection


if __name__ == "__main__":
    main()
