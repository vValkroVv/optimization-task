"""Check all FP32 writers against FP64 on identical, native BF16 GEMM operands."""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

import torch

from replay import CURRENT, FIXTURE

sys.path.insert(0, str(CURRENT / "gift"))

from direct_dw import CublasWriter, TmaWriter
from torchtitan.components.loss import MTPLoss
from torchtitan.distributed.gfsdp._staged_head_dw import merge_head_dw
from torchtitan.experiments.four_fusion.stable_ce import folded_cross_entropy


def get_operands(record, chunks):
    hidden, weight, labels = (record[key].cuda() for key in ("hidden", "weight", "labels"))
    positions = record["positions"].cuda() if record["positions"] is not None else None
    denominator = record["denominator"]
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.cuda()
    y = torch.cat((labels, MTPLoss._shift_labels(labels, 1, positions)))
    data = []
    for h, target in zip(hidden.flatten(0, 1).chunk(chunks, dim=1), y.chunk(chunks, dim=1), strict=True):
        h = h.contiguous().flatten(0, 1)
        logits = torch.nn.functional.linear(h, weight).requires_grad_()
        main, mtp = folded_cross_entropy(logits.reshape(4, -1, logits.size(-1)), target.contiguous(), 2)
        (grad,) = torch.autograd.grad(main / denominator + (mtp / denominator) * 0.3, logits)
        data.append((grad.detach(), h))
    return data


def measure(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_dw_cuda_ms": statistics.median(samples), "samples_dw_cuda_ms": samples}


def error(actual, reference):
    sums = torch.zeros((2,), device="cuda", dtype=torch.float64)
    max_abs = sums.new_zeros(())
    nonfinite = torch.zeros((), device="cuda", dtype=torch.long)
    for output, expected in zip(actual.split(256), reference.split(256), strict=True):
        diff = output.double() - expected
        sums += torch.stack((diff.square().sum(), expected.square().sum()))
        max_abs = torch.maximum(max_abs, diff.abs().max())
        nonfinite += (~output.isfinite()).sum()
    return {
        "relative_l2": float((sums[0] / sums[1]).sqrt()),
        "max_abs": float(max_abs),
        "nonfinite": int(nonfinite),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--writer", choices=("cublas", "tma"), required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    record = torch.load(FIXTURE, map_location="cpu", mmap=True, weights_only=True)["ce"]
    writer = (CublasWriter if args.writer == "cublas" else TmaWriter)()
    results = []
    for chunks in (1, 2, 4, 8, 16):
        operands = get_operands(record, chunks)
        reference = torch.zeros((128256, 4096), dtype=torch.float64, device="cuda")
        for grad, hidden in operands:
            reference.addmm_(grad.double().T, hidden.double())
        output = torch.empty_like(reference, dtype=torch.float32)

        def ordinary():
            for i, (grad, hidden) in enumerate(operands):
                contribution = torch.mm(grad.T, hidden)
                if i == 0:
                    output.copy_(contribution)
                else:
                    output.add_(contribution)

        def staged():
            packed = output.view(torch.bfloat16)
            for i, (grad, hidden) in enumerate(operands):
                torch.mm(grad.T, hidden, out=packed[:, i * 4096:(i + 1) * 4096])
            merge_head_dw[(128256,)](packed, output, 4096, num_warps=4)

        def direct():
            for i, (grad, hidden) in enumerate(operands):
                writer(grad, hidden, output, i != 0)

        methods = {"ordinary": ordinary, args.writer: direct}
        if chunks == 2:
            methods["staged"] = staged
        row = {"chunks": chunks, "rows_per_chunk": 16384 // chunks}
        for name, fn in methods.items():
            fn()
            row[name] = {**error(output, reference), **measure(fn)}
        results.append(row)
        args.output.write_text(json.dumps({"writer": args.writer, "data": results}, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        del operands, reference, output
    assert all(row[args.writer]["nonfinite"] == 0 for row in results)


if __name__ == "__main__":
    main()
