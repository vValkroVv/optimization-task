"""Profile one native operand dW chain per writer after unprofiled warmup."""

import argparse
import json
from pathlib import Path

import torch

from dw_check import FIXTURE, get_operands
from direct_dw import CublasWriter
from frontier_dw_lib import LtWriter, scratch_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("pair")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.plan.read_text())
    configs = [report["configurations"][int(i)] for i in args.pair.split(".")]
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    record = torch.load(FIXTURE, map_location="cpu", mmap=True, weights_only=True)["ce"]
    operands = get_operands(record, report["chunks"])
    output = torch.empty((report["vocab"], report["width"]), device="cuda", dtype=torch.float32)
    lt = LtWriter(report["rows_per_chunk"], report["vocab"], report["width"])
    assert lt.version == report["cublasLt_version"]
    scratch = scratch_for(*configs)
    plans = [lt.prepare(item, scratch) for item in configs]
    ex = CublasWriter()

    def baseline():
        for i, (grad, hidden) in enumerate(operands):
            ex(grad, hidden, output, i != 0)

    def candidate():
        for i, (grad, hidden) in enumerate(operands):
            lt.apply(plans[i != 0], grad, hidden, output, i != 0)

    for name, fn in (("cublas", baseline), ("lt", candidate)):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
        ) as profiler:
            with torch.profiler.record_function(f"{name}-dw-chain"):
                fn()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(args.output.with_suffix(f".{name}.json")))
    lt.close()


if __name__ == "__main__":
    main()
