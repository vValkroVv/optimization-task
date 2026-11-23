"""Screen a doubled dW reduction length independently of the CE chunk count."""

import argparse
import json
import os
from pathlib import Path

import torch

from dw_check import FIXTURE, CublasWriter, error, get_operands, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("chunks", type=int, choices=(8, 16))
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    record = torch.load(FIXTURE, map_location="cpu", mmap=True, weights_only=True)["ce"]
    operands = get_operands(record, args.chunks)
    reference = torch.zeros((128256, 4096), dtype=torch.float64, device="cuda")
    for grad, hidden in operands:
        reference.addmm_(grad.double().T, hidden.double())
    paired = [
        (torch.cat((grad_0, grad_1)), torch.cat((hidden_0, hidden_1)))
        for (grad_0, hidden_0), (grad_1, hidden_1) in zip(operands[::2], operands[1::2], strict=True)
    ]
    output = torch.empty_like(reference, dtype=torch.float32)
    writer = CublasWriter()
    stats = []
    for data in (operands, paired, paired, operands):
        def write():
            for index, (grad, hidden) in enumerate(data):
                writer(grad, hidden, output, index != 0)
        write()
        stats.append({"dW_calls": len(data), **error(output, reference), **measure(write)})
    args.output.write_text(json.dumps({
        "chunks": args.chunks,
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "scope": "dW chain CUDA events; BF16 inputs prepared ahead of timing",
        "arms": stats,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
