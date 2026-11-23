"""Check FP64 error if the native FP32 gradient is accumulated in one slot."""

import argparse
import json
from pathlib import Path
import time

import torch

from dw_check import FIXTURE, CublasWriter, error, get_operands


def measure(chunks, case, data, reference, writer):
    steps = (1, 8, 32)
    factors = [1.0] * steps[-1] if case == "repeat" else [1.0, 0.5, -1.0, 2.0] * (steps[-1] // 4)
    tensors = {1.0: data}
    for factor in set(factors) - {1.0}:
        tensors[factor] = [(grad * factor, hidden) for grad, hidden in data]
    result = {}
    for layout in ("native_association", "one_slot"):
        persistent = torch.empty_like(reference, dtype=torch.float32)
        scratch = torch.empty_like(persistent) if layout == "native_association" else None
        output = scratch if scratch is not None else persistent
        records = []
        for step, factor in enumerate(factors, 1):
            for index, (grad, hidden) in enumerate(tensors[factor]):
                add = index > 0 or (layout == "one_slot" and step > 1)
                writer(grad, hidden, output, add)
            if scratch is not None:
                if step == 1:
                    persistent.copy_(scratch)
                else:
                    persistent.add_(scratch)
            if step in steps:
                expected = reference * sum(factors[:step])
                records.append({"step": step, **error(persistent, expected)})
                del expected
        result[layout] = records
        del persistent, scratch, output
    return {"chunks": chunks, "cases": case, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("chunks", nargs="+", type=int, choices=(2, 4, 8, 16))
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    record = torch.load(FIXTURE, map_location="cpu", mmap=True, weights_only=True)["ce"]
    writer = CublasWriter()
    outcomes = []
    for chunks in args.chunks:
        start = time.monotonic()
        operands = get_operands(record, chunks)
        reference = torch.zeros((128256, 4096), dtype=torch.float64, device="cuda")
        for grad, hidden in operands:
            reference.addmm_(grad.double().T, hidden.double())
        for case in ("repeat", "signed_scales"):
            outcomes.append(measure(chunks, case, operands, reference, writer))
        del operands, reference
        args.output.write_text(json.dumps(outcomes, indent=2) + "\n")
        print(chunks, time.monotonic() - start, flush=True)


if __name__ == "__main__":
    main()
