"""Run the unmodified native replay with a benchmark-only dW writer selection."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

import replay
from frontier_dw_lib import LtWriter, scratch_for


def compare_dump(dump, reference, destination):
    from dw_check import error

    record = {"dw_fp64": error(dump.pop("dw").cuda(), reference)}
    for key, tensor in dump.items():
        bytes_ = tensor.contiguous().view(torch.uint8) if tensor.ndim else tensor.reshape(1).view(torch.uint8)
        record[key] = hashlib.sha256(memoryview(bytes_.numpy()).cast("B")).hexdigest()
    destination.write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("cublas", "lt"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--pair")
    parser.add_argument("--numerics", action="store_true")
    args = parser.parse_args()
    tuning = json.loads(args.plan.read_text())
    chunks = tuning["chunks"]
    if args.arm == "lt":
        indices = [int(i) for i in args.pair.split(".")]
        configs = [tuning["configurations"][i] for i in indices]

    for name, enabled in zip(replay.SWITCHES, replay.VARIANTS["current"], strict=True):
        os.environ[name] = str(int(enabled))
    sys.path.insert(0, str(replay.CURRENT / "gift"))
    import direct_dw

    base = direct_dw.CublasWriter
    writer = None
    warmup, samples = 10, 10
    previous_invocations = 1 + warmup + 1 + samples

    class ObservedWriter:
        def __init__(self):
            nonlocal writer
            writer = self
            self.calls = 0
            self.reference = None
            if args.arm == "lt":
                self.lt = LtWriter(tuning["rows_per_chunk"], tuning["vocab"], tuning["width"])
                assert self.lt.version == tuning["cublasLt_version"]
                scratch = scratch_for(*configs)
                self.plans = [self.lt.prepare(item, scratch) for item in configs]
            else:
                self.base = base()

        def __call__(self, grad, hidden, output, add):
            if args.arm == "lt":
                self.lt.apply(self.plans[add], grad, hidden, output, add)
            else:
                self.base(grad, hidden, output, add)
            if args.numerics and self.calls >= previous_invocations * chunks:
                if self.reference is None:
                    self.reference = torch.zeros_like(output, dtype=torch.float64)
                self.reference.addmm_(grad.double().T, hidden.double())
            self.calls += 1

    direct_dw.CublasWriter = ObservedWriter

    numerical_path = args.output.with_suffix(".numerics.json")
    save = torch.save

    def capture(dump, destination, *positional, **keywords):
        if destination != numerical_path:
            return save(dump, destination, *positional, **keywords)
        assert writer.calls == (previous_invocations + 1) * chunks
        assert writer.reference is not None
        compare_dump(dump, writer.reference, destination)

    sys.argv = [
        sys.argv[0], "current", str(args.output), "--chunks", str(chunks), "--writer", "cublas",
        "--sum-reduction", "--warmup", str(warmup), "--samples", str(samples),
    ]
    if args.numerics:
        torch.save = capture
        sys.argv += ["--dump", str(numerical_path)]
    replay.main()
    assert writer.calls == (previous_invocations + int(args.numerics)) * chunks
    result = json.loads(args.output.read_text())
    result["writer"] = args.arm
    result["dw_writer_calls"] = writer.calls
    if args.arm == "lt":
        result["cublasLt_version"] = writer.lt.version
        result["dw_cublasLt_algorithms"] = configs
        writer.lt.close()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
