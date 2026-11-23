"""Screen true BF16 hidden transposition with the existing cuBLASLt binding."""

import argparse
import ctypes as c
import importlib.util
import json
import os
from pathlib import Path

import torch

from replay_native import PRODUCTION, fp64_error
from frontier_dw_lib import (
    BF16, COMPUTE_FP32, FP32, TRANS_A, TRANS_B, Int, Layout, LtWriter, Ptr,
    scratch_for,
)


class PackedLt(LtWriter):
    """A mixed-order descriptor reads the physically token-contiguous hidden."""

    def __init__(self, rows, vocab, width):
        super().__init__(rows, vocab, width)
        self.layouts.update(
            row_alt=self.packed_layout(True, True, rows, vocab, width),
            col_alt=self.packed_layout(False, True, rows, vocab, width),
        )

    def new_layout(self, row_major, rows, vocab, width):
        return self.packed_layout(row_major, False, rows, vocab, width)

    def packed_layout(self, row_major, alternate, rows, vocab, width):
        operation = Ptr()
        self.check(self.desc_create, c.byref(operation), COMPUTE_FP32, FP32)
        if row_major:
            self.attribute(self.desc_attr, operation, TRANS_A, Int(1))
            a = self.matrix(True, BF16, rows, vocab, vocab)
            if alternate:
                self.attribute(self.desc_attr, operation, TRANS_B, Int(1))
                b = self.matrix(True, BF16, width, rows, rows)
            else:
                b = self.matrix(False, BF16, rows, width, rows)
            cd = self.matrix(True, FP32, vocab, width, width)
        else:
            self.attribute(self.desc_attr, operation, TRANS_B, Int(1))
            if alternate:
                a = self.matrix(True, BF16, width, rows, rows)
            else:
                self.attribute(self.desc_attr, operation, TRANS_A, Int(1))
                a = self.matrix(False, BF16, rows, width, rows)
            b = self.matrix(False, BF16, vocab, rows, vocab)
            cd = self.matrix(False, FP32, width, vocab, width)
        return Layout(operation, a, b, cd, row_major)


def pack(hidden):
    return hidden.T.contiguous().T


class PackedWriter:
    name = "cublasLtMatmul-packed-hidden"

    def __init__(self, tuning, winner, stream):
        self.stream = stream
        configs = [tuning["configurations"][i] for i in winner]
        rows, vocab, width = tuning["shape"]
        self.lt = PackedLt(rows, vocab, width)
        self.version = self.lt.version
        assert self.version == tuning["cublaslt_version"]
        self.pair = ".".join(map(str, winner))
        scratch = scratch_for(*configs)
        self.plans = tuple(self.lt.prepare(config, scratch) for config in configs)
        self.receipt = {
            "packed_stride": [1, rows],
            "pack_bytes_read_plus_written_each_call": 4 * rows * width,
            "plans": configs,
        }

    def __call__(self, grad, hidden, output, add, stream):
        assert stream == self.stream
        assert hidden.stride() == (hidden.size(1), 1)
        packed = pack(hidden)
        assert packed.stride() == (1, hidden.size(0))
        self.lt.apply(self.plans[add], grad, packed, output, add)


def replay_hook(plan):
    tuning = json.loads(plan.read_text())
    winner = tuning["winner"]
    if winner is None:
        raise RuntimeError("The tuning search produced no compatible packed pair")

    def make_writer(native, chunks, rows, vocab, width, stream):
        assert [rows, vocab, width] == tuning["shape"]
        assert native.version == tuning["cublaslt_version"]
        assert chunks == tuning["chunks"]
        return PackedWriter(tuning, winner, stream)

    return make_writer


def tune(chunks, path):
    from dw_check import FIXTURE, get_operands
    from frontier_dw_tune import median, paired

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2048)
    record = torch.load(FIXTURE, map_location="cpu", mmap=True, weights_only=True)["ce"]
    operands = get_operands(record, chunks)
    rows, vocab = operands[0][0].shape
    width = operands[0][1].shape[1]
    output = torch.empty((vocab, width), dtype=torch.float32, device="cuda")
    output.zero_()
    stream = torch.cuda.current_stream()
    spec = importlib.util.spec_from_file_location(
        "production_writer",
        PRODUCTION / "gift/torchtitan/distributed/gfsdp/_staged_head_dw_writer.py",
    )
    writer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer_module)
    baseline = writer_module.make_head_dw_writer("new", chunks, rows, vocab, width, stream)
    lt = PackedLt(rows, vocab, width)
    assert lt.version == baseline.version

    configs = {}
    heuristic_counts = {}
    for layout in lt.layouts:
        for cap in (0, 32 << 20):
            found = lt.candidates(layout, cap, 48)
            heuristic_counts[f"{layout}:{cap}"] = len(found)
            for item in found:
                configs[(item["order"], item["algo"])] = item
    configs = list(configs.values())
    scratch = scratch_for(*configs)
    plans = [lt.prepare(config, scratch) for config in configs]
    report = {
        "chunks": chunks, "shape": [rows, vocab, width], "cublaslt_version": lt.version,
        "torch": torch.__version__, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "heuristic_counts": heuristic_counts, "configurations": configs, "single": {},
        "native_pair": baseline.pair, "winner": None,
    }
    shortlist = []
    for add in (False, True):
        grad, hidden = operands[int(add)]
        baseline_call = lambda: baseline(grad, hidden, output, add, stream)
        successful = {"control": baseline_call, "control_repeat": baseline_call}
        failures = {}

        def apply(plan):
            lt.apply(plan, grad, pack(hidden), output, add)

        for i, plan in enumerate(plans):
            try:
                apply(plan)
            except RuntimeError as error:
                failures[i] = str(error)
            else:
                successful[i] = lambda plan=plan: apply(plan)

        samples = paired(successful, 4, 2, 29 + int(add))
        ranked = sorted(
            (key for key in successful if isinstance(key, int)),
            key=lambda key: median(samples[key]),
        )
        selection = ranked[:4]
        for desc in lt.layouts:
            layout_winner = next((i for i in ranked if configs[i]["order"] == desc), None)
            if layout_winner is not None and layout_winner not in selection:
                selection.append(layout_winner)
        shortlist.append(selection)
        report["single"][int(add)] = {
            "samples": samples, "selected": selection, "launch_failures": failures,
        }
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("Per-beta screening finished", heuristic_counts, shortlist, flush=True)

    def native_chain():
        for i, (grad, hidden) in enumerate(operands):
            baseline(grad, hidden, output, bool(i), stream)

    def make_chain(first, rest):
        def chain():
            for i, (grad, hidden) in enumerate(operands):
                lt.apply(plans[rest if i else first], grad, pack(hidden), output, bool(i))
        return chain

    chains = {f"{first}.{rest}": make_chain(first, rest)
              for first in shortlist[0] for rest in shortlist[1]}
    chains.update(control=native_chain, control_repeat=native_chain)
    chain = paired(chains, 6, 1, 71)
    ranked = sorted((name for name in chain if "." in name), key=lambda name: median(chain[name]))
    report.update(chain=chain, chain_rank=ranked)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("Full-chain screening finished", ranked[:8], flush=True)

    reference = torch.zeros_like(output, dtype=torch.float64)
    for grad, hidden in operands:
        reference.addmm_(grad.double().T, hidden.double())
    native_chain()
    baseline_error = fp64_error(output, reference)
    baseline_output = output.clone()
    report["fp64"] = {"control": baseline_error}
    for name in ranked:
        chains[name]()
        error = fp64_error(output, reference)
        error["bitwise_equal"] = bool(torch.equal(output, baseline_output))
        report["fp64"][name] = error
        if all(error[key] <= value for key, value in baseline_error.items()):
            report["winner"] = list(map(int, name.split(".")))
            break
    if report["winner"] is None and ranked:
        report["diagnostic_pair"] = list(map(int, ranked[0].split(".")))
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("FP64 screening finished", report["winner"], report.get("diagnostic_pair"), flush=True)
    torch.cuda.synchronize()
    lt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=int, choices=(2, 8))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tune(args.chunks, args.output)
