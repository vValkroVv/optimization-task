"""Tune FP32 dW accumulation on the captured loss GEMM operands."""

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch

from dw_check import FIXTURE, error, get_operands
from direct_dw import CublasWriter
from frontier_dw_lib import LtWriter, scratch_for


def trial(fn, repeat, start, end):
    torch.cuda.synchronize()
    clock = time.perf_counter
    start.record()
    t0 = clock()
    for _ in range(repeat):
        fn()
    submit = (clock() - t0) * 1000 / repeat
    end.record()
    end.synchronize()
    return {
        "stream_ms": start.elapsed_time(end) / repeat,
        "submit_ms": submit,
        "synchronized_wall_ms": (clock() - t0) * 1000 / repeat,
    }


def paired(arms, rounds, repeat, seed):
    randomize = random.Random(seed)
    names = list(arms)
    samples = {name: [] for name in names}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(rounds):
        randomize.shuffle(names)
        for name in names:
            samples[name].append(trial(arms[name], repeat, start, end))
    return samples


def median(samples, metric="stream_ms"):
    return statistics.median(sample[metric] for sample in samples)


def screen(writer, operands, output, baseline):
    configurations = {}
    for order in ("col", "row"):
        for cap in (0, 32 << 20):
            for item in writer.candidates(order, cap):
                configurations[(item["order"], item["algo"])] = item
    configs = list(configurations.values())
    scratch = scratch_for(*configs)
    plans = [writer.prepare(config, scratch) for config in configs]
    result = {"configurations": configs, "single_gemm": {}}
    for add in (False, True):
        grad, hidden = operands[int(add)]
        arms = {
            i: (lambda plan=plan: writer.apply(plan, grad, hidden, output, add))
            for i, plan in enumerate(plans)
        }
        arms["cublas"] = lambda: baseline(grad, hidden, output, add)
        good_arms = {}
        decode_errors = {}
        for name, fn in arms.items():
            try:
                fn()
            except RuntimeError as err:
                decode_errors[name] = str(err)
            else:
                good_arms[name] = fn
        samples = paired(good_arms, 3, 3, int(add))
        fastest = sorted((i for i in good_arms if i != "cublas"), key=lambda i: median(samples[i]))
        result["single_gemm"][str(int(add))] = {
            "samples": samples, "fastest": fastest,
            "zero_workspace": [i for i in fastest if not configs[i]["workspace"]],
            "launch_errors": decode_errors,
        }
    pairs = set()
    for selection in ("fastest", "zero_workspace"):
        pairs.update(
            (i, j)
            for i in result["single_gemm"]["0"][selection][:2]
            for j in result["single_gemm"]["1"][selection][:2]
        )
    return result, sorted(pairs)


def run(chunks, record, output_path):
    operands = get_operands(record, chunks)
    rows, vocab = operands[0][0].shape
    width = operands[0][1].shape[1]
    output = torch.empty((vocab, width), dtype=torch.float32, device="cuda")
    output.zero_()
    baseline = CublasWriter()
    writer = LtWriter(rows, vocab, width)
    native_mm_settings = {
        "allow_bf16_reduced_precision_reduction": str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    result, pairs = screen(writer, operands, output, baseline)
    result.update({
        "chunks": chunks, "rows_per_chunk": rows, "vocab": vocab, "width": width,
        "cublasLt_version": writer.version, "torch": torch.__version__,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "device_name": torch.cuda.get_device_name(),
        "precision": native_mm_settings,
        "scope": "eager single-stream CUDA event span incl. any submission gaps; host submit and synchronized wall reported separately",
    })
    configs = result["configurations"]

    def baseline_chain():
        for i, (grad, hidden) in enumerate(operands):
            baseline(grad, hidden, output, i != 0)

    def make_chain(first, rest):
        scratch = scratch_for(configs[first], configs[rest])
        overwrite, add = (writer.prepare(configs[index], scratch) for index in (first, rest))

        def chain():
            for i, (grad, hidden) in enumerate(operands):
                writer.apply(add if i else overwrite, grad, hidden, output, i != 0)

        return chain

    chain_arms = {f"{first}.{rest}": make_chain(first, rest) for first, rest in pairs}
    chain_arms.update({"cublas": baseline_chain, "cublas_same": baseline_chain})
    result["chain"] = paired(chain_arms, 6, 1, 19)
    result["chain_rank"] = sorted(chain_arms, key=lambda name: median(result["chain"][name]))
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"chunks {chunks}: timed chain; now computing FP64", flush=True)

    # FP64 products and reduction over the exact reused native BF16 operands.
    reference = torch.zeros_like(output, dtype=torch.float64)
    for grad, hidden in operands:
        reference.addmm_(grad.double().T, hidden.double())

    result["fp64"] = {}
    for name in result["chain_rank"]:
        chain_arms[name]()
        result["fp64"][name] = error(output, reference)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=int, choices=(2, 4, 8, 16))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    record = torch.load(FIXTURE, map_location="cpu", mmap=True, weights_only=True)["ce"]
    run(args.chunks, record, args.output)
