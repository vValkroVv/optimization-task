"""Measure the folded projection and dHidden on an actual head-matrix fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import time

import torch

from ce_bench import evaluate, file_sha256
from lt_ops import CONFIG_FIELDS, Plan, _library
from matrix_fixture import load_inputs
from torchtitan.components.loss import MTPLoss
from torchtitan.ops.exact_folded_ce import ExactFoldedCrossEntropy
from torchtitan.tools.utils import device_memory_usage

BASE = Path(__file__).resolve().parents[2]
CE_HASH = "17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8"


def inputs(model, chunks):
    rec = load_inputs(model, BASE / "inputs" / f"{model}.pt")
    activations = rec["hidden"].cuda().flatten(0, 1)
    weight = rec["weight"].cuda()
    y, pos = rec["labels"], rec["positions"]
    y = torch.cat((y, MTPLoss._shift_labels(y, 1, pos))).cuda()
    chunk_tokens = activations.size(1) // chunks
    hidden = [tile.contiguous() for tile in activations.split(chunk_tokens, dim=1)]
    labels = [tile.contiguous() for tile in y.split(chunk_tokens, dim=1)]
    loss = ExactFoldedCrossEntropy(
        str(BASE / "native-ce/centered-folded-v1.so"), library_sha256=CE_HASH
    )
    return rec, activations.shape, hidden, weight, labels, loss


def dlogits(loss, hidden, weight, labels, record):
    with torch.no_grad():
        logits = torch.nn.functional.linear(hidden, weight)
    _, _, _, dz = evaluate(loss, logits.requires_grad_(True), labels, record["head_weights"], record["denominator"])
    assert logits.data_ptr() == dz.data_ptr()
    return dz


def bits_equal(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError((actual.shape, expected.shape))
    for a, b in zip(actual.split(128), expected.split(128), strict=True):
        if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
            return False
    return True


def diff_metrics(actual, expected):
    mismatches, max_abs, sum_sq, norm_sq = 0, 0.0, 0.0, 0.0
    for a, b in zip(actual.split(64), expected.split(64), strict=True):
        mismatches += (a.view(torch.int16) != b.view(torch.int16)).count_nonzero().item()
        bf = b.float()
        diff = a.float() - bf
        max_abs = max(max_abs, diff.abs().max().item())
        sum_sq += diff.square().sum(dtype=torch.float64).item()
        norm_sq += bf.square().sum(dtype=torch.float64).item()
    return dict(mismatches=mismatches, numel=expected.numel(), max_abs=max_abs,
                relative_l2=(sum_sq / norm_sq) ** 0.5)


def memory():
    stats = torch.cuda.memory_stats()
    external, total = device_memory_usage(stats)
    return dict(allocated_peak_bytes=stats["allocated_bytes.all.peak"],
                reserved_peak_bytes=stats["reserved_bytes.all.peak"], gift_total_peak_gib=total,
                gift_external_gib=external, allocator_retries=stats["num_alloc_retries"])


class Timer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        self.end.record()
        self.end.synchronize()

    def sample(self, call):
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        self.start.record()
        value = call()
        self.end.record()
        submitted = time.perf_counter()
        self.end.synchronize()
        finished = time.perf_counter()
        result = dict(
            event_ms=self.start.elapsed_time(self.end), wall_ms=(finished - begin) * 1e3,
            submit_ms=(submitted - begin) * 1e3, wait_ms=(finished - submitted) * 1e3,
            **memory(),
        )
        return value, result


def screen_algorithms(lhs, rhs, native, result, expected, timer, exhaustive):
    variants = []
    screen = []
    for row in (False, True):
        plan = Plan(lhs, rhs, result, row=row, exhaustive=exhaustive)
        print("screen", plan.problem, plan.counts, flush=True)
        scan_report = dict(problem=plan.problem, counts=plan.counts, candidates=[])
        screen.append(scan_report)
        for config in plan.configs:
            i = config["index"]
            if i % 8 == 0:
                _, control = timer.sample(native)
            status, sample = timer.sample(lambda: plan.launch(i, lhs, rhs, result))
            entry = dict(config, status=status)
            if status == 0:
                entry.update(sample_ms=sample["event_ms"], baseline_ms=control["event_ms"])
                variants.append((entry, plan))
            scan_report["candidates"].append(entry)
    # Re-measure the short list before checking every distinct top configuration's output.
    variants.sort(key=lambda pair: pair[0]["sample_ms"])
    for entry, plan in variants[:16]:
        calls = [timer.sample(lambda: plan.launch(entry["index"], lhs, rhs, result))[1]["event_ms"] for _ in range(3)]
        entry["rescreen_ms"] = statistics.median(calls)
        entry["bitwise"] = bits_equal(result, expected)
        if not entry["bitwise"]:
            entry["error"] = diff_metrics(result, expected)
    shortlist = sorted(
        ((entry, plan) for entry, plan in variants[:16] if entry["bitwise"]),
        key=lambda pair: pair[0]["rescreen_ms"],
    )
    return screen, shortlist[:3]


def capture(call):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        call()
    return graph


def groups_by_index(groups, round_index):
    rotated = list(groups[round_index % len(groups):]) + list(groups[:round_index % len(groups)])
    return reversed(rotated) if (round_index // len(groups)) % 2 else rotated


def compare(calls, timer, rounds):
    groups = list(calls)
    for call in calls.values():
        for _ in range(3):
            call()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    result = {name: [] for name in groups}
    for i in range(rounds):
        for name in groups_by_index(groups, i):
            torch.cuda.empty_cache()
            output, sample = timer.sample(calls[name])
            del output
            result[name].append(sample)
    return result


def compare_graphs(calls, timer, rounds):
    graphs = {name: capture(call) for name, call in calls.items()}
    graph_result = {name: [] for name in graphs}
    def replay(graph):
        for _ in range(5):
            graph.replay()
    for i in range(rounds):
        for name in groups_by_index(list(graphs), i):
            _, sample = timer.sample(lambda: replay(graphs[name]))
            graph_result[name].append(sample["event_ms"] / 5)
    return graph_result


def as_json_digest(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def aligned_candidate(expected, output, shortlist, lhs, rhs):
    for entry, plan in shortlist:
        plan.run(entry["index"], lhs, rhs, output)
        yield dict(
            problem=plan.problem, config={key: entry[key] for key in CONFIG_FIELDS},
            workspace_bytes=entry["workspace_bytes"], bitwise=bits_equal(output, expected),
            baseline_sha256=as_json_digest(expected) if expected.numel() < 30000000 else None,
        )


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    timer = Timer()
    rec, shape, hidden, weight, labels, loss = inputs(args.model, args.chunks)
    index = args.part
    h = hidden[index].flatten(0, 1)
    dz = d = final = None
    report = dict(
        model=args.model, chunks=args.chunks, part=index, process=os.getpid(),
        hostname=socket.gethostname(), cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        gpu=torch.cuda.get_device_name(), sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        capability=torch.cuda.get_device_capability(), pytorch=torch.__version__,
        pytorch_git=torch.version.git_version, cuda=torch.version.cuda, cublaslt=_library().head_lt_version(),
        native_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        ce_sha256=CE_HASH, fixture_sha256=file_sha256(BASE / "inputs" / f"{args.model}.pt"),
        fixture=rec["provenance"], roles={},
    )
    for role in args.roles:
        if role != "projection" and dz is None:
            dz = dlogits(loss, hidden[index], weight, labels[index], rec)
            d = dz.flatten(0, 1)
            final = weight.new_empty(shape)
        if role == "projection":
            lhs, rhs = h, weight.T
            output = weight.new_empty((h.size(0), weight.size(0)))
            reference = torch.nn.functional.linear(hidden[index], weight).flatten(0, 1)
            def control_alloc():
                return torch.nn.functional.linear(hidden[index], weight)
            def control_prealloc():
                return torch.mm(lhs, rhs, out=output)
        elif role == "dhidden":
            lhs, rhs = d, weight
            output = weight.new_empty((d.size(0), weight.size(1)))
            reference = torch.mm(lhs, rhs)
            def control_alloc():
                return torch.mm(lhs, rhs)
            def control_prealloc():
                return torch.mm(lhs, rhs, out=output)
        else:
            lhs, rhs = dz, weight
            span = slice(index * dz.size(1), (index + 1) * dz.size(1))
            output = final[:, span]
            reference = torch.mm(d, weight).view_as(output)
            def control_alloc():
                return output.copy_(torch.mm(d, rhs).view_as(output))
            control_prealloc = control_alloc
        control_prealloc()
        assert bits_equal(output, reference), role
        scan, shortlist = screen_algorithms(lhs, rhs, control_prealloc, output, reference, timer, args.exhaustive)
        parity = list(aligned_candidate(reference, output, shortlist, lhs, rhs))
        print("shortlist", role, parity, flush=True)
        role_result = dict(screen=scan, tested=parity)
        report["roles"][role] = role_result
        if shortlist:
            del reference
            direct = role == "dhidden-direct"
            # Eager calls include the normal output allocation. The replay
            # instead keeps common inputs and the destination preallocated.
            eager = dict(control=control_alloc, control_repeat=control_alloc)
            graph = dict(control=control_prealloc, control_repeat=control_prealloc)
            def call(entry, plan, prealloc):
                dest = output if direct or prealloc else lhs.new_empty((lhs.size(0), rhs.size(1)))
                return plan.run(entry["index"], lhs, rhs, dest)
            from functools import partial
            for i, (entry, plan) in enumerate(shortlist):
                eager[f"candidate-{i}"] = partial(call, entry, plan, False)
                graph[f"candidate-{i}"] = partial(call, entry, plan, True)
            if not direct:
                output_shape = output.shape
                output = None
            role_result["eager"] = compare(eager, timer, args.rounds)
            if not direct:
                output = weight.new_empty(output_shape)
            role_result["graph"] = compare_graphs(graph, timer, args.rounds)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        del control_prealloc, control_alloc
        output = None
    print("finished", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("100b", "700b"))
    parser.add_argument("--chunks", required=True, type=int, choices=(2, 8))
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--roles", nargs="+", default=["projection", "dhidden", "dhidden-direct"])
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
