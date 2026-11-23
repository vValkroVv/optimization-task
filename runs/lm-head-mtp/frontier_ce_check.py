"""Check raw dLogits equality and measure CUDA Graph device spans."""

import argparse
import json
from pathlib import Path
from statistics import median
import sys

import torch

import replay
from frontier_ce import launch

sys.path.insert(0, str(replay.CURRENT / "gift"))
from torchtitan.components.loss import MTPLoss
from torchtitan.experiments.four_fusion import stable_ce


def capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def event_median(graphs, rounds=6, repeat=10):
    samples = {name: [] for name in graphs}
    for round_index in range(rounds):
        names = list(graphs)
        if round_index % 2:
            names.reverse()
        for name in names:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                graphs[name].replay()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) / repeat)
    return {name: {"ms": median(values), "samples_ms": values} for name, values in samples.items()}


@torch.no_grad()
def run(output):
    torch.backends.cuda.matmul.allow_tf32 = False
    fixture = torch.load(replay.FIXTURE, map_location="cpu", weights_only=True, mmap=True)
    record = fixture["ce"]
    weight = record["weight"].cuda().to(torch.bfloat16)
    labels = record["labels"].cuda()
    positions = record["positions"]
    if positions is not None:
        positions = positions.cuda()
    shifted = MTPLoss._shift_labels(labels, 1, positions)
    all_labels = torch.cat((labels, shifted))
    denominator = torch.as_tensor(record["denominator"], device="cuda", dtype=torch.float32)
    scales = tuple(weight / denominator for weight in (1.0, 0.3))
    results = []
    for chunks in (2, 4, 8, 16):
        end = labels.shape[-1] // chunks
        hidden = record["hidden"].flatten(0, 1)[:, :end].contiguous().cuda()
        logits = torch.nn.functional.linear(hidden, weight).view(-1, weight.shape[0])
        target = all_labels[:, :end].contiguous().view(-1)
        # Force every ignored-row path and both tail/first-column target stores.
        target[0] = -100
        target[1] = 0
        target[-1] = weight.shape[0] - 1
        loss, maxima, logs = zip(*[
            stable_ce._forward(x, y)
            for x, y in zip(logits.chunk(2), target.chunk(2), strict=True)
        ], strict=True)
        baseline = torch.empty_like(logits)
        candidate = torch.empty_like(logits)

        def control():
            for args in zip(
                logits.chunk(2), target.chunk(2), maxima, logs, scales, baseline.chunk(2), strict=True
            ):
                stable_ce._launch_backward(*args)

        def candidate_fn(block, warps):
            return lambda: launch(logits, target, maxima, logs, scales, candidate, block, warps)

        graphs = {"native": capture(control)}
        configs = ((2048, 4), (4096, 4), (8192, 4), (8192, 8), (16384, 8))
        control()
        for block, warps in configs:
            fn = candidate_fn(block, warps)
            fn()
            assert torch.equal(baseline, candidate), (chunks, block, warps)
            assert int(torch.count_nonzero(candidate[target == -100])) == 0
            graphs[f"{block}/{warps}"] = capture(fn)
        # Per-row FP64 oracle checks both head scales and the selected vocabulary tail.
        rows = sorted(set((0, 1, len(target) // 2, len(target) - 1, len(target) - 2)))
        exact = logits[rows].to(torch.float64).softmax(dim=1)
        for index, row in enumerate(rows):
            label = int(target[row])
            if label == -100:
                exact[index].zero_()
            else:
                exact[index, label] -= 1
                exact[index].mul_(scales[int(row >= len(target) // 2)])
        got = candidate[rows].to(torch.float64)
        quality = float(torch.linalg.vector_norm(got - exact) / torch.linalg.vector_norm(exact))
        results.append({
            "chunks": chunks,
            "shape": list(logits.shape),
            "ignored_rows": int((target == -100).sum()),
            "fixed_edge_labels": [int(target[1]), int(target[-1])],
            "head_scales": [float(s) for s in scales],
            "ce_losses": [float(x) for x in loss],
            "dlogits_bitwise_all_configs": True,
            "fp64_selected_row_relative_l2": quality,
            "selected_rows": rows,
            "timings": event_median(graphs),
        })
        del graphs
        torch.cuda.synchronize()
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
