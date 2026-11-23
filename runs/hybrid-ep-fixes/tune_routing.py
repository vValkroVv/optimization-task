"""Sweep token rows and warp counts for the standalone HybridEP metadata kernel."""

import argparse
import hashlib
import json
import statistics
import sys
from functools import partial
from pathlib import Path

import torch

import bench_routing as bench


TILES = tuple(
    (rows, warps)
    for rows in (1, 2, 4, 8, 16, 32, 64)
    for warps in (1, 2, 4, 8)
    if warps <= rows
)
CONTROL = "rows8_warps4"
TRIALS = 8


@torch.library.custom_op(
    "hybrid_ep_bench::tuned_metadata", mutates_args=(), device_types="cuda"
)
def tuned_op(
    ids: torch.Tensor, scores: torch.Tensor, experts: int, rows: int, warps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return bench.grouped(ids, scores, experts, rows=rows, warps=warps)


def wrapped_tile(ids, scores, experts, *, rows, warps):
    if (
        not bench.current._can_use_cuda_kernel(ids, scores, experts)
        or experts not in bench.EXPERT_COUNTS
        or not ids.is_contiguous()
        or not scores.is_contiguous()
    ):
        return bench.current.indices_to_map(ids, scores, experts)
    return tuned_op(ids, scores, experts, rows, warps)


def bind_variants(ids, scores, experts):
    variants = {"current_triton": partial(bench.current.indices_to_map, ids, scores, experts)}
    for rows, warps in TILES:
        variants[f"rows{rows}_warps{warps}"] = partial(
            wrapped_tile, ids, scores, experts, rows=rows, warps=warps
        )
    variants["control_repeat"] = variants[CONTROL]
    return variants


def measure(variants, batch, *, distinct_outputs):
    captures = {
        name: bench.capture(fn, batch, distinct_outputs=distinct_outputs)
        for name, fn in variants.items()
    }
    for graph, _ in captures.values():
        bench.graph_us(graph, batch)
    samples = {name: [] for name in variants}
    host = {name: [] for name in variants}
    names = list(variants)
    for trial in range(TRIALS):
        for name in names if trial % 2 == 0 else names[::-1]:
            samples[name].append(bench.graph_us(captures[name][0], batch))
            if not distinct_outputs:
                host[name].append(bench.host_us(variants[name], batch))
    return samples, host


def measure_case(ids, scores, experts):
    variants = bind_variants(ids, scores, experts)
    expected = bench.reference(ids, scores, experts)
    for name, fn in variants.items():
        bench.assert_parity(expected, fn())
        print(f"Parity passed: {name}", flush=True)
    torch.cuda.synchronize()
    graph, host = measure(variants, 64, distinct_outputs=False)
    distinct, _ = measure(variants, 24, distinct_outputs=True)
    results = {}
    for name in variants:
        results[name] = {
            "graph_us": graph[name],
            "distinct_us": distinct[name],
            "host": host[name],
            "medians": {
                "graph_us": statistics.median(graph[name]),
                "distinct_us": statistics.median(distinct[name]),
                **{
                    key: statistics.median(sample[key] for sample in host[name])
                    for key in host[name][0]
                },
            },
        }

    finalists = {"current_triton", CONTROL}
    tiles = [name for name in variants if name.startswith("rows")]
    for scope in ("graph_us", "distinct_us", "integrated_us"):
        ordered = sorted(tiles, key=lambda name: results[name]["medians"][scope])
        finalists.update(ordered[:3])
        print(scope, [(name, results[name]["medians"][scope]) for name in ordered[:4]], flush=True)
    for name in sorted(finalists):
        profile = bench.profile(variants[name], 8)
        kernels = profile["kernels"]["names_and_counts"]
        expected_name = "_routing_metadata_kernel" if name == "current_triton" else "grouped_rows"
        assert len(kernels) == 1 and expected_name in next(iter(kernels)), kernels
        assert sum(kernels.values()) == 8
        results[name]["profile"] = profile
    return results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance, fixtures = bench.load_native_inputs(args.fixture_dir)
    sources = (Path(__file__), Path(bench.__file__), Path(bench.current.__file__))
    result = {
        "input_provenance": provenance,
        "device": torch.cuda.get_device_name(),
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_build": torch.version.cuda,
        "triton": bench.triton.__version__,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "control": CONTROL,
        "tiles": TILES,
        "trials": TRIALS,
        "graph_calls": 64,
        "distinct_calls": 24,
        "cases": [],
    }
    for ranks, experts, adaptation, ids_cpu, scores_cpu in bench.fixture_cases(fixtures, bench.TOP_K_SWEEP):
        ids, scores = ids_cpu.cuda(), scores_cpu.cuda()
        case = {
            "fixture_ranks": ranks,
            "input_adaptation": adaptation,
            "tokens": ids.shape[0],
            "top_k": ids.shape[1],
            "experts": experts,
        }
        print(f"Case: {case}", flush=True)
        case["variants"] = measure_case(ids, scores, experts)
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
