"""Replay one selected CuTe change through two compiled GIFT norms with SAC."""

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
from statistics import median
import sys
from unittest.mock import patch

import torch
import yaml

from norm_f1_component import load_variant
from norm_skip import digest, measure, mismatched_bytes


def measure_graph(graph, samples, batch):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(samples):
        start.record()
        for _ in range(batch):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / batch)
    return {"sample_event_ms": times, "median_event_ms": median(times)}


def run(args):
    sys.path.append(str(args.replay_root.resolve(strict=True)))
    from residual_replay_probe import build
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config
    from torchtitan.models.common.norms.cute_gated_norm import backward, forward
    from torchtitan.tools.utils import device_memory_usage

    if Path(build.__code__.co_filename).resolve().parent != args.replay_root.resolve():
        raise RuntimeError("Expected the specified historical replay helper")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    torch.cuda.set_device(0)
    deterministic = yaml.safe_load(args.resolved_config.read_bytes())["debug"][
        "deterministic"
    ]
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        torch.utils.deterministic.fill_uninitialized_memory = False
    inductor_config.triton.cudagraphs = False
    switches = {
        "GIFT_CUTE_NORM_FUSE_RESIDUAL": "1",
        "GIFT_CUTE_NORM_SAC_STATS_ONLY": "1",
        "GIFT_FP8_QUANTIZE_WEIGHT_TAIL": "1",
        "GIFT_CUTE_NORM_BWD_SKIP": "0",
    }
    os.environ.update(switches)
    original, entry, module_name, kernel_names = {
        "f1": (
            forward,
            "gated_norm_fwd",
            "forward_f1_vector",
            ("GnFwdStats_", "GnFwdStatsVector_"),
        ),
        "b1": (
            backward,
            "gated_norm_bwd",
            "backward_b1_gate_layout",
            ("GnBwdDs_", "GnBwdDsGateLayout_"),
        ),
    }[args.stage]
    candidate = load_variant(original, args.candidate, module_name)

    def select(arm):
        if arm == "A":
            return nullcontext()
        return patch.object(original, entry, getattr(candidate, entry))

    launch, leaves, tangent, capture = build(args.fixture, "live", "B")

    def differentiate(y):
        return torch.autograd.grad(y, tuple(leaves.values()), tangent)

    def call():
        y = launch()
        return y, *differentiate(y)

    stream = None
    if args.graph_batch:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with select("A"):
            reference = call()
        with select("B"):
            trial = call()
    if stream is not None:
        stream.synchronize()
    mismatches = mismatched_bytes(("output", *leaves), reference, trial)
    del trial
    if any(mismatches.values()):
        raise RuntimeError(f"Compiled norm output or gradient changed: {mismatches}")

    profiles = {}
    pairs = []
    graph_parity = []
    with dynamo_config.patch(error_on_recompile=True):
        graphs = []
        if args.graph_batch:
            # Alternate which implementation gets the first private graph pool.
            # Keep captured outputs alive for the lifetime of their graphs.
            for capture_order in ("AB", "BA"):
                captures = {}
                parity = {}
                for arm in capture_order:
                    with select(arm), torch.cuda.stream(stream):
                        for _ in range(5):
                            call()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with select(arm), torch.cuda.graph(graph, stream=stream):
                        recorded = call()
                    for _ in range(args.warmup):
                        graph.replay()
                    torch.cuda.synchronize()
                    parity[arm] = mismatched_bytes(
                        ("output", *leaves), reference, recorded
                    )
                    if any(parity[arm].values()):
                        raise RuntimeError(
                            f"Graph output or gradient changed: {parity}"
                        )
                    captures[arm] = graph, recorded
                graphs.append(captures)
                graph_parity.append({"capture_order": capture_order, "arms": parity})
        else:
            for arm in ("A", "B"):
                with select(arm):
                    for _ in range(args.warmup):
                        call()
                        torch.cuda.synchronize()
        del reference
        for index in range(args.pairs):
            order = ("A", "B") if index % 2 == 0 else ("B", "A")
            blocks = {}
            for arm in order:
                if graphs:
                    block = measure_graph(
                        graphs[(index // 2) % 2][arm][0], args.samples, args.graph_batch
                    )
                else:
                    with select(arm):
                        block = measure(launch, differentiate, args.samples)
                        _, block["device_peak_gib"] = device_memory_usage(
                            torch.cuda.memory_stats()
                        )
                blocks[arm] = block
            pairs.append({"order": order, "arms": blocks})
        for arm, expected, inactive in (
            ("A", kernel_names[0], kernel_names[1]),
            ("B", kernel_names[1], kernel_names[0]),
        ):
            with (
                select(arm),
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as trace,
            ):
                for index in range(args.profile_calls):
                    with torch.profiler.record_function(
                        f"norm_stage_probe_call_{index}"
                    ):
                        if graphs:
                            graphs[0][arm][0].replay()
                        else:
                            call()
                        torch.cuda.synchronize()
            kernels = Counter(
                e.name for e in trace.events() if e.device_type.name == "CUDA"
            )
            ops = {
                e.name
                for e in trace.events()
                if e.name.startswith("torchtitan::cute_gated_norm_")
            }
            expected_count = args.profile_calls * (4 if args.stage == "f1" else 2)
            if (
                sum(n for name, n in kernels.items() if expected in name)
                != expected_count
                or any(inactive in name for name in kernels)
                or (
                    not graphs
                    and not {
                        "torchtitan::cute_gated_norm_fwd_stats",
                        "torchtitan::cute_gated_norm_residual_fwd",
                    }.issubset(ops)
                )
            ):
                raise RuntimeError(
                    f"Unexpected compiled stage activation for {arm}: {ops}, {kernels}"
                )
            trace_path = args.output.with_suffix(f".{arm}.trace.json")
            trace.export_chrome_trace(str(trace_path))
            profiles[arm] = {
                "kernels": kernels,
                "ops": sorted(ops),
                "trace_sha256": digest(trace_path),
            }

    with args.output.open("x") as handle:
        json.dump(
            {
                "stage": args.stage,
                "hostname": platform.node(),
                "gpu": torch.cuda.get_device_name(),
                "torch_revision": torch.version.git_version,
                "switches": switches,
                "fixture_sha256": digest(args.fixture),
                "resolved_config_sha256": digest(args.resolved_config),
                "candidate_sha256": digest(args.candidate),
                "runner_sha256": digest(Path(__file__)),
                "historical_replay_sha256": digest(Path(build.__code__.co_filename)),
                "capture": capture,
                "warmup": args.warmup,
                "samples": args.samples,
                "profile_calls": args.profile_calls,
                "graph_batch": args.graph_batch,
                "graph_parity": graph_parity,
                "mismatched_bytes": mismatches,
                "profiles": profiles,
                "pairs": pairs,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("f1", "b1"), required=True)
    for name in ("candidate", "fixture", "resolved-config", "replay-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--profile-calls", type=int, default=1)
    parser.add_argument("--graph-batch", type=int, default=0)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.pairs, args.profile_calls) < 1:
        parser.error("Replay counts must be positive")
    if args.graph_batch < 0:
        parser.error("Graph batch must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
