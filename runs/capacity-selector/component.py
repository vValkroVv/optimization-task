#!/usr/bin/env python3
"""Capture native router outputs and replay capacity preparation on eight GPUs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import partial
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns


TOKENS, TOP_K, EXPERTS, EP = 8192, 8, 128, 8
NATIVE_QUOTA = 6250
TEST_QUOTA = 2048
CAPACITY_MODULE = "torchtitan.models.common.moe.dispatch.capacity"
CAPACITY_SHA = "2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120"


def load_capacity(path=None):
    """Verify the original 128-counter selector before loading either variant."""
    if path is None:
        spec = importlib.util.find_spec(CAPACITY_MODULE)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot resolve {CAPACITY_MODULE}")
        path = Path(spec.origin)
    path = path.resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CAPACITY_SHA:
        raise RuntimeError(
            f"capacity source differs from S 128-counter baseline: {path}"
        )
    current = sys.modules.get(CAPACITY_MODULE)
    if current is not None:
        if Path(current.__file__).resolve() != path:
            raise RuntimeError(
                f"capacity module already loaded from {current.__file__}"
            )
        return current
    spec = importlib.util.spec_from_file_location(CAPACITY_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[CAPACITY_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(CAPACITY_MODULE)
        raise
    return module


def install_route_capture(output):
    """Tap the existing opaque op in the separate, one-update S diagnostic."""
    import torch

    capacity_module = sys.modules[CAPACITY_MODULE]
    if (
        hashlib.sha256(Path(capacity_module.__file__).read_bytes()).hexdigest()
        != CAPACITY_SHA
    ):
        raise RuntimeError("native capture must use the original S capacity selector")
    rank = int(os.environ["RANK"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"rank{rank}.pt"
    if path.exists():
        raise FileExistsError(path)
    captured = False

    @capacity_module._conditional_capacity_mask.register_kernel("cuda")
    def capture(scores, buckets, valid, needs_drop, num_experts, capacity):
        nonlocal captured
        if (
            not captured
            and scores.shape == (TOKENS, TOP_K)
            and num_experts == EXPERTS
            and capacity == NATIVE_QUOTA
        ):
            cpu_scores = scores.detach().to("cpu", copy=True)
            cpu_buckets = buckets.view_as(scores).to("cpu", copy=True)
            cpu_valid = valid.to("cpu")
            if not cpu_valid.all():
                raise RuntimeError(
                    "expected live native top-k routes in the captured sample"
                )
            # The buckets equal the original expert IDs when every route is live.
            record = {
                "rank": rank,
                "scores": cpu_scores,
                "expert_ids": cpu_buckets,
                "score_stride": list(scores.stride()),
                "num_experts": num_experts,
                "source_quota": capacity,
                "capacity_sha256": CAPACITY_SHA,
            }
            tmp = path.with_suffix(".tmp")
            torch.save(record, tmp)
            tmp.rename(path)
            captured = True
        # Match the original CUDA backend. The schema and fake kernel stay intact.
        if needs_drop:
            return capacity_module._sorted_capacity_mask(
                scores, buckets, valid, num_experts=num_experts, capacity=capacity
            )
        return valid.clone()


def configure_cache(root, rank):
    root = root.resolve(strict=True)
    paths = {
        "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor",
        "TRITON_CACHE_DIR": root / "triton" / f"rank_{rank}",
    }
    for name, path in paths.items():
        if not path.is_dir():
            raise RuntimeError(f"missing existing {name} directory: {path}")
        os.environ[name] = str(path)
    return {name: str(path) for name, path in paths.items()}


def make_fragment(branchless, quota, ep_group):
    from torchtitan.config.moe import TokenDropConfig
    from torchtitan.models.common.moe.dispatch.hybridep import HybridEPTokenDispatcher

    dispatcher = HybridEPTokenDispatcher(
        HybridEPTokenDispatcher.Config(
            token_drop=TokenDropConfig(
                strategy="fixed_capacity",
                expert_capacity_tokens=quota * EP,
                drop_by_group_capacity=True,
                branchless_capacity_drop=branchless,
            )
        ).with_capacity_context(
            num_tokens=TOKENS, top_k=TOP_K, num_experts=EXPERTS, ep_size=EP
        )
    )

    def fragment(scores, ids):
        return dispatcher._drop_routes(
            scores, ids, num_experts=EXPERTS, ep_group=ep_group
        )

    return fragment


def reference_inputs(scores, ids):
    import torch

    weights = scores.detach().reshape(-1)
    ids = ids.reshape(-1)
    valid = (ids >= 0) & (weights != 0)
    buckets = torch.where(valid, ids, EXPERTS)
    counts = torch.bincount(buckets, minlength=EXPERTS + 1)
    return weights, buckets, valid, counts


def exact(actual, expected):
    import torch

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def parity(capacity, ep_group, rank, scores, ids):
    """Exercise real EP8 gates plus the native shape through compile and SAC."""
    import torch
    import torch.distributed as dist
    from torch.utils.checkpoint import checkpoint
    from torchtitan.distributed.activation_checkpoint import _sac_context_fn

    device = scores.device
    balanced = torch.arange(TOKENS * TOP_K, device=device).view_as(ids) % EXPERTS
    skewed = torch.arange(TOP_K, device=device).expand_as(ids).clone()
    forced_scores = torch.ones_like(scores)
    forced_scores[0, 0] = 0
    skewed[0, 1] = -1
    forced_cases = (
        ("one_skewed_rank", skewed if rank == 1 else balanced, rank == 1, False),
        ("seven_skewed_ranks", balanced if rank == 0 else skewed, rank != 0, True),
    )
    cases = [("native", scores, ids, NATIVE_QUOTA, None, False)] + [
        (name, forced_scores, forced_ids, TEST_QUOTA, local, global_)
        for name, forced_ids, local, global_ in forced_cases
    ]
    inputs = {}
    for name, values, routes, quota, local, global_ in cases:
        weights, buckets, valid, counts = reference_inputs(values, routes)
        totals = counts[:EXPERTS].clone()
        dist.all_reduce(totals, group=ep_group)
        local_actual = bool((counts[:EXPERTS] > quota).any())
        global_actual = bool((totals.view(EP, -1).sum(1) > quota * EXPERTS).any())
        assert local is None or local_actual == local, name
        assert global_actual == global_, name
        needs_drop = local_actual and global_actual
        keep = (
            capacity._sorted_capacity_mask(
                values, buckets, valid, num_experts=EXPERTS, capacity=quota
            )
            if needs_drop
            else valid
        ).view_as(routes)
        if needs_drop:
            # All live scores tie. Expert 0 starts at row 1 due to the zero score.
            exact(
                keep[:, 0],
                (torch.arange(TOKENS, device=device) >= 1)
                & (torch.arange(TOKENS, device=device) <= quota),
            )
        inputs[name] = (values, routes, quota, keep, local_actual, global_actual)

    upstream = torch.arange(TOKENS * TOP_K, device=device).view_as(scores) % 13
    upstream = upstream.to(scores.dtype) / 8
    receipts = []
    for quota, modes in (
        (NATIVE_QUOTA, (None, (), ("comm",))),
        (TEST_QUOTA, (None, ())),
    ):
        for save in modes:
            compiled = {}
            for variant, branchless in (("S", False), ("branchless", True)):
                fragment = make_fragment(branchless, quota, ep_group)
                if save is not None:

                    def checked(scores, ids, run=fragment, categories=save):
                        return checkpoint(
                            run,
                            scores,
                            ids,
                            use_reentrant=False,
                            early_stop=False,
                            context_fn=partial(_sac_context_fn, categories, ()),
                        )
                else:
                    checked = fragment
                compiled[variant] = torch.compile(checked, fullgraph=True)

            for name, (
                values,
                routes,
                quota_in,
                keep,
                local,
                global_,
            ) in inputs.items():
                if quota != quota_in:
                    continue
                row = {
                    "case": name,
                    "source_quota": quota,
                    "sac_save": save,
                    "local_overload": local,
                    "group_overload": global_,
                    "retained_routes": int(keep.sum()),
                    "variants": {},
                }
                for variant, fn in compiled.items():
                    x = values.detach().requires_grad_()
                    # Warm compilation before measuring the forward retention and
                    # collecting operator order. The first call is checked too.
                    for iteration in range(2):
                        x.grad = None
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU]
                        ) as profile:
                            with torch.profiler.record_function(
                                "capacity_component.forward"
                            ):
                                before = torch.cuda.memory_allocated()
                                y, dropped_ids, actual_keep = fn(x, routes)
                                retained = torch.cuda.memory_allocated() - before
                            with torch.profiler.record_function(
                                "capacity_component.backward"
                            ):
                                y.backward(upstream)
                        exact(actual_keep, keep)
                        exact(dropped_ids, routes.masked_fill(~keep, -1))
                        exact(y, values * keep)
                        exact(x.grad, upstream * keep)
                        del y, dropped_ids, actual_keep
                        if iteration == 0:
                            continue
                        order = {phase: [] for phase in ("forward", "backward")}
                        for event in profile.events():
                            if event.name not in (
                                "_c10d_functional::all_reduce",
                                "c10d::allreduce_",
                            ):
                                continue
                            parent = event.cpu_parent
                            while parent is not None:
                                phase = parent.name.removeprefix("capacity_component.")
                                if phase in order:
                                    order[phase].append(event.name)
                                    break
                                parent = parent.cpu_parent
                        if not order["forward"]:
                            raise AssertionError(
                                f"missing EP all-reduce for {name}, {save}, {variant}"
                            )
                        row["variants"][variant] = {
                            "collective_order": order,
                            "forward_allocated_delta_bytes": retained,
                            "exact_mask_ids_scores_gradients": True,
                        }
                if (
                    row["variants"]["S"]["collective_order"]
                    != row["variants"]["branchless"]["collective_order"]
                ):
                    raise AssertionError(
                        f"collective participation/order changed: {row}"
                    )
                receipts.append(row)
    return receipts


def kernel_resources(kernel):
    """Read the launch records for the installed Triton build, if exposed."""
    rows = []
    for state in getattr(kernel, "device_caches", {}).values():
        if not state or not isinstance(state[0], Mapping):
            continue
        for compiled in state[0].values():
            metadata = getattr(compiled, "metadata", None)
            if metadata is None:
                continue
            rows.append(
                {
                    "name": getattr(compiled, "name", None),
                    "num_warps": getattr(metadata, "num_warps", None),
                    "registers_per_thread": getattr(compiled, "n_regs", None),
                    "spills": getattr(compiled, "n_spills", None),
                    "shared_bytes_per_cta": getattr(metadata, "shared", None),
                }
            )
    return rows


def measure(operations, group, warmup_pairs, pairs):
    import torch
    import torch.distributed as dist

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    end.record()
    end.synchronize()
    result = {}
    for scope, variants in operations.items():
        for i in range(warmup_pairs):
            for name in tuple(variants) if i % 2 == 0 else tuple(reversed(variants)):
                variants[name]()
        torch.cuda.synchronize()
        result[scope] = {name: [] for name in variants}
        for i in range(pairs):
            for name in tuple(variants) if i % 2 == 0 else tuple(reversed(variants)):
                dist.barrier(group=group)
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                start.record()
                before = perf_counter_ns()
                output = variants[name]()
                submitted = perf_counter_ns()
                end.record()
                end.synchronize()
                completed = perf_counter_ns()
                sample = {
                    "pair": i,
                    "host_submit_us": (submitted - before) / 1000,
                    "host_complete_us": (completed - before) / 1000,
                    "cuda_elapsed_us": start.elapsed_time(end) * 1000,
                    "allocated_before_bytes": allocated,
                    "reserved_before_bytes": reserved,
                    "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                    "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
                }
                del output
                sample["allocated_after_release_bytes"] = torch.cuda.memory_allocated()
                result[scope][name].append(sample)
    return result


def profile_operations(operations, group, path):
    import torch
    import torch.distributed as dist

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as trace:
        for scope, variants in operations.items():
            for name, fn in variants.items():
                dist.barrier(group=group)
                torch.cuda.synchronize()
                with torch.profiler.record_function(
                    f"capacity_component.{scope}.{name}"
                ):
                    output = fn()
                    # Diagnostic boundary only. It bounds the device span for the
                    # trace and is excluded from the unprofiled timing samples.
                    torch.cuda.synchronize()
                    del output
    trace.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    names = {
        f"capacity_component.{scope}.{name}"
        for scope, variants in operations.items()
        for name in variants
    }
    summary = {}
    for marker in events:
        if marker.get("name") not in names or marker.get("ph") != "X":
            continue
        start, end = marker["ts"], marker["ts"] + marker["dur"]
        within = [event for event in events if start <= event.get("ts", -1) < end]
        summary[marker["name"]] = {
            "device_kernels_us": [
                {"name": event["name"], "duration_us": event["dur"]}
                for event in within
                if event.get("cat") == "kernel"
            ],
            "stream_synchronizations_us": [
                event["dur"]
                for event in within
                if event.get("name") == "cudaStreamSynchronize" and "dur" in event
            ],
            "device_to_host_copies": [
                {
                    "name": event["name"],
                    "duration_us": event["dur"],
                    "args": event.get("args"),
                }
                for event in within
                if event.get("cat") == "gpu_memcpy" and "DtoH" in event["name"]
            ],
        }
    if summary.keys() != names:
        raise RuntimeError(f"incomplete component profiler capture: {path}")
    return summary


def report(output, ranks):
    summary = {}
    for scope, variants in ranks[0]["samples"].items():
        summary[scope] = {}
        for name in variants:
            variant = [rank["samples"][scope][name] for rank in ranks]
            stats = {}
            for metric in ("host_submit_us", "host_complete_us", "cuda_elapsed_us"):
                slowest = [max(row[metric] for row in pair) for pair in zip(*variant)]
                stats[metric] = {
                    "min": min(slowest),
                    "median": median(slowest),
                    "max": max(slowest),
                }
            for metric in ("allocated_peak_bytes", "reserved_peak_bytes"):
                stats[metric] = max(row[metric] for rank in variant for row in rank)
            for memory in ("allocated", "reserved"):
                stats[f"{memory}_peak_delta_bytes"] = max(
                    row[f"{memory}_peak_bytes"] - row[f"{memory}_before_bytes"]
                    for rank in variant
                    for row in rank
                )
            stats["retained_allocation_delta_bytes"] = max(
                row["allocated_after_release_bytes"] - row["allocated_before_bytes"]
                for rank in variant
                for row in rank
            )
            summary[scope][name] = stats
    (output / "summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "ranks": list(range(EP)),
                "summary": summary,
                "aggregation": "slowest rank per paired sample, then min/median/max",
                "scope": "isolation starts with empty GPU queues and excludes dispatch and quantize",
                "raw_receipts": [f"rank{r}.json" for r in range(EP)],
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--capacity-source", type=Path)
    parser.add_argument("--warmup-pairs", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=50)
    args = parser.parse_args()
    if min(args.warmup_pairs, args.pairs) <= 0:
        parser.error("warmup and measured pair counts must be positive")
    if int(os.environ["WORLD_SIZE"]) != EP:
        raise RuntimeError("component replay requires exactly eight ranks")
    rank = int(os.environ["RANK"])
    cache = configure_cache(args.cache_root, rank)

    import torch
    import torch.distributed as dist
    import triton

    capacity = load_capacity(args.capacity_source)
    from torchtitan.experiments.four_fusion.routing_metadata import indices_to_map
    from torchtitan.ops.token_drop import _capacity_mask_kernel, capacity_mask

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", device_id=device)
    group = dist.new_group(list(range(EP)), backend="nccl")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        fixture_path = args.fixture_dir / f"rank{rank}.pt"
        fixture = torch.load(fixture_path, map_location=device, weights_only=True)
        scores, ids = fixture["scores"], fixture["expert_ids"]
        if (
            fixture["rank"] != rank
            or fixture["capacity_sha256"] != CAPACITY_SHA
            or fixture["num_experts"] != EXPERTS
            or fixture["source_quota"] != NATIVE_QUOTA
            or scores.shape != (TOKENS, TOP_K)
            or ids.shape != scores.shape
            or scores.dtype != torch.float32
            or ids.dtype != torch.int64
            or list(scores.stride()) != fixture["score_stride"]
        ):
            raise RuntimeError(
                f"native router fixture does not match S: {fixture_path}"
            )
        weights, buckets, valid, counts = reference_inputs(scores, ids)
        exact(valid, torch.ones_like(valid))
        if (
            weights.data_ptr() != scores.data_ptr()
            or weights.contiguous().data_ptr() != weights.data_ptr()
        ):
            raise RuntimeError(
                "the selector requires an input copy for the actual router layout"
            )

        checks = parity(capacity, group, rank, scores, ids)
        predicate = torch.zeros((), device=device, dtype=torch.bool)
        compiled = {
            name: torch.compile(
                make_fragment(is_gpu, NATIVE_QUOTA, group), fullgraph=True
            )
            for name, is_gpu in (("S", False), ("branchless", True))
        }

        def prep(fn):
            masked_scores, masked_ids, _ = fn(scores, ids)
            # Native launches this helper inside the opaque dispatch custom op.
            # Calling it separately here isolates the work before quantization.
            return indices_to_map(masked_ids, masked_scores, EXPERTS)

        operations = {
            "selector": {
                "S": lambda: capacity._conditional_capacity_mask(
                    scores, buckets, valid, predicate, EXPERTS, NATIVE_QUOTA
                ),
                "branchless": lambda: capacity_mask(
                    weights, buckets, valid, counts, predicate, NATIVE_QUOTA
                ),
            },
            "preparation": {name: partial(prep, fn) for name, fn in compiled.items()},
        }
        exact(operations["selector"]["S"](), operations["selector"]["branchless"]())
        for control, candidate in zip(
            operations["preparation"]["S"](), operations["preparation"]["branchless"]()
        ):
            exact(control, candidate)
        del control, candidate
        samples = measure(operations, group, args.warmup_pairs, args.pairs)
        profile = profile_operations(
            operations, group, args.output / f"rank{rank}_trace.json"
        )
        receipt = {
            "rank": rank,
            "torch": str(torch.__version__),
            "torch_git": torch.version.git_version,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(),
            "cache": cache,
            "capacity_module": capacity.__file__,
            "capacity_sha256": CAPACITY_SHA,
            "fixture": str(fixture_path),
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "weights_contiguous_without_copy": True,
            "selector_output_bytes": weights.numel(),
            "parity": checks,
            "samples": samples,
            "profile": profile,
            "selector_resources": kernel_resources(_capacity_mask_kernel),
        }
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        dist.barrier(group=group)
        if rank == 0:
            report(
                args.output,
                [
                    json.loads((args.output / f"rank{i}.json").read_text())
                    for i in range(EP)
                ],
            )
    finally:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
