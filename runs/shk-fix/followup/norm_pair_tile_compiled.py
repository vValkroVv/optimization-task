"""Measure the norm-pair fusion in the compiled GIFT selective-AC replay."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import operator
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Node

from norm_f1_component import load_variant
from norm_pair_tile_ops import make_fused_pair
from norm_skip import NATIVE_SHAPE, digest, measure, mismatched_bytes


LOCAL_PREPARATION = {
    operator.getitem,
    torch.ops.aten.alias.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.prims.convert_element_type.default,
}


def is_local_preparation(node, device):
    if node.op != "call_function":
        return False
    if node.target in LOCAL_PREPARATION:
        return True
    value = node.meta.get("val")
    if type(value) is not FakeTensor or value.device != device:
        return False
    if node.target == torch.ops.aten._to_copy.default:
        source = node.args[0]
        original = source.meta.get("val") if isinstance(source, Node) else None
        return (
            type(original) is FakeTensor
            and original.device == device
            and not node.kwargs.get("non_blocking", False)
        )
    # The zero-centered norm computes 1 + weight.float() on this device.
    return (
        node.target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
        and tuple(value.shape) == (NATIVE_SHAPE[-1],)
        and value.dtype == torch.float32
        and all(
            not isinstance(arg, Node)
            or (
                type(arg.meta.get("val")) is FakeTensor
                and arg.meta["val"].device == device
            )
            for arg in node.args
        )
    )


def tuple_items(node):
    items = tuple(node.users)
    values = node.meta.get("val")
    if not isinstance(values, tuple) or any(
        item.op != "call_function"
        or item.target is not operator.getitem
        or len(item.args) != 2
        or item.args[0] is not node
        or item.kwargs
        or type(item.args[1]) is not int
        or not 0 <= item.args[1] < len(values)
        for item in items
    ):
        return None
    return items


class FuseNormPairAfterSAC(CustomGraphPass):
    """Keep the output indices chosen by AOT and fuse only an eligible edge."""

    def __init__(self, stats_pass, op, key):
        self.stats_pass = stats_pass
        self.op = op
        self.key = key
        self.rewrites = Counter()
        self.skipped = Counter()

    def uuid(self):
        return self.stats_pass.uuid(), self.key

    def __call__(self, graph):
        from torchtitan.models.common.norms.cute_gated_norm.op import (
            _forward,
            _forward_residual,
            _forward_stats,
        )

        self.stats_pass(graph)
        for consumer_op in (_forward, _forward_stats):
            consumers = graph.find_nodes(
                op="call_function", target=consumer_op._opoverload
            )
            for second in consumers:
                stats_only = consumer_op is _forward_stats
                if len(second.args) != (5 if stats_only else 6) or second.kwargs:
                    continue
                x = second.args[0]
                value = x.meta.get("val") if isinstance(x, Node) else None
                if (
                    type(value) is not FakeTensor
                    or value.device.type != "cuda"
                    or value.dtype != torch.bfloat16
                    or any(type(size) is not int for size in value.shape)
                    or tuple(value.shape) != NATIVE_SHAPE
                    or not value.is_contiguous()
                ):
                    continue
                if (
                    x.op != "call_function"
                    or x.target is not operator.getitem
                    or len(x.args) != 2
                ):
                    continue
                first, index = x.args
                if (
                    type(index) is not int
                    or index != 0
                    or not isinstance(first, Node)
                    or first.op != "call_function"
                    or first.target != _forward_residual._opoverload
                    or len(first.args) != 7
                    or first.kwargs
                ):
                    continue
                first_items = tuple_items(first)
                second_items = tuple_items(second)
                if first_items is None or second_items is None:
                    continue

                order = {node: i for i, node in enumerate(graph.nodes)}
                # Insert before the second norm, once both sets of weights are ready.
                # The first norm's other consumers must still follow the new call.
                if any(
                    user is not second and order[user] < order[second]
                    for item in first_items
                    for user in item.users
                ):
                    self.skipped["earlier_output_use"] += 1
                    continue
                # Unknown calls can have stream effects even with functional schemas.
                barriers = [
                    str(node.target)
                    for node in order
                    if order[first] < order[node] < order[second]
                    and not is_local_preparation(node, value.device)
                ]
                if barriers:
                    self.skipped.update(barriers)
                    continue
                second_args = (*second.args[1:], 0.0) if stats_only else second.args[1:]
                args = (*first.args, *second_args, stats_only)
                with graph.inserting_before(second):
                    merged = graph.call_function(self.op, args)
                    merged.meta["val"] = [*first.meta["val"], *second.meta["val"]]
                    merged.meta["eager_input_vals"] = (
                        tuple(
                            value.meta["val"] if isinstance(value, Node) else value
                            for value in args
                        ),
                        {},
                    )
                    for items, offset in ((first_items, 0), (second_items, 7)):
                        for item in items:
                            replacement = graph.call_function(
                                operator.getitem, (merged, item.args[1] + offset)
                            )
                            replacement.meta = dict(item.meta)
                            item.replace_all_uses_with(replacement)
                for item in second_items:
                    graph.erase_node(item)
                graph.erase_node(second)
                for item in first_items:
                    graph.erase_node(item)
                graph.erase_node(first)
                self.rewrites["stats_only" if stats_only else "live"] += 1
        graph.lint()


def import_replay(args):
    from torchtitan.distributed import compile as gift_compile

    sys.path.append(str(args.replay_root.resolve(strict=True)))
    from residual_replay_probe import build

    if Path(build.__code__.co_filename).resolve().parent != args.replay_root.resolve():
        raise RuntimeError("Expected the specified historical replay helper")
    with patch.object(gift_compile, "apply_compile") as capture:
        _, leaves, tangent, metadata = build(args.fixture, "live", "B")
    if capture.call_count != 1:
        raise RuntimeError("Expected one GIFT norm block")
    model, config = capture.call_args.args
    (segment,) = model.named_segments()
    return segment, config, leaves, tangent, metadata


def profile(block, backward, arm, args):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as trace:
        for index in range(args.profile_calls):
            with torch.profiler.record_function(f"norm_pair_compiled_{index}"):
                backward(block())
                torch.cuda.synchronize()
    result = Counter(
        event.name
        for event in trace.events()
        if event.device_type.name == "CUDA" and "Gn" in event.name
    )
    fused = sum(n for name, n in result.items() if "GnFwdOutWithNextStats_" in name)
    stats = sum(n for name, n in result.items() if "GnFwdStats_" in name)
    if (fused, stats) != (
        (0, 4 * args.profile_calls) if arm == "A" else (2 * args.profile_calls,) * 2
    ):
        raise RuntimeError(f"Unexpected norm-pair activation in {arm}: {result}")
    path = args.output.with_suffix(f".{arm}.trace.json")
    trace.export_chrome_trace(str(path))
    return {"kernels": result, "trace_sha256": digest(path)}


def run(args):
    from torchtitan.distributed.compile import apply_compile
    from torchtitan.models.common.norms.cute_gated_norm import compiler, forward
    from torchtitan.tools.utils import device_memory_usage

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    torch.cuda.set_device(0)
    deterministic = yaml.safe_load(args.resolved_config.read_bytes())["debug"][
        "deterministic"
    ]
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        torch.utils.deterministic.fill_uninitialized_memory = False
    switches = {
        "GIFT_CUTE_NORM_FUSE_RESIDUAL": "1",
        "GIFT_CUTE_NORM_SAC_STATS_ONLY": "1",
        "GIFT_FP8_QUANTIZE_WEIGHT_TAIL": "1",
        "GIFT_CUTE_NORM_BWD_SKIP": "0",
    }
    os.environ.update(switches)
    torch._inductor.config.triton.cudagraphs = False
    candidate = load_variant(forward, args.candidate, "forward_norm_pair_tile_reuse")
    fused_op = make_fused_pair(candidate)
    op_sha256 = digest(Path(__file__).with_name("norm_pair_tile_ops.py"))
    rewrite = FuseNormPairAfterSAC(
        compiler.GatedNormStatsOnlyPass(),
        fused_op._opoverload,
        (digest(Path(__file__)), op_sha256, digest(args.candidate)),
    )
    segment, config, leaves, tangent, metadata = import_replay(args)
    inputs = tuple(leaves.values())
    calls = {}

    def run_block(block):
        def call():
            return block(*inputs)

        return call

    for arm in ("A", "B"):
        block = deepcopy(segment.module)
        current = SimpleNamespace(**(vars(segment) | {"module": block}))
        wrapper = SimpleNamespace(named_segments=lambda: [current])
        if arm == "A":
            apply_compile(wrapper, config)
        else:
            with patch.object(compiler, "GatedNormStatsOnlyPass", return_value=rewrite):
                apply_compile(wrapper, config)
        calls[arm] = run_block(block)

    def differentiate(y):
        return torch.autograd.grad(y, inputs, tangent)

    reference = calls["A"]()
    reference = (reference, *differentiate(reference))
    trial = calls["B"]()
    trial = (trial, *differentiate(trial))
    parity = mismatched_bytes(("output", *leaves), reference, trial)
    if any(parity.values()):
        raise RuntimeError(f"Compiled norm pair changed outputs or gradients: {parity}")
    if rewrite.rewrites != Counter({"live": 1, "stats_only": 1}):
        raise RuntimeError(
            f"Expected one fusion in forward and in replay: {rewrite.rewrites}. "
            f"Skipped: {rewrite.skipped}"
        )
    del reference, trial

    pairs = []
    with torch._dynamo.config.patch(error_on_recompile=True):
        for arm in ("A", "B"):
            for _ in range(args.warmup):
                differentiate(calls[arm]())
                torch.cuda.synchronize()
        for index in range(args.pairs):
            order = ("A", "B") if index % 2 == 0 else ("B", "A")
            blocks = {}
            for arm in order:
                torch.cuda.empty_cache()
                for _ in range(4):
                    differentiate(calls[arm]())
                blocks[arm] = measure(calls[arm], differentiate, args.samples)
                _, blocks[arm]["device_peak_gib"] = device_memory_usage(
                    torch.cuda.memory_stats()
                )
            pairs.append({"order": order, "arms": blocks})
        profiles = {
            arm: profile(calls[arm], differentiate, arm, args) for arm in ("A", "B")
        }

    with args.output.open("x") as out:
        json.dump(
            {
                "scope": "compiled_residual_then_plain_norm_with_selective_ac",
                "hostname": platform.node(),
                "gpu": torch.cuda.get_device_name(),
                "torch_revision": torch.version.git_version,
                "switches": switches,
                "fixture_sha256": digest(args.fixture),
                "resolved_config_sha256": digest(args.resolved_config),
                "runner_sha256": digest(Path(__file__)),
                "op_sha256": op_sha256,
                "candidate_sha256": digest(args.candidate),
                "capture": metadata,
                "rewrites": rewrite.rewrites,
                "warmup": args.warmup,
                "samples": args.samples,
                "profile_calls": args.profile_calls,
                "mismatched_bytes": parity,
                "profiles": profiles,
                "pairs": pairs,
            },
            out,
            indent=2,
        )
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "resolved-config", "replay-root", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--profile-calls", type=int, default=8)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.pairs, args.profile_calls) < 1:
        parser.error("Replay counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
