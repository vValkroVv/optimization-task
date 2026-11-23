#!/usr/bin/env python3
"""Compare the actual MTP forward and SAC replay at native operator boundaries."""

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import mtp_ffn_root_cause as probe


def snapshot(value):
    return value.detach().clone()


def event(op, args, output):
    tensors = {
        f"out{i}": value
        for i, value in enumerate(output if isinstance(output, tuple) else (output,))
        if hasattr(value, "dtype") and value.ndim > 0
    }
    tensors.update(
        {f"in{i}": value for i, value in enumerate(args) if hasattr(value, "dtype")}
    )
    return op, {name: snapshot(value) for name, value in tensors.items()}


def observe(original, base, harness, fixture, model, device, rank, taps, phase):
    import torch
    from torch._ops import OpOverload
    from torch._subclasses.fake_tensor import FakeTensor
    from torchtitan.models.common.norms.cute_gated_norm import op

    names = {
        torch.ops.aten._scaled_dot_product_cudnn_attention.default: "cudnn_attention",
        torch.ops.torchtitan.fp8_dense_gemm.default: "fp8_dense_gemm",
        torch.ops.cudacherry.quantize_vec128.default: "quantize_vec128",
    }
    forward = model.forward
    load = op._load_kernels
    call = OpOverload.__call__
    ordinary, replay = [], []
    sealed = False
    first_backward = False

    def seal(*args, **kwargs):
        nonlocal sealed
        output = forward(*args, **kwargs)
        sealed = True
        return output

    def record(kind, inputs, output):
        if not first_backward:
            (replay if sealed else ordinary).append(event(kind, inputs, output))

    def traced_call(overload, *args, **kwargs):
        output = call(overload, *args, **kwargs)
        if (
            not first_backward
            and overload in names
            and args[0].is_cuda
            and not isinstance(args[0], FakeTensor)
        ):
            record(names[overload], args, output)
        return output

    def traced_kernels():
        norm, backward = load()

        def norm_trace(*args, **kwargs):
            output = norm(*args, **kwargs)
            record("cute_norm", args[:1], output)
            return output

        def backward_trace(*args, **kwargs):
            nonlocal first_backward
            first_backward = True
            return backward(*args, **kwargs)

        return norm_trace, backward_trace

    with ExitStack() as patches:
        patches.enter_context(patch.object(model, "forward", seal))
        patches.enter_context(patch.object(op, "_load_kernels", traced_kernels))
        patches.enter_context(patch.object(OpOverload, "__call__", traced_call))
        receipt, values, metadata = original(base, harness, fixture, model, device, rank, taps, phase)

    # A replays one region. B replays the suffix first, whose two norm operands
    # are the final two ordinary forward events.
    if replay and replay[0][0] == "cute_norm" and len(replay) == 2:
        section = "suffix"
        expected = ordinary[-2:]
    else:
        section = "whole"
        expected = ordinary
    probe.require(
        sealed
        and first_backward
        and sum(name == "cute_norm" for name, _ in ordinary) == 8
        and any(name == "cudnn_attention" for name, _ in ordinary)
        and any(name == "fp8_dense_gemm" for name, _ in ordinary)
        and [kind for kind, _ in expected] == [kind for kind, _ in replay],
        f"Unmatched MTP forward/replay events: {[k for k, _ in ordinary]} / {[k for k, _ in replay]}",
    )
    rows = []
    for (kind, original_tensors), (_, replay_tensors) in zip(expected, replay, strict=True):
        diffs = {}
        shapes = {}
        for name, value in original_tensors.items():
            observed = replay_tensors[name]
            shapes[name] = list(value.shape)
            count = value.contiguous().view(torch.uint8).ne(
                observed.contiguous().view(torch.uint8)
            ).count_nonzero().item()
            if count:
                diffs[name] = count
        rows.append({"op": kind, "shapes": shapes, "different_bytes": diffs})
    metadata["sac_trace"] = {"section_before_first_backward_op": section, "events": rows}
    return receipt, values, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    original = probe.measure
    argv = [
        __file__, "--phase", "pre-reduce", "--source-root", args.source_root,
        "--fixture", args.fixture, "--output", args.output,
    ]

    def measure(*args):
        return observe(original, *args)

    with patch.object(probe, "measure", measure), patch.object(sys, "argv", argv):
        probe.main()


if __name__ == "__main__":
    main()
