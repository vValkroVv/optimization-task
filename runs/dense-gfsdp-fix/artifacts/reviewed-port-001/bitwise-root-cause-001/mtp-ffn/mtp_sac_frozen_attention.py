#!/usr/bin/env python3
"""Compare the actual MTP forward and SAC replay at native operator boundaries."""

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import mtp_ffn_root_cause as probe
from mtp_cudnn_repeat import freeze_operand


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


def observe(original, base, harness, fixture, model, device, rank, taps, phase, substitute):
    import torch
    from torch._ops import OpOverload
    from torch._subclasses.fake_tensor import FakeTensor

    names = {
        "aten::_scaled_dot_product_cudnn_attention": "cudnn_attention",
        "torchtitan::fp8_dense_gemm": "fp8_dense_gemm",
        "torchtitan::cute_gated_norm_fwd": "cute_norm",
        "cudacherry::quantize_vec128": "quantize_vec128",
        "cudacherry::swiglu_quant_forward": "swiglu_quant",
    }
    forward = model.forward
    call = OpOverload.__call__
    ordinary, replay = [], []
    sealed = False
    first_backward = False
    attention_calls = []
    saved_attention = None
    fixed_call = None
    replaced = False
    norm_input_cotangents = []

    def seal(*args, **kwargs):
        nonlocal sealed
        output = forward(*args, **kwargs)
        sealed = True
        return output

    def record(kind, inputs, output):
        if not first_backward:
            (replay if sealed else ordinary).append(event(kind, inputs, output))

    def traced_call(overload, *args, **kwargs):
        nonlocal first_backward, saved_attention, fixed_call, replaced
        output = call(overload, *args, **kwargs)
        if first_backward and len(norm_input_cotangents) == 2:
            return output
        # Inductor also invokes real operators during compilation and autotuning.
        # Only the public op launched from the emitted Python wrapper belongs here.
        caller = sys._getframe(1)
        if caller.f_code.co_name != "call" or "assert_size_stride" not in caller.f_globals:
            return output
        name = overload.name()
        if sealed and name == "torchtitan::cute_gated_norm_bwd":
            first_backward = True
            norm_input_cotangents.append(snapshot(output[0]))
        kind = names.get(name)
        if kind is None or not args[0].is_cuda or isinstance(args[0], FakeTensor):
            return output
        record(kind, args, output)
        if kind == "cudnn_attention":
            attention_calls.append({
                "replay": sealed,
                "input_strides": [list(value.stride()) for value in args[:3]],
                "input_alignment_mod_256": [value.data_ptr() % 256 for value in args[:3]],
                "input_storage_offsets": [value.storage_offset() for value in args[:3]],
                "output_strides": [list(value.stride()) for value in output[:2]],
                "output_alignment_mod_256": [value.data_ptr() % 256 for value in output[:2]],
                "options": [*args[3:], kwargs],
            })
            if not sealed:
                saved_attention = tuple(snapshot(value) for value in output[:2])
                fixed_call = overload, tuple(freeze_operand(torch, value) for value in args[:3]), args[3:], kwargs
            elif substitute:
                probe.require(saved_attention is not None and not replaced, "Expected one attention replay")
                replaced = True
                return (*saved_attention, *output[2:])
        return output

    with ExitStack() as patches:
        patches.enter_context(patch.object(model, "forward", seal))
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
        and len(norm_input_cotangents) == 2
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
    probe.require(replaced == substitute, "Attention replay substitution did not run")
    ffn_source = ordinary if section == "suffix" else replay
    ffn_input = [tensors["in0"] for name, tensors in ffn_source if name == "quantize_vec128"][-1]
    native_attention, fixed_qkv, other_args, kwargs = fixed_call
    with torch.no_grad():
        direct = [
            native_attention(*fixed_qkv, *other_args, **kwargs)[:2] for _ in range(5)
        ]
    direct_receipt = [
        base.differences(
            {name: value.cpu() for name, value in zip(("out", "lse"), direct[0], strict=True)},
            {name: value.cpu() for name, value in zip(("out", "lse"), run, strict=True)},
        )
        for run in direct[1:]
    ]
    metadata["sac_trace"] = {
        "section_before_first_backward_op": section,
        "events": rows,
        "attention_calls": attention_calls,
        "replaced_replay_attention_with_original": replaced,
        "fixed_attention_forward_repetitions": direct_receipt,
        "local_ffn_vjp": {
            "input_sha256": base.digest(ffn_input.cpu()),
            "incoming_cotangent_sha256": base.digest(norm_input_cotangents[1].cpu()),
            "pre_reduce_weight_grad_sha256": {
                key: base.digest(value)
                for key, value in values.items()
                if key.startswith("pre_reduce/mtp.layers.0.block.feed_forward.")
            },
        },
    }
    return receipt, values, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    original = probe.measure
    arms = iter(("A", "AA", "B"))
    argv = [
        __file__, "--phase", "pre-reduce", "--source-root", args.source_root,
        "--fixture", args.fixture, "--output", args.output,
    ]

    def measure(*args):
        return observe(original, *args, substitute=next(arms) == "AA")

    with patch.object(probe, "measure", measure), patch.object(sys, "argv", argv):
        probe.main()


if __name__ == "__main__":
    main()
