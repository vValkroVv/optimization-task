#!/usr/bin/env python3
"""Inspect the actual opaque terminal CuTe gated norm backward call."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import mtp_ffn_root_cause as root_probe


OPERANDS = ("dy", "x", "rho", "p", "w_f", "gd_bf", "gd_t", "gu_t")
OUTPUTS = ("dx", "dweight", "dgate_up", "dgate_down")
NORM_FQN = "mtp.layers.0.final_norm"
EXPECTED_NORMS = 8


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def copy(tensor):
    return tensor.detach().clone()


def same_bytes(base, left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and base.digest(left.cpu()) == base.digest(right.cpu())
    )


class RuntimeCapture:
    def __init__(self, load):
        self.load = load
        self.forward_count = 0
        self.backward_count = 0
        self.last_forward = None
        self.last_forward_storage = None
        self.terminal_primal = None
        self.last_replay_storage = None
        self.sealed = False
        self.first_backward = None

    def seal(self):
        require(
            self.last_forward is not None and not self.sealed,
            "Missing ordinary terminal CuTe forward",
        )
        x, y, saved = self.last_forward
        self.last_forward_storage = tuple(value.data_ptr() for value in saved)
        self.terminal_primal = tuple(copy(value) for value in (y, x, *saved))
        self.last_forward = None
        self.sealed = True

    def kernels(self):
        native_forward, native_backward = self.load()

        def forward(*args, **kwargs):
            result = native_forward(*args, **kwargs)
            saved = (*result[1:3], *result[3])
            if not self.sealed:
                self.forward_count += 1
                # Keep only the most recent forward. The root seals this when
                # it returns the actual output of the single native MTP head.
                self.last_forward = (args[0], result[0], saved)
            elif self.backward_count == 0:
                self.last_replay_storage = tuple(value.data_ptr() for value in saved)
            return result

        def backward(*args, **kwargs):
            self.backward_count += 1
            if self.backward_count != 1:
                return native_backward(*args, **kwargs)
            require(
                self.sealed and len(args) == 10 and not kwargs,
                "Unexpected first CuTe backward call",
            )
            operands = tuple(copy(value) for value in args[:8])
            layout = [
                {
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                    "alignment_mod_16": value.data_ptr() % 16,
                }
                for value in args[:8]
            ]
            require(
                all(
                    original.stride() == frozen.stride()
                    for original, frozen in zip(args, operands)
                ),
                "Frozen input strides differ from the native call",
            )
            storage = tuple(value.data_ptr() for value in args[2:8])
            result = native_backward(*args, **kwargs)
            self.first_backward = {
                "native": native_backward,
                "inputs": operands,
                "static": args[8:],
                "outputs": tuple(copy(value) for value in result),
                "layout": layout,
                "matches_terminal_forward_saved_storage": storage
                == self.last_forward_storage,
                "matches_last_recomputed_forward_saved_storage": storage
                == self.last_replay_storage,
            }
            return result

        return forward, backward


def final_norm_weights(model, harness):
    names = {
        harness.canonical(name): parameter
        for name, parameter in model.named_parameters()
    }
    modules = [
        module
        for name, module in model.named_modules()
        if harness.canonical(name) == NORM_FQN
        and getattr(module, "norm_backend", None) == "cute"
    ]
    require(
        len(modules) == 1 and modules[0].zero_centered,
        "Expected the native zero-centered CuTe final norm",
    )
    return tuple(
        harness.cpu_tensor(names[f"{NORM_FQN}.{name}"])
        for name in ("weight", "gate_up_weight", "gate_down_weight")
    ), modules[0].gating_weight


def cpu_tensors(names, values):
    return dict(zip(names, (value.cpu() for value in values), strict=True))


def observe(
    original_measure,
    base,
    harness,
    fixture,
    model,
    device,
    rank,
    taps,
    phase,
    fixed,
    repeats,
):
    import torch
    from torchtitan.models.common.norms.cute_gated_norm import op

    require(
        phase == "pre-reduce" and taps is None,
        "Runtime CuTe probe must use the unmodified compile graphs",
    )
    (weight, gate_up, gate_down), gating = final_norm_weights(model, harness)
    capture = RuntimeCapture(op._load_kernels)
    model_forward = model.forward

    def seal_forward(*args, **kwargs):
        require(not capture.sealed, "Expected exactly one MTP forward")
        output = model_forward(*args, **kwargs)
        capture.seal()
        return output

    with (
        patch.object(op, "_load_kernels", capture.kernels),
        patch.object(model, "forward", seal_forward),
    ):
        fixture_receipt, tensors, metadata = original_measure(
            base, harness, fixture, model, device, rank, taps, phase
        )

    first = capture.first_backward
    require(
        first is not None
        and capture.forward_count == capture.backward_count == EXPECTED_NORMS,
        f"Expected eight ordinary forward/backward CuTe norms, got "
        f"{capture.forward_count}/{capture.backward_count}",
    )
    inputs = cpu_tensors(OPERANDS, first["inputs"])
    outputs = cpu_tensors(OUTPUTS, first["outputs"])
    ordinary = cpu_tensors(("y", *OPERANDS[1:]), capture.terminal_primal)
    prepared = (
        (1.0 + weight.to(torch.bfloat16).float(), inputs["w_f"]),
        (gate_down.to(torch.bfloat16), inputs["gd_bf"]),
        (gate_down.to(torch.bfloat16).t(), inputs["gd_t"]),
        (gate_up.to(torch.bfloat16).t(), inputs["gu_t"]),
    )
    expected_dy = tensors["backward/component_cotangent"].to(inputs["dy"].dtype)
    require(
        same_bytes(base, ordinary["y"], tensors["forward/component"])
        and same_bytes(base, inputs["dy"], expected_dy)
        and first["static"][0] == gating
        and all(same_bytes(base, a, b) for a, b in prepared),
        "First opaque backward could not be attributed to the native MTP final norm",
    )

    # These calls happen only after the original component autograd and GFSDP
    # have completed. Each replay launches with its own native scratch workspace.
    native = first["native"]
    observed_args = (*first["inputs"], *first["static"])
    fixed.setdefault("args", observed_args)
    require(
        first["static"] == fixed["args"][8:],
        "Fixed CuTe scalar or dtype contract differs",
    )
    with torch.no_grad(), torch.cuda.device(device):
        replay = cpu_tensors(OUTPUTS, native(*observed_args))
        fixed_reference = cpu_tensors(OUTPUTS, native(*fixed["args"]))
        fixed_differences = [
            base.differences(
                fixed_reference, cpu_tensors(OUTPUTS, native(*fixed["args"]))
            )
            for _ in range(repeats - 1)
        ]

    for name, group in (
        ("input", inputs),
        ("output", outputs),
        ("ordinary_forward", ordinary),
        ("replay_observed", replay),
        ("replay_fixed", fixed_reference),
    ):
        tensors.update({f"cute/{name}/{key}": value for key, value in group.items()})
    metadata["cute"] = {
        "fqn": NORM_FQN,
        "selected_backward_ordinal": 1,
        "ordinary_forward_count": capture.forward_count,
        "backward_count": capture.backward_count,
        "matches_terminal_forward_saved_storage": first[
            "matches_terminal_forward_saved_storage"
        ],
        "matches_last_recomputed_forward_saved_storage": first[
            "matches_last_recomputed_forward_saved_storage"
        ],
        "native_input_layout": dict(zip(OPERANDS, first["layout"], strict=True)),
        "native_gating_weight": first["static"][0],
        "native_gradient_dtypes": [str(dtype) for dtype in first["static"][1]],
        "native_input_sha256": {
            name: base.digest(value) for name, value in inputs.items()
        },
        "native_output_sha256": {
            name: base.digest(value) for name, value in outputs.items()
        },
        "terminal_ordinary_vs_backward_saved_operands": base.differences(
            {name: ordinary[name] for name in OPERANDS[1:]},
            {name: inputs[name] for name in OPERANDS[1:]},
        ),
        "observed_vs_own_direct_replay": base.differences(outputs, replay),
        "common_first_arm_operand_replay_max_calls": repeats,
        "fixed_direct_repeats_vs_first": fixed_differences,
    }
    return fixture_receipt, tensors, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-ffn-dir")
    parser.add_argument("--repeats", type=int, choices=(2, 3, 4, 5), default=3)
    args = parser.parse_args()
    original = root_probe.measure
    fixed = {}

    def measurement(*measure_args):
        return observe(original, *measure_args, fixed=fixed, repeats=args.repeats)

    argv = [
        str(Path(__file__).resolve()),
        "--phase",
        "pre-reduce",
        "--source-root",
        args.source_root,
        "--fixture",
        args.fixture,
        "--output",
        args.output,
    ]
    if args.base_ffn_dir:
        argv.extend(("--base-ffn-dir", args.base_ffn_dir))
    with (
        patch.object(root_probe, "measure", measurement),
        patch.object(sys, "argv", argv),
    ):
        root_probe.main()


if __name__ == "__main__":
    main()
