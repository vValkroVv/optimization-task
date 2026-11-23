#!/usr/bin/env python3
"""Confirm logical w2 wgrad width through the compiled dense FFN with SAC."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch


CHECKOUT = Path(__file__).resolve().parents[3]
TAIL = "GIFT_FP8_QUANTIZE_WEIGHT_TAIL"
LOGICAL_WIDTH = "GIFT_FP8_WGRAD_LOGICAL_WIDTH"


def verify_width(trace_path, arm):
    events = json.loads(Path(trace_path).read_text())["traceEvents"]
    widths = []
    for event in events:
        if event["name"] != "torchtitan::fp8_dense_gemm":
            continue
        dims = event.get("args", {}).get("Input Dims", [])
        if not dims or len(dims[0]) != 3:
            continue
        width, batch, length = dims[0]
        if (batch, length) == (2, 4096) and width in (10944, 11008):
            widths.append(width)
    expected = 10944 if arm == "B" else 11008
    if widths != [expected]:
        raise RuntimeError(
            f"expected one compiled w2 wgrad operand of width {expected}, got {widths}"
        )
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-scripts", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.arm == "B" and args.reference is None:
        parser.error("arm B requires a profiled arm A receipt")

    expected_logical_width = args.arm == "B"
    sys.path.insert(0, str(args.shared_scripts.resolve(strict=True)))
    import dense
    import measure
    from torchtitan.ops.float8 import cuda_gemm

    if not Path(cuda_gemm.__file__).resolve().is_relative_to(CHECKOUT / "gift"):
        raise RuntimeError(f"replay imported the wrong FP8Linear: {cuda_gemm.__file__}")
    dense.ROOT = CHECKOUT
    original_build = dense.build
    original_validate_ffn = dense.validate_ffn
    original_verify_kernels = dense.verify_kernels
    original_profile = measure.profile

    def build_with_tail(case, arm, fixture_path):
        os.environ[TAIL] = "1"
        os.environ[LOGICAL_WIDTH] = str(int(arm == "B"))
        return original_build(case, "B", fixture_path)

    def validate_switches(ffn, tail):
        original_validate_ffn(ffn, tail)
        if any(
            linear._wgrad_logical_width != expected_logical_width
            for linear in (ffn.w1, ffn.w2, ffn.w3)
        ):
            raise RuntimeError("FP8Linear did not bind the logical wgrad width switch")

    def check_tail_kernels(case, _arm, forward, backward):
        return original_verify_kernels(case, "B", forward, backward)

    def profile_width(*profile_args):
        with patch.object(
            torch.profiler,
            "profile",
            partial(torch.profiler.profile, record_shapes=True),
        ):
            profile = original_profile(*profile_args)
        profile["w2_wgrad_operand_width"] = verify_width(profile["path"], args.arm)
        return profile

    dense.build = build_with_tail
    dense.validate_ffn = validate_switches
    dense.verify_kernels = check_tail_kernels
    measure.profile = profile_width
    sys.argv = [
        sys.argv[0],
        "--case",
        "dense-fp8",
        "--arm",
        args.arm,
        "--fixture",
        str(args.fixture),
        "--resolved-config",
        str(args.resolved_config),
        "--output",
        str(args.output),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--profile",
    ]
    measure.main()

    receipt = json.loads(args.output.read_text())
    receipt["switches"][LOGICAL_WIDTH] = os.environ[LOGICAL_WIDTH]
    receipt["implementation_path"] = str(Path(cuda_gemm.__file__).resolve())
    if args.reference is not None:
        reference = json.loads(args.reference.read_text())
        if (
            reference["arm"] != "A"
            or reference["switches"][TAIL] != "1"
            or reference["trace"]["w2_wgrad_operand_width"] != 11008
            or reference["fixture"] != receipt["fixture"]
            or reference["resolved_config_sha256"] != receipt["resolved_config_sha256"]
        ):
            raise ValueError(
                "expected a matched reference with the accepted tail enabled"
            )
        mismatches = [
            name
            for name, signature in receipt["output_signatures"].items()
            if reference["output_signatures"].get(name) != signature
        ]
        receipt["reference"] = str(args.reference)
        receipt["bitwise_mismatches"] = mismatches
        receipt["bitwise_output_and_all_gradients"] = not mismatches
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt.get("bitwise_output_and_all_gradients", True):
        raise RuntimeError(f"dense FFN signature mismatch: {mismatches}")


if __name__ == "__main__":
    main()
