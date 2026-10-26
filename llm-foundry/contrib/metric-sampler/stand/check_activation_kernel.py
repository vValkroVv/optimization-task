from __future__ import annotations

import argparse
import sys

import torch

from metric_sampler.capture.kernels import activation_ext
from stand.activation_kernel_checks.baseline import build_baseline
from stand.activation_kernel_checks.correctness import check_correctness
from stand.activation_kernel_checks.kernel_table import time_kernels
from stand.activation_kernel_checks.path_timings import time_extension_paths


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--correctness-only", action="store_true")
    modes.add_argument("--timing-only", action="store_true", help="only the table of the kernels' GPU times")
    parser.add_argument("--breakdown", action="store_true", help="the kernels of each timed call, with their times")
    parser.add_argument(
        "--baseline-ref", help="a commit whose extension the current one is compared with and timed against"
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    if not torch.cuda.is_available():
        print("this check needs a CUDA device", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")

    activation_ext.build()
    extension = activation_ext.load()
    if extension is None:
        print("activation_ext failed to build/load", file=sys.stderr)
        sys.exit(1)
    baseline = build_baseline(arguments.baseline_ref) if arguments.baseline_ref else None

    if not arguments.timing_only:
        check_correctness(device, extension, baseline, arguments.baseline_ref)
    if arguments.correctness_only:
        return

    time_kernels(extension, baseline, arguments.breakdown)
    if arguments.timing_only:
        return

    if time_extension_paths(extension):
        sys.exit(1)


if __name__ == "__main__":
    main()
