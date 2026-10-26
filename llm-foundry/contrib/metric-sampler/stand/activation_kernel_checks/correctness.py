from __future__ import annotations

import sys

import torch

from stand.activation_kernel_checks.baseline import check_baseline_parity
from stand.activation_kernel_checks.histogram_checks import (
    check_large_integer_counts,
    check_observed_layouts,
    check_padded_histogram_cancellation,
    check_tensor_statistics,
)
from stand.activation_kernel_checks.populations import make_tensors, selection_populations
from stand.activation_kernel_checks.quantile_checks import (
    check_quantile_first_pass,
    check_quantile_microbatches,
    check_raw_counts,
    check_selection_boundaries,
    check_selection_edges,
)
from stand.activation_kernel_checks.recompile_checks import (
    check_extension_no_slowdown_on_row_count,
    check_no_recompile_on_row_count,
)
from stand.activation_kernel_checks.reduction_checks import check_row_reductions
from stand.activation_kernel_checks.weight_norm_checks import check_weight_norm_batches


def check_correctness(device: torch.device, extension, baseline, baseline_commit: str | None) -> None:
    problems = check_tensor_statistics(device, extension)
    problems.extend(check_padded_histogram_cancellation(device))
    problems.extend(check_raw_counts(selection_populations(device), extension))
    problems.extend(check_quantile_microbatches(device, extension))
    problems.extend(check_quantile_first_pass(make_tensors(device), extension))
    problems.extend(check_observed_layouts(device, extension))
    problems.extend(check_row_reductions(device))
    problems.extend(check_weight_norm_batches(device, extension))
    problems.extend(check_selection_boundaries(device))
    problems.extend(check_selection_edges(device))
    problems.extend(check_large_integer_counts(device, extension))
    problems.extend(check_no_recompile_on_row_count(device))
    if extension is not None:
        problems.extend(check_extension_no_slowdown_on_row_count(device, extension))
    if baseline is not None:
        problems.extend(
            check_baseline_parity(make_tensors(device) + selection_populations(device), extension, baseline)
        )
    _report(problems, baseline, baseline_commit)


def _report(problems: list[str], baseline, baseline_commit: str | None) -> None:
    if problems:
        print(f"{len(problems)} mismatch(es):")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print("all integer counts match exactly; float sums satisfy rounding/overflow checks")
    print("bf16/fp16 raw counts and exact quantiles equal the CPU reference and the refining path bit "
          "for bit, and the sorted reference")
    if baseline is not None:
        print(f"every integer word equals that of {baseline_commit}")
