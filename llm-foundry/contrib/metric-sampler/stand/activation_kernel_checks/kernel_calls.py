from __future__ import annotations

import torch

from metric_sampler.contract.metrics import ACTIVATION_LOG2_BINS
from stand.activation_kernel_checks.fake_recorder import (
    AMAX_OFFSET,
    COUNTS_OFFSET,
    MOMENTS_OFFSET,
    POWER_SUMS_OFFSET,
    QUANTILE_COUNTS_OFFSET,
    QUANTILES_OFFSET,
)


def launch_activation_stats_kernel(*arguments, **options):
    # Imported here: the kernel module needs triton, which the unit tests of the oracle run without.
    from metric_sampler.capture.kernels import activation_kernel

    return activation_kernel.launch_activation_stats_kernel(*arguments, **options)


def launch_statistics_kernel(matrix: torch.Tensor, words: torch.Tensor, **options):
    return launch_activation_stats_kernel(
        matrix, words, MOMENTS_OFFSET, COUNTS_OFFSET, AMAX_OFFSET, POWER_SUMS_OFFSET, ACTIVATION_LOG2_BINS, **options
    )


def observe_statistics(
    extension, tensor: torch.Tensor, words: torch.Tensor, quantile_first_pass: bool, *key_histogram: bool
):
    return extension.observe(
        tensor, words, MOMENTS_OFFSET, COUNTS_OFFSET, AMAX_OFFSET, POWER_SUMS_OFFSET,
        int(ACTIVATION_LOG2_BINS.start), ACTIVATION_LOG2_BINS.count, quantile_first_pass, *key_histogram,
    )


def observe_with_quantiles(extension, tensor: torch.Tensor, words: torch.Tensor, *key_histogram: bool,
                           offset: int = 0) -> None:
    """The extension's call that records the statistics and the exact quantiles of `tensor` into `words`, from
    the word `offset`."""
    extension.observe_with_quantiles(
        tensor, words, offset + MOMENTS_OFFSET, offset + COUNTS_OFFSET, offset + AMAX_OFFSET,
        offset + POWER_SUMS_OFFSET, int(ACTIVATION_LOG2_BINS.start), ACTIVATION_LOG2_BINS.count,
        offset + QUANTILES_OFFSET, offset + QUANTILE_COUNTS_OFFSET, *key_histogram,
    )
