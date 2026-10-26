# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Static launch-parameter lookup for Triton kernels."""

import logging

import torch

logger = logging.getLogger(__name__)
_warned_keys: set[tuple[int, int | tuple]] = set()


def params_to_kernel_kwargs(
    static_params: dict,
    key: int | tuple,
    **default_kwargs,
) -> dict:
    """Return tuned launch parameters for a key or the provided defaults."""
    if key in static_params:
        return static_params[key]

    # Avoid a graph break from mutating the process-wide warning cache.
    if not torch.compiler.is_compiling():
        warn_key = (id(static_params), key)
        if warn_key not in _warned_keys:
            _warned_keys.add(warn_key)
            logger.warning(
                "No pre-tuned Triton launch parameters for key %s; "
                "falling back to defaults %s. Performance may be suboptimal.",
                key,
                default_kwargs,
            )

    if default_kwargs:
        return default_kwargs
    return {
        "num_warps": 4,
        "num_stages": 3,
        "num_ctas": 1,
        "maxnreg": None,
    }
