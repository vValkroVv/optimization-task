# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static launch-parameter tables for the float8 triton kernels.

The kernels in this package do not use triton autotuning (it would recompile
on every new dynamic shape under expert parallelism). Instead, each kernel
module carries a small table of pre-tuned launch parameters keyed by hidden
size, tuned on H100. ``params_to_kernel_kwargs`` looks up the table and falls
back to the caller-provided defaults for unknown keys.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_warned_keys: set[tuple[int, int | tuple]] = set()


def params_to_kernel_kwargs(
    static_params: dict,
    key: int | tuple,
    **default_kwargs,
) -> dict:
    """Look up pre-tuned triton launch parameters for *key*.

    Args:
        static_params: table mapping a shape key (typically hidden size) to a
            dict of launch kwargs (num_warps, num_stages, maxnreg, ...).
        key: the lookup key for the current launch.
        default_kwargs: fallback launch kwargs used when *key* is not in the
            table.

    Returns:
        The launch kwargs dict for the triton kernel launch.
    """
    if key not in static_params:
        # Skipped under torch.compile: mutating a module-level set.
        if not torch.compiler.is_compiling():
            warn_key = (id(static_params), key)
            if warn_key not in _warned_keys:
                _warned_keys.add(warn_key)
                logger.warning(
                    "No pre-tuned triton launch parameters for key %s; "
                    "falling back to defaults %s. Performance may be suboptimal.",
                    key,
                    default_kwargs,
                )
        if default_kwargs:
            return default_kwargs
        return dict(num_warps=4, num_stages=3, num_ctas=1, maxnreg=None)
    return static_params[key]
