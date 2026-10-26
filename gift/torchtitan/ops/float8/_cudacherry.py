# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single entry point to the cudacherry FP8 CUDA kernels.

Importing this module is the guard: it does ``import cudacherry`` and fails
here -- at import, with an actionable message -- rather than mid-step with a
pybind error. Import it lazily (inside the function/method that launches a
kernel) so a non-FP8 or CPU-only run never triggers the guard.
"""

from __future__ import annotations

import os

from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor

# cudacherry ships its own Float8BlockwiseQTensor so it can run without
# torchtitan. Bind it to ours so the ~22 ``isinstance`` sites see one class;
# a mismatch makes ``isinstance`` silently False and leaks quantized storage.
QTENSOR_MODULE = "torchtitan.ops.float8.fp8_tensor"
os.environ.setdefault("CUDACHERRY_FP8_TENSOR_MODULE", QTENSOR_MODULE)

try:
    import cudacherry  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the image
    raise ImportError(
        "FP8 training requires the cudacherry package, which is not installed. "
        "The FP8 CUDA kernels are built into the training image; rebuild or pull "
        "the current image.\n"
        f"Original error: {exc}"
    ) from exc


def _check_qtensor_binding() -> None:
    from cudacherry.tensor.binding import qtensor_class

    resolved = qtensor_class()
    if resolved is not Float8BlockwiseQTensor:
        raise RuntimeError(
            "cudacherry is bound to the wrong Float8BlockwiseQTensor class.\n"
            f"  expected: {Float8BlockwiseQTensor.__module__}."
            f"{Float8BlockwiseQTensor.__qualname__}\n"
            f"  resolved: {resolved.__module__}.{resolved.__qualname__}\n"
            f"CUDACHERRY_FP8_TENSOR_MODULE is "
            f"{os.environ.get('CUDACHERRY_FP8_TENSOR_MODULE')!r}; it must be "
            f"{QTENSOR_MODULE!r} and set before cudacherry is first imported."
        )


_check_qtensor_binding()

from cudacherry.gemm import gemm as fp8_gemm  # noqa: E402
from cudacherry.quantization import (  # noqa: E402
    quantize_blk128x128,
    quantize_vec128,
    requantize_rowwise_to_columnwise,
    scaling_aware_fp8_transpose,
    swiglu_quantize,
)


__all__ = [
    "fp8_gemm",
    "quantize_blk128x128",
    "quantize_vec128",
    "requantize_rowwise_to_columnwise",
    "scaling_aware_fp8_transpose",
    "swiglu_quantize",
]
