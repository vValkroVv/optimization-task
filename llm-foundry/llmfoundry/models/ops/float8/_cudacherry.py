"""Single entry point to the cudacherry FP8 CUDA kernels.

The kernels used to live in ``cuda_kernels/`` as five separate pip packages built
from this repository.  They now live in cudacherry and are installed into the
image.  Everything in llm-foundry that needs a CUDA FP8 kernel imports it from
here, so there is exactly one place that knows cudacherry exists and exactly one
place where the environment is checked.

Importing this module is the guard: it is imported only when FP8 is actually
used, so a non-FP8 run never pays for it, and an FP8 run on an image without
cudacherry fails here -- at import, with an actionable message -- rather than
mid-training-step with a pybind error.  That ordering is the same lesson that
produced ``warmup_ext``: a surprise inside a backward pass is the worst place to
discover a missing dependency.
"""

from __future__ import annotations

import os

from llmfoundry.models.ops.float8.fp8_tensor import Float8BlockwiseQTensor

#: The module cudacherry must build its wrapper tensors from.
#:
#: llm-foundry and cudacherry each define their own ``Float8BlockwiseQTensor``.
#: They are kept identical on purpose (cudacherry has to work in repositories
#: that have no llm-foundry), and are collapsed into one *class object* here by
#: telling cudacherry to use ours.  ~22 ``isinstance`` sites depend on that.
QTENSOR_MODULE = "llmfoundry.models.ops.float8.fp8_tensor"

# Set rather than merely required: an operator who has to remember an environment
# variable will eventually not, and getting it wrong does not raise -- see
# _check_qtensor_binding below for what it costs.  setdefault so that a
# deliberate override is still possible; the check afterwards is what makes an
# override that breaks isinstance loud instead of silent.
os.environ.setdefault("CUDACHERRY_FP8_TENSOR_MODULE", QTENSOR_MODULE)

try:
    import cudacherry
except ImportError as exc:  # pragma: no cover - depends on the image
    raise ImportError(
        "FP8 training requires the cudacherry package, which is not installed. "
        "The FP8 CUDA kernels no longer live in this repository -- they are built "
        "into the training image. Rebuild or pull the current image; running FP8 "
        "on an older one will not work.\n"
        f"Original error: {exc}"
    ) from exc


def _check_qtensor_binding() -> None:
    """Fail if cudacherry resolves a different wrapper class than ours.

    This asserts the invariant directly rather than checking the environment
    variable that is supposed to produce it, so it also catches the case where
    something imported cudacherry earlier under a different setting.

    Deliberately ``raise`` and not ``assert``: ``python -O`` strips asserts, and
    this must not be strippable. A mismatch does not crash -- ``isinstance``
    quietly returns False, ``clear_tensor_data`` skips the wrapper branch, and
    ``_data``/``_scale`` stay referenced for the life of the step. The symptom is
    memory growth with no traceback, which is close to the hardest thing to trace
    back to its cause.
    """
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
            f"{QTENSOR_MODULE!r} and must be set before cudacherry is first "
            "imported. With two classes in the process, isinstance checks fail "
            "silently and quantized activations leak."
        )


_check_qtensor_binding()

# Re-exported under cudacherry's names, not llm-foundry's old ones. cudacherry
# does keep the old spellings as deprecated aliases, but importing through them
# would emit a DeprecationWarning on every process and leave two vocabularies in
# the codebase for no benefit.
#
# One exception, decided here so no call site has to decide it again: cudacherry
# calls the matmul `gemm`, which is unambiguous inside `cudacherry.gemm` and is
# not once imported bare into llm-foundry, where deep_gemm, grouped_gemm,
# _GroupedGemm and FP8Gemm all live in the same namespace.
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
