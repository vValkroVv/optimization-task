from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_silu_and_mul_clamp_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "silu_and_mul_clamp",
        *args,
        cuda_files=["elementwise/silu_and_mul_clamp.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulClampKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


def silu_and_mul_clamp(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    _jit_silu_and_mul_clamp_module(input.dtype).run(input, output, float(swiglu_limit))
