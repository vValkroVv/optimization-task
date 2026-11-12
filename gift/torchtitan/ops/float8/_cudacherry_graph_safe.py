# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep dense FP8 GEMMs outside CUDA Graph capture without changing the backend.

The opaque operations are always tagged unsafe, for both CUDA Graph controls.
They delegate to CudaCherry's public GEMM entry point, which remains responsible
for validation, layout selection, dtype defaults, accumulation and workspace
behavior. The installed GEMM was already an opaque compiler boundary; this adds
one outer custom-op dispatch per call. Graph partitioning can add host launches
and change intermediate lifetimes, so native capture safety and performance
remain runtime gates.

The operator bodies are CUDA-only. CPU implementations used by unit tests are
temporary test doubles; this module does not provide a CPU GEMM fallback.
"""

from __future__ import annotations

import torch

from cudacherry.gemm import gemm as _vendor_gemm
from cudacherry.tensor import unpack_operand

from .fp8_tensor import Float8BlockwiseQTensor, Float8OperandTuple

GemmOperand = Float8BlockwiseQTensor | Float8OperandTuple


def _call_vendor_gemm(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    a_layout: int,
    a_block_scaling_dim: int,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    b_layout: int,
    b_block_scaling_dim: int,
    out: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    alpha: float,
    accumulate: bool,
    use_split_accumulator: bool,
    allow_tf32: bool,
    workspace_size: int,
) -> torch.Tensor:
    """Rebuild the public operand tuples and preserve the vendor call path."""
    return _vendor_gemm(
        (a_data, a_scale, a_layout, a_block_scaling_dim),
        (b_data, b_scale, b_layout, b_block_scaling_dim),
        out=out,
        out_dtype=out_dtype,
        alpha=alpha,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        allow_tf32=allow_tf32,
        workspace_size=workspace_size,
    )


@torch.library.custom_op(
    "torchtitan::fp8_dense_gemm",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _fp8_dense_gemm(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    a_layout: int,
    a_block_scaling_dim: int,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    b_layout: int,
    b_block_scaling_dim: int,
    out_dtype: torch.dtype | None = None,
    alpha: float = 1.0,
    accumulate: bool = False,
    use_split_accumulator: bool = True,
    allow_tf32: bool = True,
    workspace_size: int = 32 * 1024 * 1024,
) -> torch.Tensor:
    return _call_vendor_gemm(
        a_data,
        a_scale,
        a_layout,
        a_block_scaling_dim,
        b_data,
        b_scale,
        b_layout,
        b_block_scaling_dim,
        None,
        out_dtype,
        alpha,
        accumulate,
        use_split_accumulator,
        allow_tf32,
        workspace_size,
    )


@_fp8_dense_gemm.register_fake
def _fp8_dense_gemm_fake(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    a_layout: int,
    a_block_scaling_dim: int,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    b_layout: int,
    b_block_scaling_dim: int,
    out_dtype: torch.dtype | None = None,
    alpha: float = 1.0,
    accumulate: bool = False,
    use_split_accumulator: bool = True,
    allow_tf32: bool = True,
    workspace_size: int = 32 * 1024 * 1024,
) -> torch.Tensor:
    return _fp8_dense_gemm_impl(
        a_data,
        a_scale,
        a_layout,
        a_block_scaling_dim,
        b_data,
        b_scale,
        b_layout,
        b_block_scaling_dim,
        None,
        out_dtype,
        alpha,
        accumulate,
        use_split_accumulator,
        allow_tf32,
        workspace_size,
    )


def _fp8_dense_gemm_impl(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    a_layout: int,
    a_block_scaling_dim: int,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    b_layout: int,
    b_block_scaling_dim: int,
    out: torch.Tensor | None,
    out_dtype: torch.dtype | None = None,
    alpha: float = 1.0,
    accumulate: bool = False,
    use_split_accumulator: bool = True,
    allow_tf32: bool = True,
    workspace_size: int = 32 * 1024 * 1024,
) -> torch.Tensor:
    """Shared backend body, also used by scoped CPU test doubles."""
    return _call_vendor_gemm(
        a_data,
        a_scale,
        a_layout,
        a_block_scaling_dim,
        b_data,
        b_scale,
        b_layout,
        b_block_scaling_dim,
        out,
        out_dtype,
        alpha,
        accumulate,
        use_split_accumulator,
        allow_tf32,
        workspace_size,
    )


@torch.library.custom_op(
    "torchtitan::fp8_dense_gemm_mutate",
    mutates_args={"out"},
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _fp8_dense_gemm_mutate(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    a_layout: int,
    a_block_scaling_dim: int,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    b_layout: int,
    b_block_scaling_dim: int,
    out: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    alpha: float = 1.0,
    accumulate: bool = False,
    use_split_accumulator: bool = True,
    allow_tf32: bool = True,
    workspace_size: int = 32 * 1024 * 1024,
) -> None:
    _fp8_dense_gemm_mutate_impl(
        a_data,
        a_scale,
        a_layout,
        a_block_scaling_dim,
        b_data,
        b_scale,
        b_layout,
        b_block_scaling_dim,
        out,
        out_dtype,
        alpha,
        accumulate,
        use_split_accumulator,
        allow_tf32,
        workspace_size,
    )


@_fp8_dense_gemm_mutate.register_fake
def _fp8_dense_gemm_mutate_fake(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    a_layout: int,
    a_block_scaling_dim: int,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    b_layout: int,
    b_block_scaling_dim: int,
    out: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    alpha: float = 1.0,
    accumulate: bool = False,
    use_split_accumulator: bool = True,
    allow_tf32: bool = True,
    workspace_size: int = 32 * 1024 * 1024,
) -> None:
    _fp8_dense_gemm_mutate_impl(
        a_data,
        a_scale,
        a_layout,
        a_block_scaling_dim,
        b_data,
        b_scale,
        b_layout,
        b_block_scaling_dim,
        out,
        out_dtype,
        alpha,
        accumulate,
        use_split_accumulator,
        allow_tf32,
        workspace_size,
    )


def _fp8_dense_gemm_mutate_impl(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    a_layout: int,
    a_block_scaling_dim: int,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    b_layout: int,
    b_block_scaling_dim: int,
    out: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    alpha: float = 1.0,
    accumulate: bool = False,
    use_split_accumulator: bool = True,
    allow_tf32: bool = True,
    workspace_size: int = 32 * 1024 * 1024,
) -> None:
    _call_vendor_gemm(
        a_data,
        a_scale,
        a_layout,
        a_block_scaling_dim,
        b_data,
        b_scale,
        b_layout,
        b_block_scaling_dim,
        out,
        out_dtype,
        alpha,
        accumulate,
        use_split_accumulator,
        allow_tf32,
        workspace_size,
    )


def fp8_gemm(
    a: GemmOperand,
    b: GemmOperand,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    alpha: float = 1.0,
    accumulate: bool = False,
    use_split_accumulator: bool = True,
    allow_tf32: bool = True,
    workspace_size: int = 32 * 1024 * 1024,
) -> torch.Tensor:
    """Call CudaCherry GEMM through an opaque, CUDA-Graph-unsafe task op."""
    a_data, a_scale, a_layout, a_block = unpack_operand(a, "a")
    b_data, b_scale, b_layout, b_block = unpack_operand(b, "b")

    if out is None:
        return _fp8_dense_gemm(
            a_data,
            a_scale,
            a_layout,
            a_block,
            b_data,
            b_scale,
            b_layout,
            b_block,
            out_dtype=out_dtype,
            alpha=alpha,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
            allow_tf32=allow_tf32,
            workspace_size=workspace_size,
        )

    _fp8_dense_gemm_mutate(
        a_data,
        a_scale,
        a_layout,
        a_block,
        b_data,
        b_scale,
        b_layout,
        b_block,
        out=out,
        out_dtype=out_dtype,
        alpha=alpha,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        allow_tf32=allow_tf32,
        workspace_size=workspace_size,
    )
    return out


__all__ = ["fp8_gemm"]
