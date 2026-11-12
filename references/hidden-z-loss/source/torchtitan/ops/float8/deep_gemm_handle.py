# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings

import torch

_deep_gemm = None


def _load_deep_gemm():
    global _deep_gemm
    if _deep_gemm is not None:
        return _deep_gemm
    try:
        import deep_gemm
    except ImportError as exc:  # pragma: no cover - depends on the image
        raise ImportError(
            "Blockwise fp8 grouped GEMM requires the deep_gemm package, which "
            "is not installed. The DeepGEMM kernels ship in the training image; "
            "rebuild or pull the current image.\n"
            f"Original error: {exc}"
        ) from exc
    major, minor, *_ = deep_gemm.__version__.split(".")
    if (int(major), int(minor)) != (2, 3):
        warnings.warn(
            "for optimal performance, deep_gemm version must be 2.3.*, got "
            f"{deep_gemm.__version__}"
        )
    _deep_gemm = deep_gemm
    return deep_gemm


@torch.library.custom_op(
    "torchtitan::k_grouped_fp8_gemm_contiguous", mutates_args={"d"}, device_types="cuda"
)
def _k_grouped_fp8_gemm_contiguous(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    d: torch.Tensor,
    ks: list[int] | None,
    ks_cpu: torch.Tensor | None,
    ks_tensor: torch.Tensor,
) -> None:
    """DeepGEMM's k-grouped contiguous fp8 GEMM (wgrad), accumulating into ``d``."""
    assert (
        a_data.is_cuda
        and a_scale.is_cuda
        and b_data.is_cuda
        and b_scale.is_cuda
        and d.is_cuda
        and ks_tensor.is_cuda
    )
    assert a_data.dtype == b_data.dtype == torch.float8_e4m3fn
    assert a_scale.dtype == b_scale.dtype == torch.float32
    assert d.dtype in [torch.float32, torch.bfloat16]
    assert a_data.ndim == 1 and a_scale.ndim == 2
    assert b_data.ndim == 1 and b_scale.ndim == 2

    if ks is None:
        # ``ks_cpu`` is the same counts already on the host -- blocking dispatch
        # left them in pinned memory -- so reading it costs nothing. Falling
        # back to the device tensor costs a full stream synchronization, which
        # is why the dispatch hands the host copy out at all.
        ks = (ks_cpu if ks_cpu is not None and ks_cpu.numel() else ks_tensor).tolist()

    num_groups, m, n = d.shape
    assert len(ks) == ks_tensor.shape[0] == num_groups
    if any(k % 128 for k in ks):
        raise ValueError(f"group sizes must be multiples of 128, got {ks}.")

    # DeepGEMM asserts ``a.numel() == sum(ks) * m``. A fixed-capacity layout --
    # HybridEP's non-blocking dispatch -- is sized for the worst case and its
    # tail carries no expert, so trim to the rows the groups cover. Doing it
    # here rather than in the caller is what keeps non-blocking compilable: the
    # bounds come from unbacked SymInts, and inductor emits a slice expression
    # over symbols that are not in scope in the backward it generates.
    rows = sum(ks)
    if rows * m > a_data.numel():
        raise ValueError(
            f"group sizes cover {rows} rows, more than the {a_data.numel() // m} "
            "the operand holds."
        )
    if rows * m != a_data.numel():
        blocks = rows // 128
        a_data, b_data = a_data[: rows * m], b_data[: rows * n]
        a_scale, b_scale = a_scale[:, :blocks], b_scale[:, :blocks]

    assert a_data.is_contiguous() and b_data.is_contiguous()

    # An empty batch means every k is zero, so the weight gradient is the zero
    # matrix -- which is what ``d`` already holds, because the caller allocates
    # it with torch.zeros.
    if a_data.numel() == 0:
        return

    _load_deep_gemm().k_grouped_fp8_gemm_nt_contiguous(
        (a_data, a_scale), (b_data, b_scale), d, ks, ks_tensor, d
    )


@_k_grouped_fp8_gemm_contiguous.register_fake
def _(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    d: torch.Tensor,
    ks: list[int] | None,
    ks_cpu: torch.Tensor | None,
    ks_tensor: torch.Tensor,
) -> None:
    """Fake k-grouped GEMM: writes ``d`` in place and returns nothing."""
    return None


@torch.library.custom_op(
    "torchtitan::m_grouped_fp8_gemm_nt_contiguous",
    mutates_args={"d"},
    device_types="cuda",
)
def _m_grouped_fp8_gemm_nt_contiguous(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
) -> None:
    """DeepGEMM's m-grouped contiguous fp8 GEMM, writing into ``d``."""
    assert (
        a_data.is_cuda
        and a_scale.is_cuda
        and b_data.is_cuda
        and b_scale.is_cuda
        and d.is_cuda
        and m_indices.is_cuda
    )
    assert a_data.dtype == b_data.dtype == torch.float8_e4m3fn, (
        f"{a_data.dtype=} {b_data.dtype=}"
    )
    assert a_scale.dtype == b_scale.dtype == torch.float32
    assert d.dtype == torch.bfloat16
    assert a_data.is_contiguous() and b_data.is_contiguous() and d.is_contiguous()
    assert a_data.shape[0] == m_indices.numel(), (
        f"{a_data.shape[0]=} {m_indices.numel()=}"
    )

    seq_len_a0, hid_dim_a0 = a_data.shape
    seq_len_a1, hid_dim_a1 = a_scale.shape
    assert seq_len_a0 == seq_len_a1
    assert hid_dim_a0 // 128 == hid_dim_a1

    num_groups, int_dim, _ = b_data.shape
    assert int_dim == d.shape[1]

    # Empty batch: ``d`` has no rows either, so there is nothing to write.
    if seq_len_a0 == 0:
        return

    _load_deep_gemm().m_grouped_fp8_gemm_nt_contiguous(
        (a_data, a_scale), (b_data, b_scale), d, m_indices
    )


@_m_grouped_fp8_gemm_nt_contiguous.register_fake
def _(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
) -> None:
    """Fake m-grouped GEMM: writes ``d`` in place and returns nothing."""
    return None


class _DeepGemmHandle:
    """Checks the GPU DeepGEMM needs, sets its SM budget, and calls its ops."""

    def __init__(self, num_sms: int | None = None):
        deep_gemm = _load_deep_gemm()
        p = torch.cuda.get_device_properties(0)
        assert (p.major, p.minor) == (9, 0)
        if num_sms is not None:
            deep_gemm.set_num_sms(int(num_sms))

    def k_grouped_fp8_gemm_contiguous(
        self,
        a: tuple[torch.Tensor, torch.Tensor],
        b: tuple[torch.Tensor, torch.Tensor],
        d: torch.Tensor,
        ks: list[int] | None,
        ks_cpu: torch.Tensor | None,
        ks_tensor: torch.Tensor,
    ) -> None:
        torch.ops.torchtitan.k_grouped_fp8_gemm_contiguous(
            a[0], a[1], b[0], b[1], d, ks, ks_cpu, ks_tensor
        )

    def m_grouped_fp8_gemm_nt_contiguous(
        self,
        a: tuple[torch.Tensor, torch.Tensor],
        b: tuple[torch.Tensor, torch.Tensor],
        d: torch.Tensor,
        m_indices: torch.Tensor,
    ) -> None:
        torch.ops.torchtitan.m_grouped_fp8_gemm_nt_contiguous(
            a[0], a[1], b[0], b[1], d, m_indices
        )
