# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import warnings

import torch

from torchtitan.tools.logging import logger

_deep_gemm = None
_routed_dw_overwrite_activated = False
_routed_dw_overwrite_flag = os.environ.get("GIFT_ROUTED_DW_BF16_OVERWRITE", "")
if _routed_dw_overwrite_flag not in ("", "0", "1"):
    raise ValueError("GIFT_ROUTED_DW_BF16_OVERWRITE must be 0 or 1")
_routed_dw_overwrite_enabled = _routed_dw_overwrite_flag == "1"
_ROUTED_DW_OVERWRITE_SHAPES = (
    (16, 3072, 4096),
    (16, 4096, 1536),
    (32, 3072, 4096),
    (32, 4096, 1536),
)


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


def _k_grouped_fp8_gemm(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    d: torch.Tensor,
    ks: list[int] | None,
    ks_cpu: torch.Tensor | None,
    ks_tensor: torch.Tensor,
    *,
    overwrite: bool,
) -> None:
    global _routed_dw_overwrite_activated
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

    # An empty batch has no GEMM. The legacy caller prezeros d, while the
    # overwrite caller needs the same positive zero gradient written here.
    if a_data.numel() == 0:
        if overwrite:
            d.zero_()
        return

    deep_gemm = _load_deep_gemm()
    if overwrite:
        device = d.device
        # Guard the narrower extension ABI after trimming, where the runtime
        # counts and addresses are available even under torch.compile.
        if (
            d.shape in _ROUTED_DW_OVERWRITE_SHAPES
            and d.dtype == torch.bfloat16
            and d.is_contiguous()
            and all(
                t.device == device
                for t in (a_data, a_scale, b_data, b_scale, ks_tensor)
            )
            and a_scale.shape == (m, rows // 128)
            and b_scale.shape == (n, rows // 128)
            and b_data.numel() == rows * n
            and ks_tensor.shape == (num_groups,)
            and ks_tensor.dtype == torch.int32
            and ks_tensor.is_contiguous()
            and rows <= (1 << 31) - 1
            and all(k >= 0 for k in ks)
            and all(t.data_ptr() % 16 == 0 for t in (a_data, b_data, d))
        ):
            from torchtitan.ops.float8._routed_dw_overwrite import load_extension

            extension = load_extension(device, deep_gemm)
            if extension is not None:
                extension.k_grouped_fp8_gemm_nt_contiguous_bf16_overwrite(
                    (a_data, a_scale), (b_data, b_scale), d, ks, ks_tensor
                )
                if not _routed_dw_overwrite_activated:
                    _routed_dw_overwrite_activated = True
                    logger.info("Routed BF16 dW overwrite activated")
                return

        d.zero_()
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (a_data, a_scale), (b_data, b_scale), d, ks, ks_tensor, d
    )


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
    _k_grouped_fp8_gemm(
        a_data, a_scale, b_data, b_scale, d, ks, ks_cpu, ks_tensor, overwrite=False
    )


@torch.library.custom_op(
    "torchtitan::k_grouped_fp8_gemm_bf16_overwrite",
    mutates_args={"d"},
    device_types="cuda",
)
def _k_grouped_fp8_gemm_bf16_overwrite(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    d: torch.Tensor,
    ks: list[int] | None,
    ks_cpu: torch.Tensor | None,
    ks_tensor: torch.Tensor,
) -> None:
    """Initialize every element of an uninitialized routed weight gradient."""
    _k_grouped_fp8_gemm(
        a_data, a_scale, b_data, b_scale, d, ks, ks_cpu, ks_tensor, overwrite=True
    )


@_k_grouped_fp8_gemm_contiguous.register_fake
@_k_grouped_fp8_gemm_bf16_overwrite.register_fake
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


def routed_bf16_wgrad(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    weight_shape: torch.Size,
    device: torch.device,
    ks: list[int] | None,
    ks_cpu: torch.Tensor | None,
    ks_tensor: torch.Tensor,
) -> torch.Tensor:
    if (
        _routed_dw_overwrite_enabled
        and device.type == "cuda"
        and weight_shape in _ROUTED_DW_OVERWRITE_SHAPES
    ):
        d = torch.empty(weight_shape, dtype=torch.bfloat16, device=device)
        op = torch.ops.torchtitan.k_grouped_fp8_gemm_bf16_overwrite
    else:
        d = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
        op = torch.ops.torchtitan.k_grouped_fp8_gemm_contiguous
    op(a[0], a[1], b[0], b[1], d, ks, ks_cpu, ks_tensor)
    return d


@torch.library.custom_op(
    "torchtitan::m_grouped_fp8_gemm_nt_contiguous",
    mutates_args=(),
    device_types="cuda",
)
def _m_grouped_fp8_gemm_nt_contiguous(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
) -> torch.Tensor:
    """Return a fresh bf16 result so selective AC can cache the GEMM output."""
    assert (
        a_data.is_cuda
        and a_scale.is_cuda
        and b_data.is_cuda
        and b_scale.is_cuda
        and m_indices.is_cuda
    )
    assert a_data.dtype == b_data.dtype == torch.float8_e4m3fn, (
        f"{a_data.dtype=} {b_data.dtype=}"
    )
    assert a_scale.dtype == b_scale.dtype == torch.float32
    assert a_data.is_contiguous() and b_data.is_contiguous()
    assert a_data.shape[0] == m_indices.numel(), (
        f"{a_data.shape[0]=} {m_indices.numel()=}"
    )

    seq_len_a0, hid_dim_a0 = a_data.shape
    seq_len_a1, hid_dim_a1 = a_scale.shape
    assert seq_len_a0 == seq_len_a1
    assert hid_dim_a0 // 128 == hid_dim_a1

    out = torch.empty(
        (seq_len_a0, b_data.shape[1]), dtype=torch.bfloat16, device=a_data.device
    )
    if seq_len_a0:
        _load_deep_gemm().m_grouped_fp8_gemm_nt_contiguous(
            (a_data, a_scale), (b_data, b_scale), out, m_indices
        )
    return out


@_m_grouped_fp8_gemm_nt_contiguous.register_fake
def _(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (a_data.shape[0], b_data.shape[1]), dtype=torch.bfloat16, device=a_data.device
    )


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
        m_indices: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.torchtitan.m_grouped_fp8_gemm_nt_contiguous(
            a[0], a[1], b[0], b[1], m_indices
        )
