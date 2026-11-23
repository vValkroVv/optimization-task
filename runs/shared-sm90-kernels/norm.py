"""Replay native CuTe gated norm fixtures with the production SAC compiler."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


_NATIVE_SHAPE = (2, 4096, 4096)
_NORM_OP = "torchtitan::cute_gated_norm_fwd"
_RESIDUAL_OP = "torchtitan::cute_gated_norm_residual_fwd"
_STATS_OP = "torchtitan::cute_gated_norm_fwd_stats"
_BACKWARD_OP = "torchtitan::cute_gated_norm_bwd"
_PREP_KERNELS = ("GnPrep", "GnFwdStats")
_FULL_KERNEL = "GnFwdOut"
_STATS_KERNEL = "GnFwdFinalizeStats"
_BACKWARD_KERNELS = ("GnBwdDs", "GnBwdFin", "GnBwdDx", "GnBwdWFin")
_KERNELS = (*_PREP_KERNELS, _FULL_KERNEL, _STATS_KERNEL, *_BACKWARD_KERNELS)


def _selectors(case: str, arm: str) -> tuple[str, bool, bool]:
    if case not in {"norm-residual", "sac-plain", "sac-residual"} or arm not in {
        "A",
        "B",
    }:
        raise ValueError(f"Invalid norm case or arm: {case}/{arm}")
    role = "plain" if case == "sac-plain" else "residual"
    residual_fused = case == "sac-residual" or (case == "norm-residual" and arm == "B")
    stats_only = case != "norm-residual" and arm == "B"
    return role, residual_fused, stats_only


def _to_cuda(tensor: torch.Tensor, layout: Mapping) -> torch.Tensor:
    # The historical capture keeps the values and strides but drops unused
    # backing storage. A short prefix also restores the original alignment.
    if (tuple(tensor.shape), tensor.stride(), str(tensor.dtype)) != (
        tuple(layout["shape"]),
        tuple(layout["stride"]),
        layout["dtype"],
    ):
        raise ValueError(f"CuTe fixture tensor does not match its layout: {layout}")
    span = 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    prefix = layout["address_mod_256"] // tensor.element_size()
    buffer = torch.empty(span + prefix, dtype=tensor.dtype, device="cuda")
    restored = buffer.as_strided(tensor.shape, tensor.stride(), storage_offset=prefix)
    if restored.data_ptr() % 256 != layout["address_mod_256"]:
        raise RuntimeError(f"Could not restore CuTe fixture alignment: {layout}")
    return restored.copy_(tensor)


def _compile_block(
    eps: float, gating_weight: float, residual_fused: bool, stats_only: bool
):
    from torchtitan.config import ActivationCheckpointConfig, CompileConfig
    from torchtitan.distributed.activation_checkpoint import _apply_op_sac
    from torchtitan.distributed.compile import apply_compile
    from torchtitan.models.common.norms.cute_gated_norm import gated_norm

    class Norm(nn.Module):
        def forward(self, x, weight, up, down, residual=None):
            y = gated_norm(
                x,
                weight,
                up,
                down,
                eps=eps,
                gating_weight=gating_weight,
                residual=residual if residual_fused else None,
            )
            if residual is not None and not residual_fused:
                return residual + y
            return y

    # The production crop uses selective_save=[] for layers.0. The capture is
    # at the custom-op boundary, where any weight transformation has run.
    block = _apply_op_sac(
        Norm(),
        ActivationCheckpointConfig(
            mode="selective", selective_save=[], early_stop=False
        ),
    )
    segment = SimpleNamespace(
        is_block=True, fqn="layers.0", module=block, group="layers"
    )
    model = SimpleNamespace(named_segments=lambda: [segment])
    with patch.dict(
        os.environ, {"GIFT_CUTE_NORM_SAC_STATS_ONLY": str(int(stats_only))}
    ):
        apply_compile(
            model, CompileConfig(enable=True, cudagraph_exclude_blocks=[segment.fqn])
        )
    return block


def build(
    case: str, arm: str, fixture_path: Path
) -> tuple[Callable[[], torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    role, residual_fused, stats_only = _selectors(case, arm)
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True, mmap=True)
    if fixture["torch_revision"] != torch.version.git_version:
        raise RuntimeError("CuTe capture and replay use different PyTorch revisions")
    record = fixture["records"][role]
    cpu = record["tensors"]
    layouts = record["layouts"]
    names = ("x", "weight", "up", "down")
    if role == "residual":
        names = (*names, "residual")

    # The native residual capture did not find the same call's backward. Its
    # recorded fallback is shared between A and B, as in cute_component.py.
    if "dy" in cpu:
        dy, dy_layout = cpu["dy"], layouts["dy"]
    else:
        dy, dy_layout = (
            fixture["first_tangent"]["dy"],
            fixture["first_tangent"]["layout"],
        )
    if cpu["x"].shape != _NATIVE_SHAPE or dy.shape != _NATIVE_SHAPE:
        raise ValueError("Expected the captured B2/S4096/H4096 norm and tangent")
    if cpu["x"].dtype != torch.bfloat16 or dy.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 norm input and tangent")

    leaves = {
        name: _to_cuda(cpu[name], layouts[name]).requires_grad_() for name in names
    }
    tangent = _to_cuda(dy, dy_layout)
    block = _compile_block(
        record["eps"], record["gating_weight"], residual_fused, stats_only
    )
    args = tuple(leaves.values())

    def call_forward():
        return block(*args)

    return call_forward, tangent, leaves


def _check_phases(case, arm, forward, backward, expected_forward, expected_backward):
    counts = {"forward": dict(forward), "backward": dict(backward)}
    expected = {"forward": dict(expected_forward), "backward": dict(expected_backward)}
    if counts != expected:
        raise RuntimeError(f"{case}/{arm} profiler route {counts}, expected {expected}")
    return counts


def verify_kernels(
    case: str,
    arm: str,
    forward_counts: Mapping[str, int],
    backward_counts: Mapping[str, int],
) -> dict:
    """Check one forward and one backward for the kernels observed in the native replay."""
    _, _, stats_only = _selectors(case, arm)

    def classify(counts):
        matched = Counter()
        other = {}
        for name, count in counts.items():
            kernel = next((kernel for kernel in _KERNELS if f"{kernel}_" in name), None)
            if kernel is None:
                other[name] = count
            else:
                matched[kernel] += count
        return matched, other

    forward, other_forward = classify(forward_counts)
    backward, other_backward = classify(backward_counts)
    replay_kernel = _STATS_KERNEL if stats_only else _FULL_KERNEL
    verified = _check_phases(
        case,
        arm,
        forward,
        backward,
        Counter((*_PREP_KERNELS, _FULL_KERNEL)),
        Counter((*_PREP_KERNELS, replay_kernel, *_BACKWARD_KERNELS)),
    )
    # The residual fusion uses the same CuTe kernel name with a specialized
    # epilogue. Its unfused baseline has exactly one extra pointwise add.
    expected_extra = int(case == "norm-residual" and arm == "A")
    if sum(other_forward.values()) != expected_extra or other_backward:
        raise RuntimeError(
            f"{case}/{arm} unexpected kernels outside CuTe: "
            f"forward={other_forward}, backward={other_backward}"
        )
    return {**verified, "other_forward": other_forward}


def verify_ops(
    case: str, arm: str, forward_ops: Mapping[str, int], backward_ops: Mapping[str, int]
) -> dict:
    """Distinguish fused and unfused paths when CuTe kernel names are identical."""
    _, residual_fused, stats_only = _selectors(case, arm)
    ordinary = _RESIDUAL_OP if residual_fused else _NORM_OP
    replay = _STATS_OP if stats_only else ordinary

    def norm_ops(counts):
        return {
            name: count
            for name, count in counts.items()
            if name.startswith("torchtitan::cute_gated_norm_")
        }

    return _check_phases(
        case,
        arm,
        norm_ops(forward_ops),
        norm_ops(backward_ops),
        {ordinary: 1},
        {replay: 1, _BACKWARD_OP: 1},
    )
