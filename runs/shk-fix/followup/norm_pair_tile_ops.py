"""CUDA and fake implementations for an isolated post-AOT norm pair."""

from functools import cache
from typing import List

import torch

from torchtitan.models.common.norms.cute_gated_norm.op import (
    _aligned,
    _check_device,
    _forward_fake,
    _forward_residual_fake,
    _forward_stats_fake,
    _validate_inputs,
    _validate_residual,
)


_HIDDEN, _RANK, _ROWS, _SMS = 4096, 16, 8192, 116
_CHUNKS, _SPLITS = 4, 8


def _check_shape(x):
    if x.ndim == 0 or x.shape[-1] != _HIDDEN or x.numel() != _ROWS * _HIDDEN:
        raise ValueError("Norm pair requires 8,192 rows and hidden size 4,096")


def _prepare(prep, weight, up, down):
    device = weight.device
    w_eff = torch.empty(_RANK, _HIDDEN, device=device, dtype=torch.bfloat16)
    prepared = (
        torch.empty(_HIDDEN, device=device, dtype=torch.float32),
        torch.empty(_HIDDEN, _RANK, device=device, dtype=torch.bfloat16),
        torch.empty(_RANK, _HIDDEN, device=device, dtype=torch.bfloat16),
        torch.empty(_HIDDEN, _RANK, device=device, dtype=torch.bfloat16),
    )
    prep(_aligned(up), _aligned(weight), _aligned(down), w_eff, *prepared)
    return w_eff, prepared


def make_fused_pair(candidate):
    compile_pair = cache(candidate.compile_norm_pair)

    @torch.library.custom_op(
        "torchtitan::exp_cute_norm_pair", mutates_args=(), device_types="cuda"
    )
    def fused_pair(
        x: torch.Tensor,
        weight: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        eps: float,
        lgw: float,
        residual: torch.Tensor,
        next_weight: torch.Tensor,
        next_up: torch.Tensor,
        next_down: torch.Tensor,
        next_eps: float,
        next_lgw: float,
        stats_only: bool,
    ) -> List[torch.Tensor]:
        from torchtitan.models.common.norms.cute_gated_norm import forward

        _check_shape(x)
        _validate_inputs(x, weight, up, down, eps)
        _validate_residual(x, residual)
        _validate_inputs(x, next_weight, next_up, next_down, next_eps)
        _check_device(x)
        device = x.device
        if forward.resolve_num_sms(device) != _SMS:
            raise ValueError("Norm pair requires the native 116-SM budget")

        with torch.cuda.device(device):
            prep, f1, _, splits = forward._fwd_plan(
                _HIDDEN,
                _RANK,
                (up.dtype, weight.dtype, down.dtype),
                eps,
                lgw,
                device,
                _CHUNKS,
                _SMS,
                add_residual=True,
            )
            next_prep, _, next_f2, next_splits = forward._fwd_plan(
                _HIDDEN,
                _RANK,
                (next_up.dtype, next_weight.dtype, next_down.dtype),
                next_eps,
                0.0 if stats_only else next_lgw,
                device,
                None if stats_only else _CHUNKS,
                _SMS,
            )
            if splits != _SPLITS or next_splits != _SPLITS:
                raise RuntimeError(
                    "Norm pair requires the native eight-way statistics split"
                )
            fused_f2 = compile_pair(_HIDDEN, _RANK, eps, lgw, device, _SMS)

            shape = x.shape
            x = _aligned(x).view(_ROWS, _HIDDEN)
            residual = _aligned(residual).view(_ROWS, _HIDDEN)
            w_eff, prepared = _prepare(prep, weight, up, down)
            p_part, ss_part, next_p_part, next_ss_part = forward.workspace(
                device, [(_SPLITS, _ROWS, _RANK), (_SPLITS, _ROWS)] * 2
            )
            f1(x, w_eff, p_part, ss_part)
            next_w_eff, next_prepared = _prepare(
                next_prep, next_weight, next_up, next_down
            )
            y = torch.empty_like(x)
            rho = torch.empty(_ROWS, device=device, dtype=torch.float32)
            p = torch.empty(_ROWS, _RANK, device=device, dtype=torch.float32)
            fused_f2(
                x,
                prepared[0],
                prepared[1],
                p_part,
                ss_part,
                y,
                rho,
                p,
                residual,
                next_w_eff,
                next_p_part,
                next_ss_part,
            )

            next_rho = torch.empty_like(rho)
            next_p = torch.empty_like(p)
            outputs = [
                y.view(shape),
                rho.view(shape[:-1]),
                p.view(*shape[:-1], _RANK),
                *prepared,
            ]
            if stats_only:
                next_f2(next_p_part, next_ss_part, next_rho, next_p)
            else:
                next_y = torch.empty_like(y)
                next_f2(
                    y,
                    next_prepared[0],
                    next_prepared[1],
                    next_p_part,
                    next_ss_part,
                    next_y,
                    next_rho,
                    next_p,
                    y,
                )
                outputs.append(next_y.view(shape))
            outputs.extend(
                (
                    next_rho.view(shape[:-1]),
                    next_p.view(*shape[:-1], _RANK),
                    *next_prepared,
                )
            )
            return outputs

    @fused_pair.register_fake
    def fake(
        x,
        weight,
        up,
        down,
        eps,
        lgw,
        residual,
        next_weight,
        next_up,
        next_down,
        next_eps,
        next_lgw,
        stats_only,
    ):
        _check_shape(x)
        first = _forward_residual_fake(x, weight, up, down, eps, lgw, residual)
        if stats_only:
            second = _forward_stats_fake(
                first[0], next_weight, next_up, next_down, next_eps
            )
        else:
            second = _forward_fake(
                first[0], next_weight, next_up, next_down, next_eps, next_lgw
            )
        return [*first, *second]

    return fused_pair
