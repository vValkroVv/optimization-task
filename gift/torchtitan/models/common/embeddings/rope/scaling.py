"""Frequency scaling: base inverse frequencies -> (frequencies, mscale).

Each ``RoPE.Config`` subclass maps the strategy types it supports to an
implementation (``rope.py``), so that map is also its capability declaration.
The two YaRN formulas stay separate: complex rounds its correction range, cos/sin
does not, and only cos/sin applies ``mscale``.
"""

from __future__ import annotations

import math

import torch

from torchtitan.config.schema.models.common.embeddings.rope.scaling import (
    _cos_sin_yarn_window,
    LlamaScaling,
    NoScaling,
    ScalingConfig,
    validate_cos_sin_yarn_window,
    YarnScaling,
)

__all__ = [
    "LlamaScaling",
    "NoScaling",
    "ScalingConfig",
    "YarnScaling",
    "complex_llama_scaling",
    "complex_yarn_scaling",
    "cos_sin_yarn_scaling",
    "find_correction_dim",
    "find_correction_range",
    "linear_ramp_factor",
    "no_scaling",
    "validate_cos_sin_yarn_window",
    "yarn_mscale",
]


# --- YaRN primitives --------------------------------------------------------
# ``find_correction_range`` / ``linear_ramp_factor`` back the complex
# implementation; ``yarn_mscale`` backs the cos/sin one.


def find_correction_dim(
    num_rotations: float, dim: int, base: float, max_seq_len: int
) -> float:
    """The dim index whose wavelength completes ``num_rotations`` rotations over
    ``max_seq_len`` positions."""
    return (
        dim
        * math.log(max_seq_len / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_seq_len: int,
) -> tuple[int, int]:
    """Clamp the [low_rot, high_rot] correction window to valid dim indices."""
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_factor(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    """Linear ramp in ``[0, 1]`` over ``dim`` entries between min_val and max_val."""
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
        max_val - min_val
    )
    return torch.clamp(linear_func, 0, 1)


def yarn_mscale(rope_factor: float) -> float:
    """Magnitude-preserving scale for cos/sin under YaRN."""
    if rope_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(rope_factor) + 1.0


# --- strategy implementations ----------------------------------------------
# Uniform signature so the per-backend map can type its own values; each assert
# is what that map guarantees.


def no_scaling(
    freqs: torch.Tensor, scaling: ScalingConfig, *, dim: int, theta: float
) -> tuple[torch.Tensor, float]:
    """Identity."""
    del scaling, dim, theta
    return freqs, 1.0


def complex_llama_scaling(
    freqs: torch.Tensor, scaling: ScalingConfig, *, dim: int, theta: float
) -> tuple[torch.Tensor, float]:
    """Llama3/4 low/high/medium frequency scaling for the complex backend."""
    # Guaranteed by the ``_SCALINGS`` entry that routes here: the strategy
    # signature is uniform so the map can describe its own value type.
    assert isinstance(scaling, LlamaScaling)
    del dim, theta  # the window is set by wavelength, not by the head dim
    scaling_factor = scaling.scaling_factor
    low_freq_factor = scaling.low_freq_factor
    high_freq_factor = scaling.high_freq_factor
    original_max_position_embeddings = scaling.original_max_position_embeddings
    wavelen = 2 * math.pi / freqs
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    freqs = torch.where(wavelen > low_freq_wavelen, freqs / scaling_factor, freqs)
    smooth_factor = (original_max_position_embeddings / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_freqs = (
        1 - smooth_factor
    ) * freqs / scaling_factor + smooth_factor * freqs
    is_medium_freqs = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    freqs = torch.where(is_medium_freqs, smoothed_freqs, freqs)
    return freqs, 1.0


def complex_yarn_scaling(
    freqs: torch.Tensor, scaling: ScalingConfig, *, dim: int, theta: float
) -> tuple[torch.Tensor, float]:
    """YaRN (DeepSeek V3 style). ``mscale`` stays 1.0 -- MLA folds it into
    ``softmax_scale`` instead."""
    assert isinstance(scaling, YarnScaling)  # guaranteed by the _SCALINGS entry
    low, high = find_correction_range(
        scaling.beta_fast, scaling.beta_slow, dim, theta, scaling.original_seq_len
    )
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / scaling.rope_factor * (1 - smooth) + freqs * smooth
    return freqs, 1.0


def cos_sin_yarn_scaling(
    freqs: torch.Tensor, scaling: ScalingConfig, *, dim: int, theta: float
) -> tuple[torch.Tensor, float]:
    """YaRN for cos/sin; returns ``(inv_freq, mscale)``.

    NTK-by-parts needs the non-inverse ``freq``, recomputed here. The cache
    builder bakes ``mscale`` into cos and sin, landing ``mscale**2`` on the logits.
    """
    assert isinstance(scaling, YarnScaling)  # guaranteed by the _SCALINGS entry
    del freqs  # recomputed below from dim/theta, as the reference does
    rope_factor = scaling.rope_factor
    mscale = yarn_mscale(rope_factor)

    freq = theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    d_half = dim / 2
    low, high = _cos_sin_yarn_window(scaling, dim=dim, theta=theta)

    ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
    mask = 1 - ramp.clamp(0, 1)

    interpolation = 1.0 / (rope_factor * freq)
    extrapolation = 1.0 / freq
    return interpolation * (1 - mask) + extrapolation * mask, mscale
