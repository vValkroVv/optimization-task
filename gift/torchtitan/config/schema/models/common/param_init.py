"""Parameter-init specs shared by the model families."""

import dataclasses
from dataclasses import dataclass, replace

from torchtitan.config.configurable import component
from torchtitan.config.schema.protocols.module import InitSpec

__all__ = [
    "DepthScaled",
    "depth_scaled_std",
    "DimScaled",
    "FusedW13Init",
    "materialize_init",
    "NormUnitScale",
]


def depth_scaled_std(base_std: float, layer_id: int) -> float:
    """Compute depth-dependent std: base_std / sqrt(2 * (layer_id + 1))."""
    return base_std / (2 * (layer_id + 1)) ** 0.5


@component("depth_scaled")
@dataclass(frozen=True, slots=True)
class DepthScaled(InitSpec):
    """``base`` with its ``std`` scaled by depth (:func:`depth_scaled_std`).

    Resolved per block by :func:`materialize_init`; the runtime never sees it.
    """

    base: InitSpec


@component("dim_scaled")
@dataclass(frozen=True, slots=True)
class DimScaled(InitSpec):
    """``base`` with ``std = dim ** -0.5`` (model width) and truncation bounds at
    three stds: the LM head's init. Overwrites the base's ``std``."""

    base: InitSpec


@component("norm_unit_scale")
@dataclass(frozen=True, slots=True)
class NormUnitScale(InitSpec):
    """Norm weight giving unit scale: zeros if the norm is ``zero_centered``
    (scale ``1 + weight``), else ones. ``apply_init`` resolves it."""


def materialize_init(spec: InitSpec, *, layer_id: int, dim: int) -> InitSpec:
    """*spec* with its depth- and width-scaled inits resolved for a block at
    *layer_id* in a model of width *dim*, nested specs included."""
    if isinstance(spec, (DepthScaled, DimScaled)):
        base = materialize_init(spec.base, layer_id=layer_id, dim=dim)
        if not hasattr(base, "std"):
            raise ValueError(
                f"{type(spec).__name__} needs a base init with a std, got "
                f"{type(base).__name__}"
            )
        if isinstance(spec, DepthScaled):
            return replace(base, std=depth_scaled_std(base.std, layer_id))
        std = dim**-0.5
        bounds = dict(a=-3 * std, b=3 * std) if hasattr(base, "a") else {}
        return replace(base, std=std, **bounds)
    nested = {
        f.name: materialize_init(value, layer_id=layer_id, dim=dim)
        for f in dataclasses.fields(spec)
        if isinstance(value := getattr(spec, f.name), InitSpec)
    }
    return replace(spec, **nested) if nested else spec


@component("fused_w13")
@dataclass(frozen=True, slots=True)
class FusedW13Init(InitSpec):
    """Init of the fused ``(E, 2, F, D)`` expert weight, one spec per half.

    Slot 0 of dim 1 is gate (w1), slot 1 is up (w3).
    """

    gate: InitSpec
    up: InitSpec
