"""Model-agnostic training regularizers and magnitude control.

The grad-only auxiliary-loss splice (:class:`AddAuxiliaryLoss`), the hidden
z-loss on activation magnitudes, and tanh soft-capping of logits.
"""

from dataclasses import dataclass

from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.config.schema.models.common.regularization import (
    HiddenZLoss,
    set_hidden_z_loss,
)
from torchtitan.config.schema.protocols.types import MeshAxisName

__all__ = [
    "AddAuxiliaryLoss",
    "masked_mean",
    "hidden_z_loss",
    "HiddenZLoss",
    "attach_hidden_z_loss",
    "set_hidden_z_loss",
    "apply_softcap",
    "AuxLossState",
    "declare_aux_loss",
    "wire_aux_loss",
]

_KINDS_ATTR = "_aux_loss_kinds"
_SCALE_ATTR = "aux_loss_scale"
_TOTALS_ATTR = "aux_loss_totals"


@dataclass
class AuxLossState:
    """Splice weight and per-name totals for one model part.

    ``totals`` is one flat vector each splice views into, so a reset is one
    kernel and a reduce is one collective.
    """

    scale: torch.Tensor
    names: list[str]
    totals: torch.Tensor

    def reset(self) -> None:
        self.totals.zero_()


def declare_aux_loss(module: nn.Module, *kinds: str) -> None:
    """Register ``module`` as splicing aux/z terms of these ``kinds``."""
    kept = getattr(module, _KINDS_ATTR, ())
    setattr(module, _KINDS_ATTR, kept + tuple(k for k in kinds if k not in kept))
    if not hasattr(module, _TOTALS_ATTR):
        setattr(module, _SCALE_ATTR, None)
        setattr(module, _TOTALS_ATTR, {})


def wire_aux_loss(
    model: nn.Module,
    *,
    device: torch.device,
    pp_group: dist.ProcessGroup | None = None,
) -> AuxLossState:
    """Point every declared splice in ``model`` at a shared scale and its total.

    Unwired, splices weigh 1.0 and record nothing, so a model used outside the
    trainer is unaffected. ``pp_group`` unions the names across stages once, so
    the per-step reduce lines up -- a rank owns only its own stage's modules.
    """
    scale = torch.ones((), device=device)
    owned: dict[str, tuple[nn.Module, str]] = {}
    for fqn, module in model.named_modules():
        for kind in getattr(module, _KINDS_ATTR, ()):
            base, dot, suffix = kind.partition(".")
            owned[f"{base}/{fqn or '<root>'}{dot}{suffix}"] = (module, kind)

    names = set(owned)
    if pp_group is not None:
        per_stage: list = [None] * dist.get_world_size(pp_group)
        dist.all_gather_object(per_stage, sorted(names), group=pp_group)
        names = set().union(*per_stage)

    # Sorted: the reduce is positional, so every rank must order names alike.
    ordered = sorted(names)
    totals = torch.zeros(len(ordered), device=device)
    for index, name in enumerate(ordered):
        if name in owned:
            module, kind = owned[name]
            getattr(module, _TOTALS_ATTR)[kind] = totals[index]  # a view
            setattr(module, _SCALE_ATTR, scale)
    return AuxLossState(scale=scale, names=ordered, totals=totals)


def aux_loss_args(
    module: nn.Module, kind: str
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``(scale, total)`` for one splice, or ``(None, None)`` when unwired."""
    scale = getattr(module, _SCALE_ATTR, None)
    if scale is None:
        return None, None
    return scale, getattr(module, _TOTALS_ATTR)[kind]


# Opaque: dynamo rejects side effects escaping an autograd.Function. Mutating:
# a result-free op declaring none is dead code, and inductor drops it.
@torch.library.custom_op("torchtitan::record_aux_loss", mutates_args={"total"})
def record_aux_loss(total: torch.Tensor, value: torch.Tensor) -> None:
    total.add_(value)


@record_aux_loss.register_fake
def _(total: torch.Tensor, value: torch.Tensor) -> None:
    return None


def masked_mean(
    per_token: torch.Tensor, token_mask: torch.Tensor | None
) -> torch.Tensor:
    """Mean of ``(B, L)`` per-token values over the true positions of ``token_mask``."""
    if token_mask is None:
        return per_token.mean()
    mask = token_mask.to(per_token.dtype)
    # Rescaling the all-positions mean, rather than dividing by the real count,
    # keeps the unmasked reduction: an all-true mask scales by exactly 1.0.
    return (per_token * mask).mean() * (per_token.numel() / mask.sum().clamp(min=1.0))


# What the trainer's reduce covers: the ``loss`` mesh, dp_replicate * dp_shard * cp.
_REDUCED_AXES = frozenset(
    {MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD, MeshAxisName.CP}
)


def _local_contribution(value: DTensor) -> torch.Tensor:
    """This rank's share of a scalar the trainer sums over the dp/cp mesh.

    Reducing a sharded axis away leaves ``Partial``: on an axis the trainer
    reduces that is the local contribution, so keep it; on any other (tp) nothing
    downstream adds the pieces up, so finish it here with one scalar all-reduce.
    """
    axes = value.device_mesh.mesh_dim_names
    if axes is None:
        raise ValueError(f"aux/z-loss needs a named mesh, got {value.placements}.")
    target = tuple(
        Replicate() if (p.is_partial() and axis not in _REDUCED_AXES) else p
        for axis, p in zip(axes, value.placements)
    )
    if target != value.placements:
        value = value.redistribute(value.device_mesh, target)
    return value.to_local()


# While True, the splices record nothing. Pipeline metadata inference runs a
# real forward+backward with inputs materialized as ``torch.empty``; the aux
# values computed off that garbage are noise (NaN in the step-1 totals was
# the tell) and must not enter the step's accounting. Toggled only by
# ``suppress_aux_recording``; read eagerly in the splice backward, so it must
# not be flipped around a torch.compile'd region.
_SUPPRESS_RECORDING = False


@contextmanager
def suppress_aux_recording():
    """No splice records while active: for the pipeline metadata probe."""
    global _SUPPRESS_RECORDING
    previous = _SUPPRESS_RECORDING
    _SUPPRESS_RECORDING = True
    try:
        yield
    finally:
        _SUPPRESS_RECORDING = previous


class AddAuxiliaryLoss(torch.autograd.Function):
    """Route an auxiliary scalar into the backward graph without changing the forward value.

    ``forward`` returns ``x`` unchanged; ``backward`` sends a gradient of ones to ``loss``.
    The net effect on gradients is identical to ``total = main_loss + loss``, but the
    reported ``main_loss`` scalar stays clean (the aux is grad-only).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        loss: torch.Tensor,
        scale: torch.Tensor | None = None,
        total: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert loss.numel() == 1
        ctx.save_for_backward(loss)
        # On ctx: both are mutated in place, which trips the version counter that
        # save_for_backward keeps.
        ctx.scale, ctx.total = scale, total
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (loss,) = ctx.saved_tensors
        scale, total = ctx.scale, ctx.total
        # Weight 1.0 fits a per-rank mean; this loss is a share of a global-token
        # sum the DP reduce adds up, so unscaled it grows with dp * grad_accum.
        # Derived from `loss` to keep a DTensor's subclass and placements.
        weight = (
            torch.ones_like(loss) if scale is None else torch.ones_like(loss) * scale
        )
        if total is not None and not _SUPPRESS_RECORDING:
            # Here, not in forward: backward runs once per microbatch, forward
            # twice under AC. Weighted, so the sum is the term's real pull.
            value = loss.detach() * weight
            if isinstance(value, DTensor):
                # The op has no sharding rule, so hand it a plain tensor.
                value = _local_contribution(value)
            record_aux_loss(total, value.to(total.dtype))
        grad_loss = weight if ctx.needs_input_grad[1] else None
        return grad_output, grad_loss, None, None


def hidden_z_loss(h: torch.Tensor, *, coef: float) -> torch.Tensor:
    """z-loss on hidden-state magnitude: ``coef * logsumexp(|h|, dim=-1)**2``, mean over tokens.

    Penalizes large activations. Computed in float32 (``logsumexp`` + square can
    round/overflow in bf16 at large feature dims). Padding is not excluded --
    unlike the router losses, which mask it.
    """
    return (coef * torch.logsumexp(h.float().abs(), dim=-1).square()).mean()


def attach_hidden_z_loss(
    spec: HiddenZLoss, module: nn.Module
) -> torch.utils.hooks.RemovableHandle | None:
    """Register the hook that splices ``spec``'s z-loss into ``module``'s graph.

    The hook only rewrites the autograd graph -- no state is stashed, so it
    is safe under activation checkpointing (recompute reproduces it) and
    pipeline parallelism (nothing is shared between microbatches).
    """
    if spec.coef == 0.0:
        return None
    coef = spec.coef
    declare_aux_loss(module, "hidden_z_loss")

    if spec.on == "output":

        def _hook(mod, _args, output):
            if not mod.training:
                return output
            return AddAuxiliaryLoss.apply(
                output,
                hidden_z_loss(output, coef=coef),
                *aux_loss_args(mod, "hidden_z_loss"),
            )

        return module.register_forward_hook(_hook)

    def _pre_hook(mod, args):
        if not mod.training:
            return args
        x = args[0]
        spliced = AddAuxiliaryLoss.apply(
            x,
            hidden_z_loss(x, coef=coef),
            *aux_loss_args(mod, "hidden_z_loss"),
        )
        return (spliced,) + args[1:]

    return module.register_forward_pre_hook(_pre_hook)


def apply_softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Bound values to ``[-cap, cap]`` via ``cap * tanh(x / cap)``.

    Computed in float32 then cast back: ``tanh`` of a large low-precision
    quotient loses resolution near saturation. ``cap == 0.0`` disables the cap
    (exact identity, no dtype round-trip).
    """
    if cap == 0.0:
        return x
    return (cap * torch.tanh(x.float() / cap)).to(x.dtype)
