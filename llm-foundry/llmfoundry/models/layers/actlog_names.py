# Copyright 2024 LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Naming for the hand-rolled activation-log metrics.

`ActivationMonitor`'s forward hooks name a metric after the module they are
attached to, so those names are always right.  The metrics the model code emits
itself, through `global_actlog_state.use_monitor_variable`, are named from a
template plus `global_actlog_state.layer_idx` -- a single counter advanced once
per iteration of the main decoder loop.

That counter is only meaningful *inside* the loop.  Modules that run after it
(the MTP heads, built with ``layer_idx=-1``) would inherit the last layer's
index, so their metrics silently overwrite that layer's.  The helpers here give
every such module an explicit name resolved at construction time instead.

Two flavours:

* :func:`attach_actlog_name` -- for a module that reports one thing, named after
  itself (an MLP, a group of experts).  Sets ``_actlog_name``.
* :func:`attach_actlog_prefix` -- for a module that reports several logical
  quantities that are not submodules (a decoder block logging its residual
  stream, a router logging its scores).  Sets ``_actlog_prefix``, and callers
  build ``self._actlog_prefix + "<logical name>"``.

Both also set ``_actlog_layer_depends``, which must be passed to
`use_monitor_variable` so a resolved MTP name is not run through `.format()`.
"""

from torch import nn

__all__ = [
    "MTP_ACTLOG_PREFIX",
    "attach_actlog_name",
    "attach_actlog_prefix",
    "resolve_actlog_name",
    "stamp_mtp_predictor_index",
]

# Activation-log prefix for modules built outside the decoder loop (MTP heads).
# `stamp_mtp_predictor_index` inserts the predictor index right after it.
MTP_ACTLOG_PREFIX = "model.model.mtp_block."


def resolve_actlog_name(base: str, layer_idx: int | None) -> tuple[str, bool]:
    """Resolve an activation-log name template.

    Returns ``(name_template, layer_depends)`` for
    :meth:`GlobalActivationLoggingState.use_monitor_variable`.

    ``layer_idx == -1`` marks a module that belongs to an MTP head rather than
    to a decoder layer; it gets a layer-independent name.  That name carries no
    predictor index yet -- the module does not know which head it belongs to --
    so ``LlamaDecoderMTPBlock`` stamps one in via
    :func:`stamp_mtp_predictor_index` once the heads are built.

    An empty ``base`` yields the bare layer prefix, dot included.
    """
    if layer_idx == -1:
        return f"{MTP_ACTLOG_PREFIX}{base}", False
    return "model.model.layers.{}." + base, True


def attach_actlog_name(module: nn.Module, base: str, layer_idx: int | None) -> None:
    """Store the resolved name of a module that reports one metric family."""
    module._actlog_name, module._actlog_layer_depends = resolve_actlog_name(
        base, layer_idx
    )


def attach_actlog_prefix(module: nn.Module, layer_idx: int | None) -> None:
    """Store the layer prefix for a module that reports several logical metrics."""
    module._actlog_prefix, module._actlog_layer_depends = resolve_actlog_name(
        "", layer_idx
    )


def stamp_mtp_predictor_index(mtp_layer: nn.Module, predictor_idx: int) -> None:
    """Insert the predictor index into the activation-log names of one MTP head.

    Every module inside an MTP layer is built with ``layer_idx=-1``, so they all
    resolve to ``model.model.mtp_block.<...>``.  With ``mtp_predictor_num > 1``
    the heads would share metric keys and overwrite each other (last head wins),
    so re-key them to ``model.model.mtp_block.<i>.<...>`` -- the layout
    ``modelling_giga_mix`` already uses for the per-head hidden states.

    Applied unconditionally (also for a single head) so a series is not silently
    renamed when ``mtp_predictor_num`` changes between runs.

    Call after the head is fully built: the names being re-keyed are assigned in
    the submodules' constructors.
    """
    for module in mtp_layer.modules():
        for attr in ("_actlog_name", "_actlog_prefix"):
            name = getattr(module, attr, None)
            if isinstance(name, str) and name.startswith(MTP_ACTLOG_PREFIX):
                suffix = name[len(MTP_ACTLOG_PREFIX):]
                setattr(module, attr, f"{MTP_ACTLOG_PREFIX}{predictor_idx}.{suffix}")
