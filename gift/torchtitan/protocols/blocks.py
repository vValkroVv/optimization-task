"""Structural description of a model: repeated ``Block``s and ordered ``Segment``s.

A model states only what it is; policy (FSDP grouping, reshard, prefetch, AC
mode, pipeline cuts) stays in ``torchtitan/distributed/``. Nothing here imports
``torch.distributed``.
"""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass

import torch.nn as nn

from .module import Module
from typing import TypeAlias

from torchtitan.config.schema.protocols.blocks import BlockConfig

__all__ = ["Block", "Segment", "peel", "walk_segments"]


class Block(Module):
    """A repeated unit: what AC, compile, FSDP and PP each treat as one item.

    A pure marker, deliberately without role accessors: a ``moe_modules()`` here
    would make "has MoE" a first-class question. A consumer that cares
    about a feature finds it by type — ``isinstance(m, MoE)``.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = BlockConfig


@dataclass(frozen=True)
class Segment:
    """One element of a model's ordered structural description.

    ``fqn`` comes from the real module tree, so it cannot disagree with
    ``model.get_submodule(fqn)``.
    """

    fqn: str
    module: nn.Module
    is_block: bool
    # Enough to swap the module in place: AC wraps a block and re-registers it.
    container: nn.Module
    key: str

    @property
    def group(self) -> str:
        """FQN of the parent container, ``""`` at the root.

        Load-bearing, not just cosmetic: it names a block group in log lines and
        in ``apply_fsdp(coalesce_block_groups=...)``, so it is user-visible.
        """
        return self.fqn.rpartition(".")[0]

    def rebind(self, new: nn.Module) -> None:
        """Swap this segment's module in its parent container.

        Not ``replace``: ``dataclasses.replace`` is used a few lines below with the
        opposite meaning (a pure copy, not a mutation of the module tree).
        """
        self.container.register_module(self.key, new)


def _defines_forward(module: nn.Module) -> bool:
    """Whether a module's type declares a ``forward`` of its own.

    A bare ``nn.ModuleList``/``ModuleDict`` does not: its children are called
    individually (``self.deepstack_merger_list[i](x)``), so it never completes a
    forward. A segment holding one would sit in an FSDP group whose post-forward
    hook never fires, keeping the group gathered for the whole step.
    """
    return type(module).forward is not nn.Module.forward


def peel(module: nn.Module) -> nn.Module:
    """Look through the AC wrapper, which hides the real module from ``isinstance``.

    Keyed on the attribute the wrapper registers, not on "one child and its own
    forward": that shape also fits a one-entry container, so the target depended
    on arity. ``apply_compile`` compiles in place and leaves no wrapper.
    """
    while (inner := getattr(module, "_checkpoint_wrapped_module", None)) is not None:
        module = inner
    return module


def _contains_block(module: nn.Module) -> bool:
    """Whether a subtree holds any ``Block``."""
    return any(isinstance(m, Block) for m in module.modules())


def _owns_state(module: nn.Module) -> bool:
    """Whether a subtree owns any parameter or buffer.

    Owning nothing means nothing to wrap, shard or place — a container emptied by
    a pipeline split, or an ``nn.Identity`` sentinel — so it is not a segment.
    """
    return (
        next(module.parameters(), None) is not None
        or next(module.buffers(), None) is not None
    )


def walk_segments(root: nn.Module) -> Iterator[Segment]:
    """Segments of ``root``'s subtree, relative to ``root``, in forward order.

    A ``Block`` (possibly behind a wrapper) is one segment and is not descended
    into, so the outermost wins — an MTP head, not the block inside it. A segment
    is always a module that runs: a container without its own ``forward`` is
    descended into, not reported. That descent is also what lets a model declare
    third-party blocks, which it does by relabelling reported FQNs.

    Recursion goes through ``Module.named_segments`` so a nested model's own
    override is honoured; that is what lets a tower carry its own blocks and cost
    the parent nothing to embed.
    """
    for name, child in root.named_children():
        inner = peel(child)
        if isinstance(inner, Block):
            # ``child``, not ``inner``: the wrapper is the wrap/shard target.
            yield Segment(
                fqn=name, module=child, is_block=True, container=root, key=name
            )
            continue

        nested: list[Segment] = []
        if _contains_block(child) or not _defines_forward(inner):
            # Descend to reach blocks, or through a container that never runs a
            # forward of its own. Otherwise a non-block segment stays the largest
            # block-free part that does run (``merger``, not each ``Linear``).
            if inner is child and isinstance(child, Module):
                nested = list(child.named_segments())
            else:
                # Wrapper around a non-block: walk the real tree so its own name
                # stays in the FQN and ``get_submodule`` keeps resolving.
                nested = list(walk_segments(child))

        if nested:
            for seg in nested:
                yield dataclasses.replace(seg, fqn=f"{name}.{seg.fqn}")
        elif _owns_state(child):
            yield Segment(
                fqn=name, module=child, is_block=False, container=root, key=name
            )
