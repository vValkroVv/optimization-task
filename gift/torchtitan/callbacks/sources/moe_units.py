"""Discovery of MoE layers for the monitoring sources.

Found by type — ``isinstance(module, MoE)`` over ``named_modules()`` — so a
nested MoE (an MTP head's, a tower's) needs no special case and the model
exposes nothing monitoring-specific.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from torchtitan.callbacks.sources.base import canonical_fqn
from torchtitan.models.common.moe import MoE
from torchtitan.protocols.moe_monitoring import DispatchObservable


@dataclass(frozen=True, kw_only=True)
class MoEMonitorUnit:
    """The parts of one MoE layer that a monitor needs to reach.

    ``module`` is the MoE itself, and it is there because it owns
    ``expert_bias_E``: the router returns its scores UNBIASED, while the matrix
    it routes with — is ``scores + expert_bias``.
    """

    name: str
    module: nn.Module
    router: nn.Module
    experts: nn.Module
    dispatcher: DispatchObservable | None = None


def iter_moe_monitor_units(model_parts: Iterable[nn.Module]) -> list[MoEMonitorUnit]:
    """Every MoE in the parts, found by type, named by its canonical FQN."""
    return [
        MoEMonitorUnit(
            name=canonical_fqn(fqn),
            module=module,
            router=module.router,
            experts=module.experts,
            dispatcher=module.experts.token_dispatcher,
        )
        for model_part in model_parts
        for fqn, module in model_part.named_modules()
        if isinstance(module, MoE)
    ]
