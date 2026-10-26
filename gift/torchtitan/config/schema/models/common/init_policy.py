"""Parameter-init policy: one init table per ``init_role``, applied to every
module config without a ``param_init`` of its own."""

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import ClassVar

from torchtitan.config.schema.models.common.norms.base import BaseNormConfig
from torchtitan.config.schema.models.common.param_init import (
    NormUnitScale,
    materialize_init,
)
from torchtitan.config.schema.protocols.module import (
    InitSpec,
    Ones,
    ParamInit,
    Skip,
    Zeros,
    iter_module_configs,
)

__all__ = [
    "apply_init",
    "EMBEDDING_SKIP_INIT",
    "InitPolicy",
    "tied_embedding_init",
]


@dataclass(kw_only=True)
class InitPolicy:
    linear: ParamInit
    norm: ParamInit
    embedding: ParamInit
    # Roles that fall back to ``linear`` when unset.
    linear_out: ParamInit | None = None
    """Output projections of a sublayer (``wo``, ``w2`` / ``w3``); the role a
    family scales by depth."""
    lm_head: ParamInit | None = None
    router_gate: ParamInit | None = None
    # Roles only some families have.
    gated_norm: ParamInit | None = None
    experts: ParamInit | None = None
    conv: ParamInit | None = None
    o_norm: ParamInit | None = None
    gated_delta: ParamInit | None = None
    sinks: ParamInit | None = None

    _LINEAR_FALLBACK: ClassVar[frozenset[str]] = frozenset(
        {"linear_out", "lm_head", "router_gate"}
    )

    def __post_init__(self) -> None:
        # replace() copies the policy shallowly; every role needs its own table.
        for role in fields(self):
            table = getattr(self, role.name)
            if table is not None:
                setattr(self, role.name, deepcopy(dict(table)))

    def for_role(self, role: str) -> ParamInit | None:
        try:
            value = getattr(self, role)
        except AttributeError:
            raise ValueError(f"init policy has no role {role!r}") from None
        if value is None and role in self._LINEAR_FALLBACK:
            return self.linear
        return value


EMBEDDING_SKIP_INIT: ParamInit = {"weight": Skip()}
"""The embedding of a weight-tied model: it shares the LM head's weight."""


def tied_embedding_init(policy: InitPolicy) -> ParamInit:
    """Embedding init of a weight-tied model: skipped, the LM head's init decides."""
    # ``init.embedding`` does not apply: the LM head's table initializes the
    # shared weight.
    return dict(EMBEDDING_SKIP_INIT)


def apply_init(config, policy: InitPolicy, *, layer_id: int, dim: int) -> None:
    """Give every module config under *config* without a ``param_init`` its
    role's table, then resolve the norm-unit-scale, depth- and width-scaled
    inits of every module against it, *layer_id* and *dim*."""
    for fqn, module in iter_module_configs(config):
        if module._explicit_param_init_names is None:
            module._explicit_param_init_names = frozenset(module.param_init or {})
        if module.init_role is None:
            module.init_role = type(module).default_init_role
        if module.param_init is None:
            if not module.init_role:
                continue
            table = policy.for_role(module.init_role)
            if table is None:
                raise ValueError(
                    f"{fqn}: no param_init and the init policy has nothing for "
                    f"role {module.init_role!r}"
                )
            module.param_init = table
        elif module.param_init and module.init_role:
            # A module override changes named entries without dropping the role's
            # other parameters. Empty tables are rejected by model validation.
            table = policy.for_role(module.init_role)
            if table is not None:
                module.param_init = {**table, **module.param_init}
        module.param_init = {
            name: materialize_init(
                _norm_unit_scale(spec, module, fqn), layer_id=layer_id, dim=dim
            )
            for name, spec in module.param_init.items()
        }


def _norm_unit_scale(spec: InitSpec, module, fqn: str) -> InitSpec:
    """*spec*, with a :class:`NormUnitScale` resolved against the norm
    *module*: zeros when it is zero-centered, ones otherwise."""
    if not isinstance(spec, NormUnitScale):
        return spec
    # Not every norm is in the ``norm`` family (a GDN's fused output norm).
    if hasattr(module, "zero_centered"):
        return Zeros() if module.zero_centered else Ones()
    if isinstance(module, BaseNormConfig):
        return Ones()
    raise ValueError(
        f"{fqn}: norm_unit_scale initializes a norm's weight, but "
        f"{type(module).__name__} is not a norm"
    )
