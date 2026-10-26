"""Hidden z-loss config and its glob placement over a model config tree.

``torchtitan.models.common.regularization.attach_hidden_z_loss`` applies it.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase

from torchtitan.config.errors import ConfigFieldError
from torchtitan.tools.logging import logger

__all__ = ["HiddenZLoss", "set_hidden_z_loss"]


def validate_z_loss_coef(coef: float, field: str) -> None:
    """Loss penalties must not reverse the objective or make it non-finite."""
    if not math.isfinite(coef) or coef < 0:
        raise ConfigFieldError(field, f"must be finite and >= 0, got {coef}")


@dataclass(frozen=True)
class HiddenZLoss:
    """A hidden z-loss on one module, declared on that module's config.

    ``on="input"`` penalizes the first input, which reaches activations no
    module produces, e.g. the trunk hidden state before the final norm.
    """

    coef: float
    on: str = "output"

    def __post_init__(self):
        validate_z_loss_coef(self.coef, "coef")
        if self.on not in ("output", "input"):
            raise ValueError(
                f"HiddenZLoss.on must be 'output' or 'input', got {self.on!r}"
            )


def set_hidden_z_loss(
    model_config, patterns: Mapping[str, "float | HiddenZLoss"]
) -> None:
    """Declare a hidden z-loss on every module config whose FQN matches a pattern.

    Patterns are shell globs over config FQNs
    (``layers.3.attention.qkv_linear.wq``), which mirror module FQNs. A config
    whose class declares ``hidden_z_loss_projections`` also matches
    ``<fqn>.<projection>`` (``layers.3.moe.experts.w1``): the pattern sets that
    projection's coefficient field instead of a module hook. A pattern
    matching nothing is a typo, not a no-op; later patterns win on overlap.
    """
    from torchtitan.config.schema.protocols.module import iter_module_configs

    matched = {pattern: 0 for pattern in patterns}
    for fqn, config in iter_module_configs(model_config):
        projections = getattr(type(config), "hidden_z_loss_projections", {})
        for pattern, spec in patterns.items():
            if fnmatchcase(fqn, pattern):
                matched[pattern] += 1
                config.hidden_z_loss = (
                    spec if isinstance(spec, HiddenZLoss) else HiddenZLoss(coef=spec)
                )
            for name, coef_field in projections.items():
                if not fnmatchcase(f"{fqn}.{name}", pattern):
                    continue
                matched[pattern] += 1
                if isinstance(spec, HiddenZLoss) and spec.on != "output":
                    # Penalizing the output instead would be silently wrong.
                    logger.warning(
                        f"hidden z-loss pattern {pattern!r} with on={spec.on!r} "
                        f"skips {fqn}.{name}: a projection only has an output."
                    )
                    continue
                coef = spec.coef if isinstance(spec, HiddenZLoss) else spec
                setattr(config, coef_field, float(coef))

    unmatched = sorted(p for p, n in matched.items() if n == 0)
    if unmatched:
        raise ValueError(
            f"hidden z-loss patterns match no module: {unmatched}. "
            "Patterns are globs over module FQNs."
        )
