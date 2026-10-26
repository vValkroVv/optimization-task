import math

from torchtitan.config.configurable import config, OwnerFilled
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.protocols.module import ModuleConfig


def validate_norm_eps(eps: float) -> None:
    if not math.isfinite(eps) or eps <= 0:
        raise ConfigFieldError("eps", "must be finite and positive")


@config(family="norm", default="rmsnorm")
class BaseNormConfig(ModuleConfig):
    """A normalization layer, picked by ``type``.

    ``normalized_shape`` left ``None`` is filled by the owner; until then the
    config cannot be built.
    """

    normalized_shape: OwnerFilled[int] = None

    def build(self, **kwargs):
        if self.normalized_shape is None:
            raise ValueError(
                f"{type(self).__qualname__} has no normalized_shape: the config is "
                "partial (an owner fills the width), it cannot be built as it is."
            )
        return ModuleConfig.build(self, **kwargs)
