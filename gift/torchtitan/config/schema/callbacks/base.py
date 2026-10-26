from typing import ClassVar

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.validation import Issue


@config(family="callback")
class CallbackConfig(Configurable.Config):
    requires: ClassVar[tuple[str, ...]] = ()

    def validate_dependencies(self, previous: list["CallbackConfig"]) -> list[Issue]:
        """Check dependencies against callbacks built earlier in YAML order."""
        return []
