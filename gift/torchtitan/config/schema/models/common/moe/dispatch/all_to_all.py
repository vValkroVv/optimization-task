from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.moe.dispatch.base import (
    BaseTokenDispatcherConfig,
)


@config(type="all_to_all")
class AllToAllTokenDispatcherConfig(BaseTokenDispatcherConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.moe.dispatch.all_to_all:AllToAllTokenDispatcher"
    )
