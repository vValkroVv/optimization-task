from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.moe.dispatch.base import (
    BaseTokenDispatcherConfig,
)


@config(type="deepep")
class DeepEPTokenDispatcherConfig(BaseTokenDispatcherConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.moe.dispatch.deepep:DeepEPTokenDispatcher"
    )

    num_sms: int | None = None
