from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="update_dynamic_gate")
class UpdateDynamicGateConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.builtin.update_dynamic_gate:UpdateDynamicGate"
    )
