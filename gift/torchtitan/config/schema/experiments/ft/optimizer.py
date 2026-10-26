from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.components.optimizer import OptimizersContainerConfig


# Its own family, not a member of the core `optimizer` family: only the ft
# trainer takes it, and the ft trainer takes nothing else.
@config(family="ft optimizer", default="ft", type="ft")
class FTOptimizersContainerConfig(OptimizersContainerConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.ft.optimizer:FTOptimizersContainer"
    )
