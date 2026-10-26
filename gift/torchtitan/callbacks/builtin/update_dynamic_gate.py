from torch import nn

from typing import TypeAlias

from torchtitan.config.schema.callbacks.builtin.update_dynamic_gate import (
    UpdateDynamicGateConfig,
)
from torchtitan.callbacks.base import MutableState, Mutation, Mutator
from torchtitan.config import Configurable


class UpdateDynamicGate(Configurable, Mutator):
    """Switches modules that expose ``switch2uniform(step)`` at each step.

    Kept for the upcoming dynamic-gate modules: today no module in the tree
    implements ``switch2uniform``, so the callback is a no-op until they land.
    """

    # Changes the routing behaviour of the gates, not any tensor.
    mutates = frozenset({Mutation.routing})

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = UpdateDynamicGateConfig

    def __init__(self, config: Config) -> None:
        self.config = config

    def step(self, *, model_parts: list[nn.Module], step: int) -> None:
        for model_part in model_parts:
            for module in model_part.modules():
                switch = getattr(module, "switch2uniform", None)
                if callable(switch):
                    switch(step)

    def after_optimizer_step(self, state: MutableState) -> None:
        self.step(model_parts=state.trainer.model_parts, step=state.step)
