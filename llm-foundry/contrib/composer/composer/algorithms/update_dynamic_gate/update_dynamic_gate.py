from __future__ import annotations

import logging
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils.model_utils import get_composer_lm_backbone

log = logging.getLogger(__name__)

__all__ = [
    "UpdateDynamicGate",
]


class UpdateDynamicGate(Algorithm):
    """
    Composer Algorithm that switches between FirstGroup and Uniform routers.
    """

    def __init__(self):
        self.dynamic_gates_list = []

    def match(self, event: Event, state: State) -> bool:
        return event in (Event.INIT, Event.BEFORE_TRAIN_BATCH)

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.INIT:
            for layer in get_composer_lm_backbone(state.model).layers:
                if hasattr(layer, "self_attn"):
                    if not layer.self_attn.is_gate:
                        continue
                    gate = layer.self_attn._gate
                    if hasattr(gate, "gate_type") and gate.gate_type == "dynamic":
                        self.dynamic_gates_list.append(gate)

        elif event == Event.BEFORE_TRAIN_BATCH:
            curr_batch_num = int(state.timestamp.batch)
            for gate in self.dynamic_gates_list:
                gate.switch2uniform(curr_step=curr_batch_num)
