from __future__ import annotations

import logging
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils.model_utils import get_composer_language_model

log = logging.getLogger(__name__)

__all__ = ["SkipDisbalanceStep", "zero_params"]


def zero_params(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()


class SkipDisbalanceStep(Algorithm):
    """
    Composer Algorithm that skip training step while detect high router expert disbalance.
    """

    def __init__(self, skip_step: bool = True):
        self.skip_step = skip_step

    def match(self, event: Event, state: State) -> bool:
        return event in (Event.AFTER_BACKWARD,)

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if self.skip_step:
            if event == Event.AFTER_BACKWARD:
                outer_model = state.model.model
                language_model = get_composer_language_model(state.model)
                if language_model._check_disbalance():
                    zero_params(outer_model)
        return
