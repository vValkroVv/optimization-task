# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Maintain Exponential Moving Average (EMA) of Weights."""
import logging

from composer.core import Callback, State, Time
from composer.loggers import Logger
from giga_fsdp import GigaFSDP
from composer.utils import TimerType, callback_timer

__all__ = ['EMA']


class EMA(Callback):
    """Updates EMA of model parameters.

    This callback iterates over all model layers to recalculate EMA of parameters.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import EMA
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[EMA(0.1, 5)],
            ... )
    """

    def __init__(
        self,
        alpha: float,
        update_period: int = 1,
        start_batch: int = 1,
        start_batch_force_update: bool = True
    ) -> None:
        assert update_period > 0, f"update_period should be positive integer, but got {update_period}"
        update_interval = f"{update_period}ba"
        self.update_interval = Time.from_timestring(update_interval)

        if start_batch <= 0:
            raise ValueError(f"`start_batch` is expected to be a positive integer, got {start_batch}.")

        self.start_batch = Time.from_timestring(f"{start_batch}ba")
        self.start_batch_force_update = start_batch_force_update

        assert alpha > 0 and alpha < 1, f"alpha = {alpha} is out of (0, 1) interval"
        self.alpha = alpha

    def before_load(self, state: State, logger: Logger) -> None:
        assert isinstance(state.model.model, GigaFSDP), f'state.model.model of type {type(state.model.model)} is not an instance of GigaFSDP'
        state.model.model.setup_ema_buffer()

    def after_load(self, state: State, logger: Logger) -> None:
        assert isinstance(state.model.model, GigaFSDP), f'state.model.model of type {type(state.model.model)} is not an instance of GigaFSDP'

        state.model.model.update_ema_buffer_after_load()

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def batch_end(self, state: State, logger: Logger) -> None:
        assert isinstance(state.model.model, GigaFSDP), f'state.model.model of type {type(state.model.model)} is not an instance of GigaFSDP'
        if state.timestamp.get(self.start_batch.unit).value == self.start_batch.value:
            state.model.model.update_ema_buffer_after_load(force_update=self.start_batch_force_update)
        if (state.timestamp.get(self.start_batch.unit).value >= self.start_batch.value) and (state.timestamp.get(self.update_interval.unit).value % self.update_interval.value == 0):
            state.model.model.update_ema_params(self.alpha)

    def eval_before_all(self, state: State, logger: Logger) -> None:
        assert isinstance(state.model.model, GigaFSDP), f'state.model.model of type {type(state.model.model)} is not an instance of GigaFSDP'
        if state.timestamp.get(self.start_batch.unit).value >= self.start_batch.value:
            state.model.model.swap_ema_and_weights()

    def eval_after_all(self, state: State, logger: Logger) -> None:
        self.eval_before_all(state, logger)

    def eval_standalone_start(self, state: State, logger: Logger) -> None:
        self.eval_before_all(state, logger)

    def eval_standalone_end(self, state: State, logger: Logger) -> None:
        self.eval_before_all(state, logger)

    def state_dict(self):
        return {'start_batch': self.start_batch.value}
