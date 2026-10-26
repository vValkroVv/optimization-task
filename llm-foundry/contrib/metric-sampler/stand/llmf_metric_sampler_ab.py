from __future__ import annotations

import time

from composer.core import State
from composer.loggers import Logger
from composer.utils import dist
from composer.utils.shared_state import global_actlog_state
from llmfoundry.callbacks.metric_sampler import MetricSampler
from llmfoundry.models.layers.moe.routers import MoEGate

__all__ = ["MetricSamplerAB"]

ARMS = ("off", "base", "tokens", "full", "acts", "detailed")
TOKEN_ARMS = ("tokens", "full", "acts", "detailed")
GRAD_ARMS = ("full", "acts", "detailed")
ACTIVATION_ARMS = ("acts", "detailed")
DISTRIBUTION_INTERVAL = 100
DETAILED_DISTRIBUTION_INTERVAL = 1


class MetricSamplerAB(MetricSampler):
    def __init__(self, output_dir: str, window_steps: int = 50, block_batches: int = 20):
        if block_batches <= 0:
            raise ValueError("block_batches must be positive")
        super().__init__(output_dir, window_steps)
        self.block_batches = block_batches
        self.batch_starts: list[tuple[int, str, float]] = []
        self.arm = "off"
        self.routing_observers: list[tuple[MoEGate, object]] = []
        self.loss_observer = None
        self.pre_reduce_targets: list[tuple[object, object]] = []
        self.activation_observer = None

    def fit_start(self, state: State, logger: Logger):
        super().fit_start(state, logger)
        if self.disabled:
            raise RuntimeError("metric sampler A/B setup failed")
        self.routing_observers = [
            (gate, gate.routing_observer) for _, gate in self._routed_gates
        ]
        self.loss_observer = self.causal_lm.loss_observer
        self.pre_reduce_targets = [
            (super_tensor, super_tensor.pre_reduce_observer)
            for super_tensor in self._pre_reduce_super_tensors
        ]
        self.activation_observer = self._activation_observer

    def batch_start(self, state: State, logger: Logger):
        if self.disabled:
            raise RuntimeError("metric sampler A/B capture failed")
        # The batch counter advances before BATCH_END, so the arm is decided once per step here.
        step = state.timestamp.batch.value
        self.arm = ARMS[(step // self.block_batches) % len(ARMS)]
        self.batch_starts.append((step, self.arm, time.perf_counter()))
        token_capture_enabled = self.arm in TOKEN_ARMS
        for gate, observer in self.routing_observers:
            gate.routing_observer = observer if token_capture_enabled else None
        self.causal_lm.loss_observer = self.loss_observer if token_capture_enabled else None
        for super_tensor, observer in self.pre_reduce_targets:
            super_tensor.pre_reduce_observer = observer if self.arm in GRAD_ARMS else None
        global_actlog_state.activation_observer = None
        for dispatcher, _ in self._dispatcher_observers:
            dispatcher.assignment_observer = None
        if self.arm == "off":
            return
        self.distribution_interval = (
            DETAILED_DISTRIBUTION_INTERVAL if self.arm == "detailed" else DISTRIBUTION_INTERVAL
        )
        super().batch_start(state, logger)

    def before_forward(self, state: State, logger: Logger):
        if self.arm != "off":
            super().before_forward(state, logger)
            if not self.disabled and self.arm in ACTIVATION_ARMS:
                global_actlog_state.activation_observer = self.activation_observer
            else:
                global_actlog_state.activation_observer = None

    def after_forward(self, state: State, logger: Logger):
        if self.arm != "off":
            super().after_forward(state, logger)

    def after_loss(self, state: State, logger: Logger):
        if self.arm != "off":
            super().after_loss(state, logger)

    def before_backward(self, state: State, logger: Logger):
        if self.arm != "off":
            super().before_backward(state, logger)

    def after_backward(self, state: State, logger: Logger):
        if self.arm != "off":
            super().after_backward(state, logger)

    def after_train_batch(self, state: State, logger: Logger):
        if self.arm != "off":
            super().after_train_batch(state, logger)

    def batch_end(self, state: State, logger: Logger):
        if self.arm != "off":
            super().batch_end(state, logger)

    def fit_end(self, state: State, logger: Logger):
        super().fit_end(state, logger)
        starts_dir = self.output_dir / "ab_batch_starts"
        starts_dir.mkdir(parents=True, exist_ok=True)
        with open(starts_dir / f"rank{dist.get_global_rank():04d}.tsv", "w") as file:
            for step, arm, started in self.batch_starts:
                file.write(f"{step}\t{arm}\t{started:.6f}\n")
