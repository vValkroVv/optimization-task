from __future__ import annotations

import logging
from typing import Iterable

import torch.nn as nn
from composer.core import Callback, State
from composer.loggers import Logger

from llmfoundry.experimental.prores import prores_alpha

__all__ = ["ProResWarmup"]

log = logging.getLogger(__name__)


class ProResWarmup(Callback):
    """Update and log ProRes residual scales for GigaMix Sandwich Norm."""

    def __init__(
        self,
        batch_log_interval: int = 1,
        log_per_layer: bool = False,
    ) -> None:
        if batch_log_interval <= 0:
            raise ValueError(
                f"batch_log_interval must be positive, got {batch_log_interval}."
            )
        self.batch_log_interval = batch_log_interval
        self.log_per_layer = log_per_layer
        self._enabled = False
        self._complete = False
        self._modules: list[nn.Module] = []
        self._last_alphas: list[float] = []

    def _iter_modules(self, state: State) -> Iterable[nn.Module]:
        roots = [state.model]
        inner = getattr(state.model, "model", None)
        if inner is not None:
            roots.append(inner)
        seen_roots: set[int] = set()
        seen_modules: set[int] = set()
        for root in roots:
            root = getattr(root, "_module", root)
            root_id = id(root)
            if root_id in seen_roots or not hasattr(root, "modules"):
                continue
            seen_roots.add(root_id)
            for module in root.modules():
                module_id = id(module)
                if module_id in seen_modules:
                    continue
                seen_modules.add(module_id)
                yield module

    def _find_modules(self, state: State) -> list[nn.Module]:
        modules = [
            module
            for module in self._iter_modules(state)
            if getattr(module, "_prores_root_module", False)
            and getattr(module, "_prores_enabled", False)
            and hasattr(module, "set_prores_alpha")
        ]
        return sorted(modules, key=lambda module: getattr(module, "_prores_layer_idx"))

    def fit_start(self, state: State, logger: Logger) -> None:
        self._modules = self._find_modules(state)
        if not self._modules:
            log.warning("ProResWarmup: no enabled GigaMix ProRes modules found.")
            return
        self._enabled = True
        self._update_modules(step=int(state.timestamp.batch.value))
        self._log_metrics(logger)
        log.info("ProResWarmup enabled for %d GigaMix layers.", len(self._modules))

    def batch_start(self, state: State, logger: Logger) -> None:
        if not self._enabled:
            return
        if not self._complete:
            self._update_modules(step=int(state.timestamp.batch.value))
        if state.timestamp.batch.value % self.batch_log_interval == 0:
            self._log_metrics(logger)

    def _update_modules(self, step: int) -> None:
        alphas = []
        for module in self._modules:
            alpha = prores_alpha(
                step=step,
                layer_idx=int(getattr(module, "_prores_layer_idx")),
                num_layers=int(getattr(module, "_prores_num_layers")),
                warmup_steps=int(getattr(module, "_prores_warmup_steps")),
                start_value=float(getattr(module, "_prores_start_value")),
                schedule=getattr(module, "_prores_schedule"),
                power=float(getattr(module, "_prores_power")),
            )
            module.set_prores_alpha(alpha)
            alphas.append(alpha)
        self._last_alphas = alphas
        self._complete = bool(alphas) and min(alphas) >= 1.0

    def _log_metrics(self, logger: Logger) -> None:
        if not self._last_alphas:
            return
        alpha_min = min(self._last_alphas)
        alpha_max = max(self._last_alphas)
        alpha_mean = sum(self._last_alphas) / len(self._last_alphas)
        metrics = {
            "prores/alpha_min": alpha_min,
            "prores/alpha_max": alpha_max,
            "prores/alpha_mean": alpha_mean,
            "prores/alpha/layer_000": self._last_alphas[0],
            "prores/alpha/layer_last": self._last_alphas[-1],
            "prores/complete": float(self._complete),
        }
        if self.log_per_layer:
            for module, alpha in zip(self._modules, self._last_alphas):
                layer_idx = int(getattr(module, "_prores_layer_idx"))
                metrics[f"prores/alpha/layer_{layer_idx:03d}"] = alpha
        logger.log_metrics(metrics)
