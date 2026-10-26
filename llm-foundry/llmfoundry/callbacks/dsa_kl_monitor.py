import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

from llmfoundry.models.layers import dsa_monitor

__all__ = ["DSAMonitor", "DSAKLMonitor"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _KLMonitorConfig:
    enabled: bool = False
    log_interval: int = 20


@dataclass(frozen=True)
class _QueryMonitorConfig:
    enabled: bool = False
    log_interval: int = 100
    num_queries: int | None = 128
    q_start_pos: int = 0
    topk_values: tuple[int, ...] | None = None


def _config_dict(name: str, value: Mapping | None, valid_keys: set[str]) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    config = dict(value)
    unknown = set(config) - valid_keys
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
    return config


def _parse_kl_config(value: Mapping | None) -> _KLMonitorConfig:
    config = _config_dict("kl_loss", value, {"enabled", "log_interval"})
    result = _KLMonitorConfig(**config)
    if not isinstance(result.enabled, bool):
        raise TypeError("kl_loss.enabled must be a boolean")
    if result.log_interval <= 0:
        raise ValueError("kl_loss.log_interval must be positive")
    return result


def _parse_query_config(name: str, value: Mapping | None) -> _QueryMonitorConfig:
    config = _config_dict(
        name,
        value,
        {"enabled", "log_interval", "num_queries", "q_start_pos", "topk_values"},
    )
    topk_values = config.get("topk_values")
    if topk_values is not None:
        if not isinstance(topk_values, Sequence) or isinstance(
            topk_values, (str, bytes)
        ):
            raise TypeError(f"{name}.topk_values must be a list or null")
        if not topk_values:
            raise ValueError(f"{name}.topk_values must not be empty")
        if any(
            isinstance(topk, bool) or not isinstance(topk, int) for topk in topk_values
        ):
            raise TypeError(f"{name}.topk_values must contain integers")
        if any(topk <= 0 for topk in topk_values):
            raise ValueError(f"{name}.topk_values must contain positive values")
        if len(set(topk_values)) != len(topk_values):
            raise ValueError(f"{name}.topk_values must not contain duplicates")
        config["topk_values"] = tuple(sorted(topk_values))

    result = _QueryMonitorConfig(**config)
    if not isinstance(result.enabled, bool):
        raise TypeError(f"{name}.enabled must be a boolean")
    if result.log_interval <= 0:
        raise ValueError(f"{name}.log_interval must be positive")
    if result.num_queries is not None and result.num_queries <= 0:
        raise ValueError(f"{name}.num_queries must be positive or null")
    if result.q_start_pos < 0:
        raise ValueError(f"{name}.q_start_pos must be non-negative")
    return result


class DSAMonitor(Callback):
    """Log independently configurable DSA training and selection metrics."""

    def __init__(
        self,
        kl_loss: Mapping | None = None,
        attention_mass: Mapping | None = None,
        indexer_topk: Mapping | None = None,
    ):
        self.kl_loss = _parse_kl_config(kl_loss)
        self.attention_mass = _parse_query_config("attention_mass", attention_mass)
        self.indexer_topk = _parse_query_config("indexer_topk", indexer_topk)
        self.attention_mass_topk_values = self.attention_mass.topk_values or ()
        self.indexer_topk_values = self.indexer_topk.topk_values or ()
        self.layers = []
        self.tags = []
        self._device = None
        self._kl_loss_now = False
        self._attention_mass_now = False
        self._indexer_topk_now = False
        self._any_enabled = (
            self.kl_loss.enabled
            or self.attention_mass.enabled
            or self.indexer_topk.enabled
        )

    def _resolve_attention_mass_topk_values(self) -> tuple[int, ...]:
        unsupported_layers = [
            tag for tag, module in zip(self.tags, self.layers) if int(module.dsa_stage) != 1
        ]
        if unsupported_layers:
            raise ValueError(
                "attention_mass is supported only for DSA stage 1; "
                f"non-stage-1 layers: {unsupported_layers}"
            )
        model_topk_values = {int(module.indexer_topk) for module in self.layers}
        configured = self.attention_mass.topk_values
        if configured is None:
            if len(model_topk_values) != 1:
                raise ValueError(
                    "attention_mass.topk_values must be set when DSA layers use "
                    f"different model indexer_topk values: {sorted(model_topk_values)}"
                )
            return (next(iter(model_topk_values)),)

        return configured

    def _resolve_indexer_topk_values(self) -> tuple[int, ...]:
        model_topk_values = {int(module.indexer_topk) for module in self.layers}
        configured = self.indexer_topk.topk_values
        if configured is None:
            if len(model_topk_values) != 1:
                raise ValueError(
                    "indexer_topk.topk_values must be set when DSA layers use "
                    f"different model indexer_topk values: {sorted(model_topk_values)}"
                )
            return (next(iter(model_topk_values)),)

        stage_2_topk_values = [
            int(module.indexer_topk)
            for module in self.layers
            if int(module.dsa_stage) == 2
        ]
        if stage_2_topk_values:
            max_supported = min(stage_2_topk_values)
            unsupported = [topk for topk in configured if topk > max_supported]
            if unsupported:
                raise ValueError(
                    "indexer_topk.topk_values must not exceed model.indexer_topk "
                    f"for DSA stage 2 ({max_supported}); unsupported values: {unsupported}"
                )
        return configured

    def fit_start(self, state: State, logger: Logger):
        self.layers.clear()
        self.tags.clear()
        if not self._any_enabled:
            return
        for name, module in state.model.named_modules():
            if (
                hasattr(module, "dsa_stage")
                and hasattr(module, "indexer")
                and hasattr(module, "loss")
            ):
                match = re.search(r"(mtp_)?layers\.(\d+)", name)
                self.layers.append(module)
                # mtp_block.mtp_layers.N would otherwise collide with layers.N under
                # the same "layer_N" tag: the dict write silently drops one of them
                # while dsa_kl/mean still averages over all modules.
                self.tags.append(
                    f"{'mtp_' if match.group(1) else ''}layer_{match.group(2)}"
                    if match
                    else name
                )
        self._device = next(state.model.parameters()).device
        if not self.layers:
            log.warning("DSAMonitor: no DSA layers found; nothing to log.")
            return

        if self.attention_mass.enabled:
            self.attention_mass_topk_values = self._resolve_attention_mass_topk_values()
        if self.indexer_topk.enabled:
            self.indexer_topk_values = self._resolve_indexer_topk_values()
        for module in self.layers:
            if self.kl_loss.enabled:
                module._compute_kl_value = False
            if self.attention_mass.enabled:
                module._compute_attention_mass_value = False
                module.attention_mass_monitor_num_queries = (
                    self.attention_mass.num_queries
                )
                module.attention_mass_monitor_q_start_pos = (
                    self.attention_mass.q_start_pos
                )
                module.attention_mass_monitor_stats = None
                module.attention_mass_monitor_topk_values = (
                    self.attention_mass_topk_values
                )
            if self.indexer_topk.enabled:
                module._compute_indexer_topk_value = False
                module.indexer_topk_monitor_num_queries = self.indexer_topk.num_queries
                module.indexer_topk_monitor_q_start_pos = self.indexer_topk.q_start_pos
                module.indexer_topk_monitor_topk_values = self.indexer_topk_values
                module.indexer_topk_monitor_stats = None

    def batch_start(self, state: State, logger: Logger):
        if not self._any_enabled or not self.layers:
            return
        batch = state.timestamp.batch.value
        self._kl_loss_now = (
            self.kl_loss.enabled and batch % self.kl_loss.log_interval == 0
        )
        self._attention_mass_now = (
            self.attention_mass.enabled
            and batch % self.attention_mass.log_interval == 0
        )
        self._indexer_topk_now = (
            self.indexer_topk.enabled and batch % self.indexer_topk.log_interval == 0
        )
        for module in self.layers:
            if self.kl_loss.enabled:
                module._compute_kl_value = self._kl_loss_now
            if self.attention_mass.enabled:
                module._compute_attention_mass_value = self._attention_mass_now
                if self._attention_mass_now:
                    module.attention_mass_monitor_stats = None
            if self.indexer_topk.enabled:
                module._compute_indexer_topk_value = self._indexer_topk_now
                if self._indexer_topk_now:
                    module.indexer_topk_monitor_stats = None

    def _reduce_layer_stats(
        self, attribute: str, shape: int | tuple[int, ...]
    ) -> torch.Tensor:
        stats = []
        for module in self.layers:
            value = getattr(module, attribute)
            stats.append(
                torch.zeros(
                    shape,
                    device=self._device,
                    dtype=torch.float32,
                )
                if value is None
                else value.to(self._device, torch.float32)
            )
            setattr(module, attribute, None)
        stats_tensor = torch.stack(stats)
        if dist.get_world_size() > 1:
            # Raw sums/counts combine disjoint SP query shards and DP samples.
            dist.all_reduce(stats_tensor, reduce_operation="SUM")
        return stats_tensor

    @staticmethod
    def _log_query_counts(
        metrics: dict,
        prefix: str,
        stats: torch.Tensor,
        monitored_index: int,
        eligible_index: int,
        total_index: int,
    ) -> None:
        reference = stats[0]
        monitored_count = float(reference[monitored_index])
        eligible_count = float(reference[eligible_index])
        total_count = float(reference[total_index])
        metrics[f"{prefix}/monitored_q_count"] = monitored_count
        metrics[f"{prefix}/eligible_q_count"] = eligible_count
        metrics[f"{prefix}/total_q_count"] = total_count
        if total_count > 0:
            metrics[f"{prefix}/monitored_q_fraction"] = monitored_count / total_count
            metrics[f"{prefix}/eligible_q_fraction"] = eligible_count / total_count

    def batch_end(self, state: State, logger: Logger):
        if not (
            self._kl_loss_now or self._attention_mass_now or self._indexer_topk_now
        ):
            return

        metrics = {}
        if self._kl_loss_now:
            values = torch.stack(
                [
                    torch.as_tensor(
                        module.loss,
                        device=self._device,
                        dtype=torch.float32,
                    )
                    for module in self.layers
                ]
            )
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(values, reduce_operation="SUM")
                values = values / world_size
            values = values.tolist()
            metrics.update(
                {f"dsa_kl/{tag}": value for tag, value in zip(self.tags, values)}
            )
            metrics["dsa_kl/mean"] = sum(values) / len(values)

        if self._attention_mass_now:
            stats = self._reduce_layer_stats(
                "attention_mass_monitor_stats",
                (
                    len(self.attention_mass_topk_values),
                    dsa_monitor.NUM_ATTENTION_MASS_STATS,
                ),
            )
            for topk_index, topk in enumerate(self.attention_mass_topk_values):
                values = []
                for tag, layer_stats in zip(self.tags, stats[:, topk_index]):
                    monitored_count = layer_stats[
                        dsa_monitor.ATTENTION_MASS_MONITORED_Q_COUNT
                    ]
                    if float(monitored_count) <= 0:
                        continue
                    value = float(
                        layer_stats[dsa_monitor.ATTENTION_MASS_SUM] / monitored_count
                    )
                    metrics[f"dsa_attention_mass/topk_{topk}/attention_mass/{tag}"] = (
                        value
                    )
                    values.append(value)
                if values:
                    metrics[f"dsa_attention_mass/topk_{topk}/attention_mass/mean"] = (
                        sum(values) / len(values)
                    )
            self._log_query_counts(
                metrics,
                "dsa_attention_mass",
                stats[:, 0],
                dsa_monitor.ATTENTION_MASS_MONITORED_Q_COUNT,
                dsa_monitor.ATTENTION_MASS_ELIGIBLE_Q_COUNT,
                dsa_monitor.ATTENTION_MASS_TOTAL_Q_COUNT,
            )

        if self._indexer_topk_now:
            stats = self._reduce_layer_stats(
                "indexer_topk_monitor_stats",
                (
                    len(self.indexer_topk_values),
                    dsa_monitor.NUM_INDEXER_TOPK_STATS,
                ),
            )
            value_stats = (
                ("position_range", dsa_monitor.POSITION_RANGE_SUM),
                ("q_pos_minus_min_topk", dsa_monitor.POSITION_DISTANCE_SUM),
                ("position_std", dsa_monitor.POSITION_STD_SUM),
            )
            for topk_index, topk in enumerate(self.indexer_topk_values):
                layer_means = {name: [] for name, _ in value_stats}
                for tag, layer_stats in zip(self.tags, stats[:, topk_index]):
                    monitored_count = layer_stats[
                        dsa_monitor.INDEXER_TOPK_MONITORED_Q_COUNT
                    ]
                    if float(monitored_count) <= 0:
                        continue
                    for name, index in value_stats:
                        value = float(layer_stats[index] / monitored_count)
                        metrics[f"dsa_indexer_topk/topk_{topk}/{name}/{tag}"] = value
                        layer_means[name].append(value)
                for name, values in layer_means.items():
                    if values:
                        metrics[f"dsa_indexer_topk/topk_{topk}/{name}/mean"] = sum(
                            values
                        ) / len(values)
            self._log_query_counts(
                metrics,
                "dsa_indexer_topk",
                stats[:, 0],
                dsa_monitor.INDEXER_TOPK_MONITORED_Q_COUNT,
                dsa_monitor.INDEXER_TOPK_ELIGIBLE_Q_COUNT,
                dsa_monitor.INDEXER_TOPK_TOTAL_Q_COUNT,
            )

        if metrics:
            logger.log_metrics(metrics)


class DSAKLMonitor(DSAMonitor):
    """Backward-compatible wrapper for the original KL-only callback."""

    def __init__(self, batch_log_interval: int = 20):
        super().__init__(
            kl_loss={
                "enabled": True,
                "log_interval": batch_log_interval,
            },
        )
