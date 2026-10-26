import time

import torch

from typing import TypeAlias

from torchtitan.config.schema.callbacks.sources.data_source import DataSourceConfig
from torchtitan.callbacks.base import ObserverState
from torchtitan.callbacks.metrics.sample import Sample
from torchtitan.config.schema.callbacks.metrics.sample import MetricConfig
from torchtitan.callbacks.sources.base import MonitorSource
from torchtitan.components.loss import IGNORE_INDEX


class DataSource(MonitorSource):
    """Collects dataloader timing and optional batch statistics.

    Built when the preset lists it: ``sources={"data": DataSource.Config(...)}``
    in the monitoring callback's config — there is no enable flag.

    Logged metrics:
        - ``time_seconds/batch/dataloader_fetch``: wall-clock time spent in
          ``next(data_iterator)``, summed over the step's microbatches

    Additional metrics when ``emit_batch_stats`` is enabled:
        - ``data/microbatches``: microbatches fetched this step
        - ``data/local_samples``: sum of batch sizes over microbatches
        - ``data/seq_len``: label sequence length of the last microbatch
        - ``data/label_tokens``: total label positions
        - ``data/supervised_label_tokens``: labels != IGNORE_INDEX
        - ``data/ignored_label_tokens``: labels == IGNORE_INDEX
        - ``data/ignored_label_ratio``: ignored / total label positions
        - ``data/microbatch_loss/{mean,max,min}``: spread of this step's
          per-microbatch losses — a coarse bad-data-chunk detector
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = DataSourceConfig

    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._reset_counters()
        self._fetch_start_time: float | None = None

    def batch_start(self, state: ObserverState) -> None:
        super().batch_start(state)
        self._reset_counters()

    def before_dataloader(self, state: ObserverState) -> None:
        del state
        if not self.active:
            return
        self._fetch_start_time = time.perf_counter()

    def after_dataloader(self, state: ObserverState) -> None:
        if not self.active:
            return
        if self._fetch_start_time is not None:
            self.dataloader_fetch_seconds += (
                time.perf_counter() - self._fetch_start_time
            )
            self._fetch_start_time = None

        if self.config.emit_batch_stats and state.labels is not None:
            self._accumulate_label_metrics(state.labels)

    def after_forward(self, state: ObserverState) -> None:
        if not self.active or not self.config.emit_batch_stats or state.loss is None:
            return
        # Kept as a device scalar: .item() here would host-sync mid-step,
        # once per microbatch. The emitter batches the transfer at batch_end.
        self.microbatch_losses.append(state.loss.detach().float().reshape(()))

    def batch_end(self, state: ObserverState) -> None:
        del state
        if not self.active:
            return
        self._record_tag(
            "time_seconds/batch/dataloader_fetch",
            self.dataloader_fetch_seconds,
        )

        if not self.config.emit_batch_stats or self.microbatches == 0:
            return
        self._record_tag("data/microbatches", self.microbatches)
        self._record_tag("data/local_samples", self.local_samples)
        if self.seq_len is not None:
            self._record_tag("data/seq_len", self.seq_len)
        self._record_tag("data/label_tokens", self.label_tokens)
        if self.supervised_label_tokens is not None:
            ignored = self.label_tokens - self.supervised_label_tokens
            self._record_tag(
                "data/supervised_label_tokens", self.supervised_label_tokens
            )
            self._record_tag("data/ignored_label_tokens", ignored)
            self._record_tag(
                "data/ignored_label_ratio", ignored / max(self.label_tokens, 1)
            )
        if self.microbatch_losses:
            losses = torch.stack(self.microbatch_losses)
            self._record_tag("data/microbatch_loss/mean", losses.mean())
            self._record_tag("data/microbatch_loss/max", losses.max())
            self._record_tag("data/microbatch_loss/min", losses.min())

    def _reset_counters(self) -> None:
        self.dataloader_fetch_seconds = 0.0
        self.microbatch_losses: list[torch.Tensor] = []
        self.microbatches = 0
        self.local_samples = 0
        self.seq_len: int | None = None
        self.label_tokens = 0
        # Device scalar for the same no-mid-step-sync reason as the losses.
        self.supervised_label_tokens: torch.Tensor | None = None

    def _accumulate_label_metrics(self, labels: torch.Tensor) -> None:
        self.microbatches += 1
        if labels.ndim >= 1:
            self.local_samples += int(labels.shape[0])
        if labels.ndim >= 2:
            self.seq_len = int(labels.shape[1])

        self.label_tokens += int(labels.numel())
        supervised = (labels != IGNORE_INDEX).sum().float()
        self.supervised_label_tokens = (
            supervised
            if self.supervised_label_tokens is None
            else self.supervised_label_tokens + supervised
        )

    def _record_tag(self, tag: str, value: torch.Tensor | int | float) -> None:
        sample_name, metric_name = tag.rsplit("/", 1)
        self._record(
            Sample(
                name=sample_name,
                value=value,
                metrics=[MetricConfig(name=metric_name)],
            )
        )
