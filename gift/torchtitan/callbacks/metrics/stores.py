from pathlib import Path
import uuid

import torch
import torch.distributed as dist

from torchtitan.callbacks.metrics.sample import MetricRecord
from torchtitan.config.schema.callbacks.metrics.stores import (
    MetricStoreConfig,
    StoreMode,
)


class MetricStore:
    def __init__(self, config: MetricStoreConfig) -> None:
        self.config = config
        self._records: list[MetricRecord] = []

    def write(self, record: MetricRecord) -> None:
        if self.config.mode == StoreMode.none:
            return
        cpu_record = self._to_cpu(record)
        if self.config.mode == StoreMode.cpu:
            self._records.append(cpu_record)
            return
        self._write_disk(cpu_record)

    def records(self) -> list[MetricRecord]:
        return list(self._records)

    def _to_cpu(self, record: MetricRecord) -> MetricRecord:
        value = record.value
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        return MetricRecord(
            name=record.name,
            value=value,
            step=record.step,
            reduce=record.reduce,
        )

    def _write_disk(self, record: MetricRecord) -> None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        directory = Path(self.config.directory) / f"step_{record.step}" / f"rank_{rank}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{record.name.replace('/', '__')}-{uuid.uuid4().hex}.pt"
        torch.save(record, path)
