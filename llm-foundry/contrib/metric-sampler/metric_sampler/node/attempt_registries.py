from __future__ import annotations

import json

from metric_sampler.contract.catalog import METRICS
from metric_sampler.contract.raw_layout import AttemptDir, write_atomically


class AttemptRegistries:
    """The attempt-level files that describe the data of every window: the metric catalog and the stream names."""

    def __init__(self, attempt_dir: AttemptDir) -> None:
        self._attempt_dir = attempt_dir

    def write_metrics(self) -> None:
        registry = [
            {
                "metric_id": metric.metric_id,
                "name": metric.name,
                "family": metric.family.value,
                "value_type": metric.value_type.value,
                "node_merge": metric.node_merge.value,
                "unit": metric.unit,
                "description": metric.description,
                "bins": (
                    {"start": metric.bins.start, "width": metric.bins.width, "count": metric.bins.count}
                    if metric.bins is not None
                    else None
                ),
                "summarize_experts": metric.summarize_experts,
                "detailed": metric.detailed,
                "job_identical": metric.job_identical,
                "required": metric.required,
            }
            for metric in METRICS
        ]
        write_atomically(self._attempt_dir.metrics_path, json.dumps(registry, indent=2).encode("utf-8"))

    def write_streams(self, stream_names: tuple[str, ...]) -> None:
        streams = [{"stream_id": index, "name": name} for index, name in enumerate(stream_names)]
        write_atomically(self._attempt_dir.streams_path, json.dumps(streams, indent=2).encode("utf-8"))
