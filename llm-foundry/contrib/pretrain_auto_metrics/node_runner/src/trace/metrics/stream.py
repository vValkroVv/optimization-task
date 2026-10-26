import json
import math
import threading
from pathlib import Path
from typing import Optional

from ...logger import Logger
from ..paths import node_trace_dir
from .model import MetricSample


class MetricsStream:
    """The file belongs to the node, not to a scraper: every engine on the node runs
    its own scraper thread and they all append here, so writes are serialized."""

    _path: Optional[Path] = None
    _done_path: Optional[Path] = None
    _non_finite: set[str] = set()
    _lock = threading.Lock()

    @classmethod
    def setup(cls, config_manager) -> None:
        cls._non_finite = set()
        trace_dir = node_trace_dir(
            config_manager.artifacts_path, config_manager.node_name
        )
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            Logger.warning(f"metrics stream disabled, cannot create {trace_dir}: {e}")
            cls._path = None
            cls._done_path = None
            return
        trace_id = str(config_manager.job_id)
        cls._path = Path(trace_dir, f"metrics-{trace_id}.jsonl")
        cls._done_path = Path(trace_dir, f"metrics-{trace_id}.done")

    @classmethod
    def write(cls, samples: list[MetricSample]) -> None:
        if cls._path is None or not samples:
            return
        lines = [
            json.dumps(sample.model_dump(), separators=(",", ":"))
            for sample in samples
            if cls._is_finite(sample)
        ]
        if not lines:
            return
        try:
            with cls._lock, open(cls._path, "a") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            Logger.warning(f"metrics sample batch dropped ({len(lines)} rows): {e}")

    @classmethod
    def _is_finite(cls, sample: MetricSample) -> bool:
        if math.isfinite(sample.value):
            return True
        if sample.metric not in cls._non_finite:
            cls._non_finite.add(sample.metric)
            Logger.warning(
                f"{sample.metric} reported a non-finite value: such samples are "
                "left out of the trace, its finite ones are kept"
            )
        return False

    @classmethod
    def close(cls) -> None:
        if cls._done_path is None:
            return
        try:
            cls._done_path.touch()
        except Exception as e:
            Logger.warning(f"metrics completion marker not written: {e}")
