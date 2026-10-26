import json
import queue
import threading
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from ...logger import Logger
from ..paths import node_trace_dir
from .model import TraceEvent

CLOSE_TIMEOUT_SECONDS = 5.0


class EventWriter:
    def __init__(self, nfs_path: Path, trace_id: str = ""):
        self.trace_id = trace_id
        self._queue: "queue.Queue[Optional[TraceEvent]]" = queue.Queue()
        self._file = None
        self._thread: Optional[threading.Thread] = None
        try:
            nfs_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(nfs_path, "a", buffering=1)
        except Exception as e:
            Logger.warning(f"event writer disabled, cannot open {nfs_path}: {e}")
            return
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    @staticmethod
    def trace_path(artifacts_path: str | Path, node_name: str, trace_id: str) -> Path:
        return Path(
            node_trace_dir(artifacts_path, node_name), f"events-{trace_id}.jsonl"
        )

    def emit(self, **kwargs) -> None:
        kwargs.setdefault("trace_id", self.trace_id)
        try:
            self.write(TraceEvent(**kwargs))
        except ValidationError as e:
            Logger.warning(f"trace event dropped, invalid {kwargs.get('kind')!r}: {e}")

    def write(self, ev: TraceEvent) -> None:
        if self._file is None:
            return
        self._queue.put(ev)

    def close(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=CLOSE_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            Logger.warning(
                f"event writer did not drain in {CLOSE_TIMEOUT_SECONDS}s, "
                f"~{self._queue.qsize()} events are missing from the trace"
            )

    def _drain(self) -> None:
        for ev in iter(self._queue.get, None):
            try:
                line = ev.model_dump(mode="json", exclude_none=True)
                self._file.write(json.dumps(line, separators=(",", ":")) + "\n")
            except Exception as e:
                Logger.warning(f"trace event dropped: {ev.kind}: {e}")
        self._file.close()
