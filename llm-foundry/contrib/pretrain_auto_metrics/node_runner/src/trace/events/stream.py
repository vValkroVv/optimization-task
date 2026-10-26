import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator, Optional

from .model import JOB_EVENT_ID, EventStatus, PhaseName
from .writer import EventWriter


class EventsStream:
    _writer: Optional[EventWriter] = None

    @classmethod
    def setup(cls, config_manager) -> None:
        if not config_manager.is_master:
            return
        trace_id = str(config_manager.job_id)
        cls._writer = EventWriter(
            EventWriter.trace_path(
                Path(config_manager.artifacts_path),
                config_manager.node_name,
                trace_id,
            ),
            trace_id=trace_id,
        )
        cls._record_job(config_manager)

    @classmethod
    def close(cls, status: EventStatus = "ok") -> None:
        if cls._writer is None:
            return
        cls._emit(kind="job", event="end", name="job", status=status)
        cls._writer.close()
        cls._writer = None

    @classmethod
    def shooter(
        cls,
        evaluator_name: str,
        *,
        phase: PhaseName,
        secondary_model: Optional[str] = None,
    ):
        return cls._event(
            kind="shooter",
            name=evaluator_name,
            phase=phase,
            secondary_model=secondary_model,
            parent_event_id=JOB_EVENT_ID,
        )

    @classmethod
    def barrier_start(cls, stage: str) -> None:
        cls._emit(kind="barrier", event="start", name=stage)

    @classmethod
    def barrier_end(cls, stage: str) -> None:
        cls._emit(kind="barrier", event="end", name=stage, status="ok")

    @classmethod
    def barrier(cls, endpoint: Callable) -> Callable:
        lock = threading.Lock()
        ended: dict[str, bool] = {}

        @wraps(endpoint)
        def wrapper(*args, **kwargs):
            response = endpoint(*args, **kwargs)
            with lock:
                if response.stage not in ended:
                    ended[response.stage] = False
                    cls.barrier_start(response.stage)
                if response.released and not ended[response.stage]:
                    ended[response.stage] = True
                    cls.barrier_end(response.stage)
            return response

        return wrapper

    @classmethod
    @contextmanager
    def _event(cls, *, kind: str, name: str, **fields) -> Iterator[None]:
        cls._emit(kind=kind, event="start", name=name, **fields)
        status, error = "ok", None
        try:
            yield
        except BaseException:
            status, error = "error", traceback.format_exc()[:2000]
            raise
        finally:
            cls._emit(
                kind=kind, event="end", name=name, status=status, error=error, **fields
            )

    @classmethod
    def _emit(cls, **kwargs) -> None:
        if cls._writer is not None:
            cls._writer.emit(**kwargs)

    @classmethod
    def _record_job(cls, config_manager) -> None:
        offset = datetime.now().astimezone().utcoffset()
        model_info = config_manager.config.autometrics.model_info
        cls._emit(
            kind="job",
            event="start",
            name="job",
            run_name=config_manager.config.run_name,
            checkpoint_path=str(config_manager.checkpoint_path),
            model_class=model_info.model_class,
            model_size=model_info.size,
            model_username=model_info.username,
            global_nnodes=config_manager.global_nnodes,
            tz_offset_seconds=offset.total_seconds() if offset is not None else 0.0,
        )
