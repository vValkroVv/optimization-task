import time
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

PhaseName = Literal["single", "scoring", "parallel"]
EventKind = Literal["barrier", "phase", "shooter", "job"]
EventStatus = Literal["ok", "error", "timeout"]

JOB_EVENT_ID = "job"


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: EventKind
    event: Literal["start", "end"]

    trace_id: str
    name: str
    ts: float = Field(default_factory=time.time)

    parent_event_id: Optional[str] = None
    phase: Optional[PhaseName] = None
    secondary_model: Optional[str] = None

    status: Optional[EventStatus] = None
    error: Optional[str] = None

    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    model_class: Optional[str] = None
    model_size: Optional[str] = None
    model_username: Optional[str] = None
    global_nnodes: Optional[int] = None
    tz_offset_seconds: Optional[float] = None

    @model_validator(mode="after")
    def _check_shape(self) -> "TraceEvent":
        if self.kind in ("phase", "shooter") and self.phase is None:
            raise ValueError(f"{self.kind} event requires phase")
        if self.kind != "shooter" and self.secondary_model is not None:
            raise ValueError("secondary_model only valid for kind='shooter'")
        if self.event == "end" and self.status is None:
            raise ValueError("end event requires status")
        return self

    @computed_field
    @property
    def event_id(self) -> str:
        if self.kind == "barrier":
            return f"barrier:{self.name}"
        if self.kind == "phase":
            return f"phase:{self.phase}"
        if self.kind == "job":
            return JOB_EVENT_ID
        return f"shooter:{self.phase}:{self.name}:{self.secondary_model or ''}"
