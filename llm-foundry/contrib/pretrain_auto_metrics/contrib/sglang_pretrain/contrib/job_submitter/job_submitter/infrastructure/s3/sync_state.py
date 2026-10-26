from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RemoteObjectState:
    etag: str | None = None
    version_id: str | None = None
