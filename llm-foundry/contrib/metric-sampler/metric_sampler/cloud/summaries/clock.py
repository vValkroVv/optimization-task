from __future__ import annotations

import threading
import time

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class Clock:
    """The time and the sleeping of the loops. `stopped` is set when the service has to end."""

    def __init__(self) -> None:
        self.stopped = threading.Event()

    def seconds(self) -> int:
        return int(time.time())

    def timestamp(self) -> str:
        return time.strftime(TIMESTAMP_FORMAT, time.gmtime())

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
