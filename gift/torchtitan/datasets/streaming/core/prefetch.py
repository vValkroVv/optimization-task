"""Bounded CPU prefetch with a quiescent checkpoint boundary.

Adapted from TorchTitan revision ``860152d111e4bac25daecbe3cc0cd84c68c8af9c``
under the BSD 3-Clause license included in this repository's ``LICENSE``.
"""

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import copy_context
from threading import Condition, Thread
from typing import Generic, TypeVar

_T = TypeVar("_T")


class PrefetchIterator(Iterator[_T], Generic[_T]):
    """One producer owns the source; the consumer owns checkpoint collectives.

    Pause waits for the current item, then freezes both source and ready items.
    The caller must serialize its source state inside that same pause.
    """

    def __init__(
        self,
        source: Callable[[], Iterator[_T]],
        *,
        capacity: int,
        ready: Sequence[_T] = (),
        finished: bool = False,
        initialize: Callable[[], None] | None = None,
    ) -> None:
        if capacity < 1 or len(ready) > capacity:
            raise ValueError(
                "Prefetch capacity must be positive and cover ready items."
            )
        self._source: Callable[[], Iterator[_T]] | None = source
        self._capacity = capacity
        self._ready = deque(ready)
        self._finished = finished
        self._initialize = initialize
        self._condition = Condition()
        self._paused = False
        self._active = False
        self._closed = False
        self._error: BaseException | None = None
        self._thread = Thread(
            target=copy_context().run,
            args=(self._produce,),
            name="streaming-batch-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _produce(self) -> None:
        source = None
        try:
            if self._initialize is not None:
                self._initialize()
            if self._finished:
                return
            factory = self._source
            self._source = None
            assert factory is not None
            source = factory()
            del factory
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._closed
                        or self._finished
                        or (not self._paused and len(self._ready) < self._capacity)
                    )
                    if self._closed or self._finished:
                        return
                    self._active = True
                try:
                    item = next(source)
                except StopIteration:
                    with self._condition:
                        self._finished = True
                    return
                except BaseException as error:
                    with self._condition:
                        self._error = error
                        self._finished = True
                    return
                else:
                    with self._condition:
                        self._ready.append(item)
                finally:
                    with self._condition:
                        self._active = False
                        self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
        finally:
            # Generator contexts and DataLoader workers belong to this thread.
            try:
                close = getattr(source, "close", None)
                if close is not None:
                    close()
            except BaseException as error:
                with self._condition:
                    if self._error is None:
                        self._error = error
            finally:
                with self._condition:
                    self._finished = True
                    self._condition.notify_all()

    def __next__(self) -> _T:
        with self._condition:
            self._condition.wait_for(
                lambda: self._ready or self._finished or self._closed
            )
            if self._ready:
                item = self._ready.popleft()
                self._condition.notify_all()
                return item
            if self._error is not None:
                raise self._error
            raise StopIteration

    @contextmanager
    def paused(self) -> Iterator[tuple[list[_T], bool]]:
        """Serialize ready items and the external source state under this pause."""
        with self._condition:
            self._paused = True
            self._condition.wait_for(lambda: not self._active)
            try:
                if self._error is not None:
                    raise self._error
                yield list(self._ready), self._finished
            finally:
                self._paused = False
                self._condition.notify_all()

    def close(self) -> None:
        """Request shutdown; an in-flight source read finishes in its own thread."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise RuntimeError(
                "The prefetch source has not finished its in-flight read."
            )
