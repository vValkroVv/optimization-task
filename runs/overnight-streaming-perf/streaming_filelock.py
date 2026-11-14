"""Task-scoped stable-inode locks for Mosaic Streaming on NFS."""

from __future__ import annotations

import errno
import fcntl
import os
from contextlib import suppress
from typing import Any, cast

from filelock._unix import UnixFileLock


class NFSUnixFileLock(UnixFileLock):
    """Keep the flock inode stable so independently opened locks still contend."""

    def _release(self) -> None:
        fd = cast("int", self._context.lock_file_fd)
        self._context.lock_file_fd = None
        fcntl.flock(fd, fcntl.LOCK_UN)
        with suppress(OSError):
            os.close(fd)

    def _fallback_to_soft_lock(self) -> None:
        raise OSError(errno.ENOSYS, "flock is unavailable; soft locking is disabled")


class StreamingFileLockBindings:
    """Temporarily bind Streaming's cache and barrier locks to the stable class."""

    def __init__(self) -> None:
        self._originals: list[tuple[Any, Any]] = []
        self._modules: list[tuple[str, Any]] = []

    def __enter__(self) -> "StreamingFileLockBindings":
        from streaming.base import dataset
        from streaming.base.shared import barrier

        modules = (
            ("streaming.base.dataset", dataset),
            ("streaming.base.shared.barrier", barrier),
        )
        try:
            for name, module in modules:
                original = module.FileLock
                self._originals.append((module, original))
                self._modules.append((name, module))
                module.FileLock = NFSUnixFileLock
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def startup_metadata(self) -> dict[str, object]:
        import filelock

        if not self._originals or any(
            module.FileLock is not NFSUnixFileLock for _, module in self._modules
        ):
            raise RuntimeError("Streaming stable-inode lock bindings are not active")
        return {
            "installed_filelock_version": filelock.__version__,
            "selected_lock_class": f"{NFSUnixFileLock.__module__}.{NFSUnixFileLock.__qualname__}",
            "bindings": [name for name, _ in self._modules],
        }

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for module, original in reversed(self._originals):
            module.FileLock = original
        self._originals.clear()
        self._modules.clear()
