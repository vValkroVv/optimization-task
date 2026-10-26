from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from .client import RLIClient


class RLIClientSession(RLIClient):
    """Reuse HTTP connections until aclose(); RLI sessions close separately."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        verify: bool = True,
        timeout: float = 30.0,
        trace_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            verify=verify,
            timeout=timeout,
            trace_path=trace_path,
        )
        self._client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def _http_client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                verify=self.verify,
                timeout=self.timeout,
            )
        yield self._client

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()
