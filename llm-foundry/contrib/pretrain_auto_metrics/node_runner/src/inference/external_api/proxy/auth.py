from __future__ import annotations

import asyncio
import time
from base64 import b64encode
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


@dataclass(frozen=True, slots=True)
class IftTokenRefreshError(Exception):
    detail: str


class IftTokenProvider:
    refresh_interval_sec = 3400.0

    def __init__(self, user_pass: str, base_url: str):
        self.user_pass = user_pass
        self.base_url = base_url.rstrip("/")
        self._token: str | None = None
        self._refreshed_at = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self, client: httpx.AsyncClient) -> str:
        token = self._fresh_token()
        if token is not None:
            return token

        async with self._lock:
            token = self._fresh_token()
            if token is not None:
                return token

            self._token = await self._request_token(client)
            self._refreshed_at = time.monotonic()
            return self._token

    def _fresh_token(self) -> str | None:
        if (
            self._token is not None
            and time.monotonic() - self._refreshed_at < self.refresh_interval_sec
        ):
            return self._token
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        reraise=True,
        retry=retry_if_exception_type((httpx.RequestError, IftTokenRefreshError)),
    )
    async def _request_token(self, client: httpx.AsyncClient) -> str:
        encoded = b64encode(self.user_pass.encode("utf-8")).decode("ascii")
        response = await client.post(
            f"{self.base_url}/v1/token",
            headers={
                "Authorization": f"Basic {encoded}",
                "Content-Type": "application/json",
            },
        )

        if response.status_code >= 400:
            raise IftTokenRefreshError(
                f"IFT token refresh failed with status {response.status_code}"
            )

        payload = self._json_payload(response)
        token = payload.get("tok")
        if not isinstance(token, str) or token.strip() == "":
            raise IftTokenRefreshError("IFT token response does not contain tok")
        return token

    @staticmethod
    def _json_payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise IftTokenRefreshError("IFT token response is not valid JSON") from exc

        if not isinstance(payload, dict):
            raise IftTokenRefreshError("IFT token response must be a JSON object")
        return payload
