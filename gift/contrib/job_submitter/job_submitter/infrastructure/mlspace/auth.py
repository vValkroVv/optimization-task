from __future__ import annotations

import time
from typing import Optional

import requests

from job_submitter.infrastructure.mlspace.config import MlSpaceConfig
from job_submitter.infrastructure.mlspace.exceptions import MlSpaceApiError

LOGIN_TIMEOUT_SECONDS = 30
DEFAULT_TOKEN_TTL_SECONDS = 300
REFRESH_MARGIN_SECONDS = 60
NOT_LOGGED_IN_EXPIRES_AT = 0.0


class ServiceAuth:
    def __init__(self, config: MlSpaceConfig, session: requests.Session) -> None:
        self._config = config
        self._session = session
        self._token: Optional[str] = None
        self._expires_at = NOT_LOGGED_IN_EXPIRES_AT

    def headers(self, region: str) -> dict[str, str]:
        if (
            self._token is None
            or time.time() > self._expires_at - REFRESH_MARGIN_SECONDS
        ):
            self._login()
        return {
            "Authorization": f"Bearer {self._token}",
            "x-api-key": self._config.api_key,
            "x-workspace-id": self._config.workspace_id,
            "x-region": region,
        }

    def reset(self) -> None:
        self._token = None

    def _login(self) -> None:
        response = self._session.post(
            f"{self._config.base_url}/service_auth",
            json={
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
            },
            timeout=LOGIN_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            raise MlSpaceApiError(
                f"service_auth failed: HTTP {response.status_code} {response.text[:300]}"
            )
        payload = response.json()
        token = payload.get("token") or payload
        self._token = token["access_token"]
        self._expires_at = time.time() + int(
            token.get("expires_in", DEFAULT_TOKEN_TTL_SECONDS)
        )
