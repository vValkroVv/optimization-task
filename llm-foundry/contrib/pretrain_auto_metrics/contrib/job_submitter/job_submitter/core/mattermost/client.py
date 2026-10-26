from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import requests

from job_submitter.infrastructure.retry import RetryPolicy

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://mm.sberdevices.ru"
_TIMEOUT = 10

_TOKEN_FILE_PATHS = (
    "/home/jovyan/.mm_token",  # SR004, SR006, SR008
)


def _is_retryable(exc: BaseException) -> bool:
    """Retry on network errors and 5xx."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500
    return False


class MattermostClient:
    """HTTP client for Mattermost user lookups."""

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {token}"
        self._cache: Dict[str, dict] = {}

    @classmethod
    def from_env(cls) -> Optional[MattermostClient]:
        """Create from ``MM_TOKEN`` env var or NFS file.  ``None`` if no token."""
        token = cls._resolve_token()
        if not token:
            return None
        return cls(
            base_url=os.environ.get("MM_URL", _DEFAULT_URL),
            token=token,
        )

    @staticmethod
    def _resolve_token() -> str:
        token = os.environ.get("MM_TOKEN", "").strip()
        if token:
            return token

        for path in _TOKEN_FILE_PATHS:
            try:
                p = Path(path)
                if p.is_file():
                    token = p.read_text(encoding="utf-8").strip()
                    if token:
                        logger.debug("MM token loaded from %s", path)
                        return token
            except OSError:
                continue

        return ""

    def get_user(self, username: str) -> dict:
        """Return raw user dict.  Raises on any error."""
        if username in self._cache:
            return self._cache[username]

        data = self._fetch(username)
        self._cache[username] = data
        return data

    @RetryPolicy(
        exceptions=(requests.ConnectionError, requests.Timeout, requests.HTTPError),
        should_retry=_is_retryable,
        delay=lambda attempt, error: 0.5 * 2 ** (attempt - 1),
    )
    def _fetch(self, username: str) -> dict:
        url = f"{self._base_url}/api/v4/users/username/{username}"
        resp = self._session.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
