from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from job_submitter.core.mattermost.client import MattermostClient

import requests

_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass(frozen=True, slots=True)
class User:
    """A user identity, optionally verified against Mattermost."""

    username: str
    mm_id: str = ""
    first_name: str = ""
    last_name: str = ""
    verified: bool = False
    skipped: bool = False
    error: Optional[str] = None

    @property
    def display_name(self) -> str:
        full = f"{self.first_name} {self.last_name}".strip()
        return full or self.username

    @property
    def is_valid_format(self) -> bool:
        return bool(self.username and _USERNAME_RE.match(self.username))

    @staticmethod
    def _sanitize(raw: str) -> str:
        """Strip whitespace, drop leading ``@``, lowercase."""
        return raw.strip().lstrip("@").lower()

    @classmethod
    def verify(cls, raw_login: str, client: MattermostClient | None = None) -> User:
        """Sanitize -> format check -> MM API -> ``User``.

        Never raises. All outcomes encoded in the returned object.
        """
        username = cls._sanitize(raw_login)

        if not username:
            return cls(username="", error="Empty username after cleanup")

        if not _USERNAME_RE.match(username):
            return cls(
                username=username,
                error=(
                    f"Username '{username}' has invalid characters. "
                    "Allowed: letters, numbers, dots, hyphens, underscores."
                ),
            )

        if client is None:
            return cls(
                username=username,
                skipped=True,
                error="No MM client - verification skipped",
            )

        try:
            data = client.get_user(username)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 404:
                return cls(
                    username=username,
                    error=f"User '{username}' not found in Mattermost",
                )
            return cls(username=username, error=f"Mattermost error: {exc}")
        except requests.RequestException as exc:
            return cls(username=username, error=f"Mattermost unavailable: {exc}")

        return cls(
            username=data.get("username", username),
            mm_id=data.get("id", ""),
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", ""),
            verified=True,
        )
