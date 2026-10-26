from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import urlsplit

_FALSE_FLAG_VALUES = frozenset({"0", "false", "no", "off", ""})


def _env_flag(raw: Optional[str], *, default: bool) -> bool:
    """Interpret a textual env flag (``true``/``false``/``1``/``0`` ...)."""
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_FLAG_VALUES


def silence_insecure_request_warnings(verify_ssl: bool) -> None:
    """Mute urllib3 InsecureRequestWarning when TLS verification is intentionally off."""
    if verify_ssl:
        return
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:  # pragma: no cover - urllib3 is always present via requests/boto3
        pass


@dataclass(frozen=True, slots=True)
class S3DatabaseConfig:
    endpoint_url: str
    region: str
    bucket: str
    key_prefix: str
    # Provider-specific transport knobs. On-prem S3 gateways terminate TLS with a
    # private certificate and require path-style addressing; managed S3 keeps the
    # secure virtual-hosted defaults. Both are opt-in via env, so the defaults
    # below leave the managed-cloud behaviour unchanged.
    verify_ssl: bool = True
    force_path_style: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> S3DatabaseConfig:
        source = os.environ if env is None else env
        verify_ssl = _env_flag(source.get("S3_VERIFY_SSL"), default=True)
        silence_insecure_request_warnings(verify_ssl)
        return cls(
            endpoint_url=source.get("S3_ENDPOINT_URL", ""),
            region=source.get("AWS_REGION", "us-east-1"),
            bucket=source.get("S3_BUCKET", ""),
            key_prefix=source.get("S3_KEY_PREFIX", "tracker/"),
            verify_ssl=verify_ssl,
            force_path_style=_env_flag(
                source.get("S3_FORCE_PATH_STYLE"), default=False
            ),
        )

    def object_url(self, key: str) -> str:
        full_key = self.full_key(key)
        endpoint = urlsplit(self.endpoint_url)
        if not endpoint.scheme or not endpoint.netloc:
            raise ValueError(
                f"Invalid S3 endpoint URL: {self.endpoint_url!r}. "
                "Expected format like https://host"
            )
        if self.force_path_style:
            return f"{endpoint.scheme}://{endpoint.netloc}/{self.bucket}/{full_key}"
        return f"{endpoint.scheme}://{self.bucket}.{endpoint.netloc}/{full_key}"

    def full_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}" if self.key_prefix else key
