from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from job_submitter.infrastructure.s3.constants import S3_ADDRESSING_STYLE

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class S3StorageConfig:
    endpoint_url: str = "${oc.env:S3_ENDPOINT_URL}"
    region: str = "${oc.env:AWS_REGION}"
    bucket: str = "${oc.env:S3_BUCKET}"
    key: str = "${oc.env:S3_KEY}"
    aws_access_key_id: str = "${oc.env:AWS_ACCESS_KEY_ID,''}"
    aws_secret_access_key: str = "${oc.env:AWS_SECRET_ACCESS_KEY,''}"
    aws_session_token: str = "${oc.env:AWS_SESSION_TOKEN,''}"
    # TLS verification and addressing style are provider-specific: on-prem
    # proxies (e.g. the o-cluster S3 gateway) terminate TLS with a private cert
    # and require path-style addressing, while rcloud OBS uses the defaults.
    verify_ssl: bool = True
    addressing_style: str = S3_ADDRESSING_STYLE


def load_config(overrides: DictConfig | None = None) -> S3StorageConfig:
    from omegaconf import OmegaConf

    cfg = OmegaConf.structured(S3StorageConfig)
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]


def build_db_url(config: S3StorageConfig) -> str:
    endpoint = urlsplit(config.endpoint_url)
    if not endpoint.scheme or not endpoint.netloc:
        raise ValueError(
            f"Invalid endpoint URL: {config.endpoint_url!r}. "
            "Expected format like https://host"
        )
    return f"{endpoint.scheme}://{config.bucket}.{endpoint.netloc}/{config.key}"
