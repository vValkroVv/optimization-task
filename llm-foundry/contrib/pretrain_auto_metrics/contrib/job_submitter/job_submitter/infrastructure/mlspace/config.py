from __future__ import annotations

import os
from dataclasses import dataclass

from job_submitter.infrastructure.mlspace.exceptions import MlSpaceConfigError

DEFAULT_API_URL = "https://api.ai.cloud.ru/public/v2"
API_URL_ENV = "MLS_API_URL"
CREDENTIAL_ENVS = {
    "client_id": "MLS_CLIENT_ID",
    "client_secret": "MLS_CLIENT_SECRET",
    "api_key": "MLS_API_KEY",
    "workspace_id": "MLS_WORKSPACE_ID",
}


@dataclass(frozen=True, slots=True)
class MlSpaceConfig:
    base_url: str
    client_id: str
    client_secret: str
    api_key: str
    workspace_id: str

    @classmethod
    def from_env(cls) -> MlSpaceConfig:
        values = {
            field: os.environ.get(name, "").strip()
            for field, name in CREDENTIAL_ENVS.items()
        }
        missing = [
            CREDENTIAL_ENVS[field] for field, value in values.items() if not value
        ]
        if missing:
            raise MlSpaceConfigError(
                f"{', '.join(missing)} required for the ML Space public API"
            )
        return cls(
            base_url=os.environ.get(API_URL_ENV, DEFAULT_API_URL).strip().rstrip("/"),
            **values,
        )
