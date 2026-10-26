from __future__ import annotations

import os
from enum import Enum

CLOUD_API_ENV = "JOB_SUBMITTER_CLOUD_API"


class CloudApi(str, Enum):
    GATEWAY = "gateway"
    PUBLIC = "public"

    @classmethod
    def from_env(cls) -> CloudApi:
        value = os.environ.get(CLOUD_API_ENV, cls.GATEWAY.value).strip().lower()
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"{CLOUD_API_ENV} must be {cls.GATEWAY.value} or {cls.PUBLIC.value}, got {value!r}"
            ) from None
