import os
import pytest

from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, Optional

from .modality import MODALITIES


@dataclass
class TestStabilityManager:
    unstable_modalities: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "TestStabilityManager":
        unstable_modalities = {
            modality_name.lower(): os.getenv(f"XFAIL_{modality_name}", "0") == "1"
            for modality_name in MODALITIES
        }
        return cls(unstable_modalities=unstable_modalities)

    def is_unstable(self, modality: str) -> bool:
        return self.unstable_modalities.get(modality.lower(), False)


STABILITY_MANAGER = TestStabilityManager.from_env()


def xfail_if_unstable(modality: str, reason: Optional[str] = None):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if STABILITY_MANAGER.is_unstable(modality):
                default_reason = f"{modality.upper()} tests are marked as unstable"
                pytest.xfail(reason or default_reason)
            return func(*args, **kwargs)

        return wrapper

    return decorator