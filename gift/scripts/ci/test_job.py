from __future__ import annotations

from dataclasses import dataclass

from job_submitter.jobs.base import BaseJob


class _RedactedEnvironment(dict[str, str]):
    def __repr__(self) -> str:
        return "<redacted>"


@dataclass(slots=True, kw_only=True)
class TestJobTorchTitan(BaseJob):
    """Cloud test job whose checkout is verified on the GPU worker before execution."""

    script: str
    repo_root: str

    def _script(self) -> str:
        return self.script

    def _external_code_libraries(self) -> list[str]:
        # checkout's submodules must shadow the copies baked into the image
        return [
            self.repo_root,
            f"{self.repo_root}/contrib/streaming",
            f"{self.repo_root}/contrib/flash-linear-attention",
        ]

    def _build_env(self) -> dict[str, str]:
        return _RedactedEnvironment(BaseJob._build_env(self))
