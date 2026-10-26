import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List

from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import PAM_BASE_IMAGES


@dataclass(slots=True, kw_only=True)
class TestSglangPretrainJob(BaseJob):
    # Tell pytest this is not a test class (it has a `Test` prefix but is a
    # job-submission helper).
    __test__ = False

    script: str

    BASE_IMAGES: ClassVar[dict[str, str]] = PAM_BASE_IMAGES["custom"]

    @property
    def sglang_pretrain_root(self) -> Path:
        return Path(__file__).resolve().parents[5]

    def _external_code_libraries(self) -> List[str]:
        return [str(self.sglang_pretrain_root / "python")]

    @staticmethod
    def _make_run_name_suffix(repo_path: Path) -> str:
        commit_sha = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
        author = subprocess.check_output(
            ["git", "-C", str(repo_path), "log", "-1", "--pretty=%an"],
            text=True,
        ).strip()
        return f"{commit_sha} by {author}"

    def _resolve_job_desc(self) -> str:
        run_suffix = self._make_run_name_suffix(self.sglang_pretrain_root)
        return self.exp_name + f" | {run_suffix}"

    def _script(self) -> str:
        return self.script

    def _before_submit(self) -> None:
        pass
