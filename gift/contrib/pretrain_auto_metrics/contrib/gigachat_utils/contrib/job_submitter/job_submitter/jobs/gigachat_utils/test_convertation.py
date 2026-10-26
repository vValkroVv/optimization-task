from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List

from git import Repo
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import LLMF_BASE_IMAGES


@dataclass(slots=True, kw_only=True)
class TestConvertationJobGigachatUtils(BaseJob):
    script: str

    BASE_IMAGES: ClassVar[dict[str, str]] = LLMF_BASE_IMAGES

    @property
    def gigachat_utils_root(self) -> Path:
        return Path(__file__).absolute().parent.parent.parent.parent.parent.parent

    @staticmethod
    def _make_run_name_suffix(repo_path: str) -> str:
        repo = Repo(repo_path)
        last_commit = next(repo.iter_commits())
        commit_sha = str(last_commit)[:8]
        author = last_commit.author
        return f"{commit_sha} by {author}"

    def _external_code_libraries(self) -> List[str]:
        return [
            str(self.llm_f_root / "contrib" / "gigachat-utils"),
            str(self.llm_f_root / "contrib" / "gigafsdp"),
            str(self.llm_f_root / "contrib" / "flash-linear-attention"),
        ]

    def _resolve_job_desc(self) -> str:
        run_suffix = self._make_run_name_suffix(self.gigachat_utils_root)
        return self.exp_name + f" | {run_suffix}"

    def _script(self) -> str:
        return self.script

    def _before_submit(self) -> None:
        pass
