from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional

from git import Repo
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import PAM_BASE_IMAGES, load_pam_base_images


@dataclass(slots=True, kw_only=True)
class TestJobPAM(BaseJob):
    script: str

    BASE_IMAGES: ClassVar[dict[str, str]] = PAM_BASE_IMAGES["custom"]

    @property
    def pam_root(self) -> Path:
        return Path(__file__).absolute().parent.parent.parent.parent.parent.parent

    def _external_code_libraries(self) -> List[str]:
        return [
            str(self.pam_root),
            str(self.pam_root / "contrib" / "gigafsdp"),
            str(self.pam_root / "contrib" / "gigachat_utils" / "convertation"),
            str(self.pam_root / "contrib" / "lm_evaluation_harness"),
            str(self.pam_root / "contrib" / "sglang_pretrain" / "python"),
        ]

    @staticmethod
    def _make_run_name_suffix(repo_path: str) -> str:
        repo = Repo(repo_path)
        last_commit = next(repo.iter_commits())
        commit_sha = str(last_commit)[:8]
        author = last_commit.author
        return f"{commit_sha} by {author}"

    def _resolve_job_desc(self) -> str:
        run_suffix = self._make_run_name_suffix(self.pam_root)
        return self.exp_name + f" | {run_suffix}"

    def _script(self) -> str:
        return self.script

    def _resolve_base_image(self) -> Optional[str]:
        return (
            self.base_image
            or load_pam_base_images(self.pam_root)["custom"][self.region]
        )

    def _before_submit(self) -> None:
        pass
