from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Union

from job_submitter.asserts.train import TrainJobAssertor
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import LLMF_BASE_IMAGES


@dataclass(slots=True, kw_only=True)
class TrainJobLLMF(BaseJob, TrainJobAssertor):
    path_to_config: Union[str, Path]
    script_type: str = "pytorch2"
    use_clearml: bool = False
    script_args: str = ""
    processes_per_worker: int = 8
    enable_auto_track: bool = True
    BASE_IMAGES: ClassVar[dict[str, str]] = LLMF_BASE_IMAGES

    def _script(self) -> str:
        return f"{self.llm_f_root}/scripts/train/train.py {self.path_to_config} use_clearml={str(self.use_clearml).lower()} {self.script_args}"

    def _external_code_libraries(self) -> list[str]:
        return [
            str(self.llm_f_root),
            str(self.llm_f_root / "contrib" / "gigafsdp"),
            str(self.llm_f_root / "contrib" / "composer"),
            str(self.llm_f_root / "contrib" / "streaming"),
            str(self.llm_f_root / "contrib" / "flash-linear-attention"),
        ]

    def _before_submit(self) -> None:
        pass

    def submit(self) -> BaseJob:
        cfg = self.to_config()
        self._anonimize_config(self.path_to_config)
        self.assert_job(self, cfg)
        self._before_submit()
        return self._submit_config(cfg)
