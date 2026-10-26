from argparse import Namespace
from dataclasses import dataclass
from typing import ClassVar, List

from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import LLMF_BASE_IMAGES


@dataclass(slots=True, kw_only=True)
class TestJobLLMF(BaseJob):
    script: str
    args: Namespace

    BASE_IMAGES: ClassVar[dict[str, str]] = LLMF_BASE_IMAGES

    def _external_code_libraries(self) -> List[str]:
        return [
            f"{self.llm_f_root}",
            f"{self.llm_f_root}/contrib/composer",
            f"{self.llm_f_root}/contrib/streaming",
            f"{self.llm_f_root}/contrib/gigafsdp",
            f"{self.llm_f_root}/contrib/gigachat-utils/convertation/",
            f"{self.llm_f_root}/contrib/flash-linear-attention",
        ]

    def _script(self) -> str:
        return self.script

    def _before_submit(self) -> None:
        pass

    def _resolve_n_gpus(self) -> int:
        if self.args.modality == "UNIT":
            return 1
        else:
            return self.n_gpus
