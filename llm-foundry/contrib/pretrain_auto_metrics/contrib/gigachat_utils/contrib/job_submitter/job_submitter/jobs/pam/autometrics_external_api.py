from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from job_submitter.asserts.autometrics import AutometricsExternalAPIAssertor
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import PAM_BASE_IMAGES, load_pam_base_images


@dataclass(slots=True, kw_only=True)
class AutometricsJobPAMExternalAPI(BaseJob, AutometricsExternalAPIAssertor):
    path_to_config: Optional[str] = None
    sglang_tag: str = "custom"
    output_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None

    n_workers: int = 1
    n_gpus: int = 1

    BASE_IMAGES: ClassVar[dict[str, dict[str, str]]] = PAM_BASE_IMAGES

    @property
    def pam_root(self) -> Path:
        return Path(__file__).absolute().parent.parent.parent.parent.parent.parent

    def _script(self) -> str:
        if self.submission_type == "slurm":
            raise ValueError("Slurm is not supported for external API")

        script = (
            f"--wdir {self.pam_root}/node_runner "
            f"python3 node_runner.py --config {self.path_to_config}"
        )
        if self.output_dir:
            script += f" --checkpoint {self.output_dir}"
        return script

    @property
    def base_images(self) -> dict[str, dict[str, str]]:
        return load_pam_base_images(self.pam_root)

    def _resolve_base_image(self) -> Optional[str]:
        assert self.sglang_tag in self.base_images, self._pretty_params_print(
            {"sglang_tag": ["opensource", "custom", "custom_vision"]}
        )
        return self.base_image or self.base_images[self.sglang_tag][self.region]

    def _before_submit(self) -> None:
        return None

    def _external_code_libraries(self) -> list[str]:
        no_sglang_libs = [
            str(self.pam_root),
            str(self.pam_root / "contrib" / "gigafsdp"),
            str(self.pam_root / "contrib" / "gigachat_utils" / "convertation"),
            str(self.pam_root / "contrib" / "lm_evaluation_harness"),
        ]
        if self.sglang_tag == "custom":
            no_sglang_libs.append(
                str(self.pam_root / "contrib" / "sglang_pretrain" / "python")
            )
        return no_sglang_libs
