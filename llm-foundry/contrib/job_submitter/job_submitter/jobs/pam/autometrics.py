from dataclasses import dataclass
from pathlib import Path

from job_submitter.jobs.autometrics import AutometricsJob
from job_submitter.jobs.image_registry import load_pam_base_images


@dataclass(slots=True, kw_only=True)
class AutometricsJobPAM(AutometricsJob):
    @property
    def pam_root(self) -> Path:
        return Path(__file__).absolute().parent.parent.parent.parent.parent.parent

    @property
    def base_images(self) -> dict[str, dict[str, str]]:
        return load_pam_base_images(self.pam_root)

    def _script(self) -> str:
        if self.submission_type == "slurm":
            return f"cd {self.pam_root}/node_runner && python3 node_runner.py --checkpoint {self.checkpoint_dir} --mode {self.mode}"
        else:
            return f"--wdir {self.pam_root}/node_runner python3 node_runner.py --checkpoint {self.checkpoint_dir} --mode {self.mode}"

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
