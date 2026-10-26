import os
from dataclasses import dataclass
from pathlib import Path

from job_submitter.jobs.autometrics import AutometricsJob


@dataclass(slots=True, kw_only=True)
class AutometricsJobVERL(AutometricsJob):
    @property
    def verl_root(self) -> Path:
        if root_path := os.getenv("VERL_ROOT_PATH"):
            return Path(root_path)
        return (
            Path(__file__).absolute().parent.parent.parent.parent.parent.parent.parent
        )

    def _script(self) -> str:
        if self.submission_type == "slurm":
            return f"cd {self.verl_root}/sber_src/contribs/pretrain_auto_metrics/node_runner && python3 node_runner.py --checkpoint {self.checkpoint_dir} --mode {self.mode}"
        else:
            return f"--wdir {self.verl_root}/sber_src/contribs/pretrain_auto_metrics/node_runner python3 node_runner.py --checkpoint {self.checkpoint_dir} --mode {self.mode}"

    def _external_code_libraries(self) -> list[str]:
        no_sglang_libs = [
            str(self.verl_root),
            str(
                self.verl_root
                / "sber_src"
                / "contribs"
                / "pretrain_auto_metrics"
                / "contrib"
                / "gigafsdp"
            ),
            str(
                self.verl_root
                / "sber_src"
                / "contribs"
                / "pretrain_auto_metrics"
                / "contrib"
                / "gigachat_utils"
                / "convertation"
            ),
            str(
                self.verl_root
                / "sber_src"
                / "contribs"
                / "pretrain_auto_metrics"
                / "contrib"
                / "lm_evaluation_harness"
            ),
        ]
        if self.sglang_tag == "custom":
            no_sglang_libs.append(
                str(
                    self.verl_root
                    / "sber_src"
                    / "contribs"
                    / "pretrain_auto_metrics"
                    / "contrib"
                    / "sglang_pretrain"
                    / "python"
                )
            )
        return no_sglang_libs
