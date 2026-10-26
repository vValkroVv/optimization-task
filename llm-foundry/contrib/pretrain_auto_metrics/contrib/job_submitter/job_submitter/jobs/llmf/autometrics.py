from dataclasses import dataclass
from pathlib import Path

from job_submitter.jobs.autometrics import AutometricsJob


@dataclass(slots=True, kw_only=True)
class AutometricsJobLLMF(AutometricsJob):
    @property
    def llm_f_root(self) -> Path:
        return Path(__file__).absolute().parent.parent.parent.parent.parent.parent

    def _script(self) -> str:
        if self.submission_type == "slurm":
            return f"cd {self.llm_f_root}/contrib/pretrain_auto_metrics/node_runner && python3 node_runner.py --checkpoint {self.checkpoint_dir} --mode {self.mode}"
        else:
            return f"--wdir {self.llm_f_root}/contrib/pretrain_auto_metrics/node_runner python3 node_runner.py --checkpoint {self.checkpoint_dir} --mode {self.mode}"

    def _external_code_libraries(self) -> list[str]:
        no_sglang_libs = [
            str(self.llm_f_root),
            str(self.llm_f_root / "contrib" / "gigafsdp"),
            str(self.llm_f_root / "contrib" / "gigachat-utils" / "convertation"),
            str(
                self.llm_f_root
                / "contrib"
                / "pretrain_auto_metrics"
                / "contrib"
                / "lm_evaluation_harness"
            ),
        ]
        if self.sglang_tag == "custom":
            no_sglang_libs.append(
                str(
                    self.llm_f_root
                    / "contrib"
                    / "pretrain_auto_metrics"
                    / "contrib"
                    / "sglang_pretrain"
                    / "python"
                )
            )
        return no_sglang_libs
