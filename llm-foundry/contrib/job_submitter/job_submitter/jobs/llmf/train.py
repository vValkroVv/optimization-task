from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Union

from job_submitter.asserts.train import TrainJobAssertor
from job_submitter.core.job_templates.region_configs import METRIC_SAMPLER_RAW_ROOTS
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import LLMF_BASE_IMAGES
from job_submitter.jobs.llmf.metric_sampler import PRESETS, MetricSamplerConfig, output_dir


@dataclass(slots=True, kw_only=True)
class TrainJobLLMF(BaseJob, TrainJobAssertor):
    path_to_config: Union[str, Path]
    script_type: str = "pytorch2"
    use_clearml: bool = False
    script_args: str = ""
    processes_per_worker: int = 8
    enable_auto_track: bool = True
    # The metric sampler: a preset of metric_sampler_configs/ (light, standard, full) or the path of a YAML file of
    # that shape; None leaves the training config as it is.
    metric_sampler: Optional[str] = None
    # Fields of the metric sampler config to change, nested like its YAML, e.g. {"capture": {"every_n_steps": 4}}.
    metric_sampler_overrides: Dict[str, Any] = field(default_factory=dict)
    # Why this run may exceed what the metric sampler backend takes per step (a short debugging run, say): with a
    # reason the capacity check warns instead of refusing the job.
    metric_sampler_unsafe_reason: Optional[str] = None
    BASE_IMAGES: ClassVar[dict[str, str]] = LLMF_BASE_IMAGES

    def _script(self) -> str:
        sampler = "".join(f"'{override}' " for override in self.metric_sampler_training_overrides())
        return f"{self.llm_f_root}/scripts/train/train.py {self.path_to_config} use_clearml={str(self.use_clearml).lower()} {sampler}{self.script_args}"

    def metric_sampler_training_overrides(self) -> list[str]:
        """The training config overrides that set up the metric sampler, none without it."""
        if self.metric_sampler is None:
            return []
        assert self.metric_sampler in PRESETS or Path(self.metric_sampler).is_file(), self._pretty_params_print(
            {"metric_sampler": [*PRESETS, "/path/to/metric_sampler.yaml"]}
        )
        assert self.region in METRIC_SAMPLER_RAW_ROOTS, self._pretty_params_print(
            {"region": sorted(METRIC_SAMPLER_RAW_ROOTS)}
        )
        training_config = self._load_config(Path(self.path_to_config))
        config = MetricSamplerConfig.load(self.metric_sampler, self.metric_sampler_overrides)
        return config.training_overrides(
            output_dir(METRIC_SAMPLER_RAW_ROOTS[self.region], training_config), training_config
        )

    def _external_code_libraries(self) -> list[str]:
        return [
            str(self.llm_f_root),
            str(self.llm_f_root / "contrib" / "gigafsdp"),
            str(self.llm_f_root / "contrib" / "composer"),
            str(self.llm_f_root / "contrib" / "metric-sampler"),
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
