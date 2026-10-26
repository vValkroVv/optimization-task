from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar, Optional

from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...config_model import ModelInfoConfig, PlanbenchEvaluatorConfig
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import MetricsEvaluator
from ...subprocess_pool_runner import SubprocessPoolRunner

_RESULT_FILENAME_BY_TASK: dict[str, str] = {
    "t1": "task_1_plan_generation.json",
}


class PlanbenchEvaluator(MetricsEvaluator):
    WORK_SUBDIR: ClassVar[str] = "run"
    RUNNER_SCRIPT: ClassVar[str] = "run_planbench.py"
    VERSION_FILE: ClassVar[str] = "VERSION"
    VAL_SUBPATH: ClassVar[tuple[str, ...]] = (
        "contrib",
        "LLMs-Planning",
        "planner_tools",
        "VAL",
    )
    OPENAI_API_KEY: ClassVar[str] = "empty"

    def setup(
        self,
        autometric_runs_config: PlanbenchEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self.all_metrics: dict[str, float] = {}

        self._create_metrics_dir(self.artifacts_path, "planbench")
        self.planbench_root = self._resolve_bench_root(
            autometric_runs_config.planbench_root
        )
        self._work_dir = self.metrics_dir / self.WORK_SUBDIR
        self._runner = self.planbench_root / self.RUNNER_SCRIPT
        self._val_dir = self.planbench_root.joinpath(*self.VAL_SUBPATH)

        Logger.info(
            f"PlanbenchEvaluator: repo={self.planbench_root}, "
            f"base_url={autometric_runs_config.base_url}, "
            f"engine={autometric_runs_config.engine}, "
            f"task={autometric_runs_config.task}, "
            f"domains={autometric_runs_config.domains}"
        )

    @classmethod
    def _resolve_bench_root(cls, root: str) -> Path:
        path = Path(root)
        if path.is_absolute():
            return path
        pretrain_root = Path(__file__).resolve().parents[4]
        return pretrain_root / path

    def _build_env(self, base_url: str) -> dict[str, str]:
        env = os.environ.copy()
        env["OPENAI_API_BASE"] = base_url
        env["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        env["VAL"] = str(self._val_dir)
        return env

    def _result_path(self, domain: str) -> Path:
        cfg = self.autometric_runs_config
        return (
            self._work_dir
            / "results"
            / domain
            / cfg.engine
            / _RESULT_FILENAME_BY_TASK[cfg.task]
        )

    def _is_already_calculated(self) -> bool:
        return all(
            self._result_path(d).is_file() for d in self.autometric_runs_config.domains
        )

    def run(self) -> None:
        cfg = self.autometric_runs_config
        if self._is_already_calculated():
            Logger.info(
                f"PlanbenchEvaluator: cached results in {self._work_dir}, skipping run"
            )
            self._collect_metrics()
            return

        self._work_dir.mkdir(parents=True, exist_ok=True)
        proc_env = self._build_env(cfg.require_base_url())
        cmds_to_run = {
            domain: [
                cfg.venv_python,
                str(self._runner),
                *cfg.build_cli_args(domain=domain, work_dir=self._work_dir),
            ]
            for domain in cfg.domains
        }
        SubprocessPoolRunner(
            cmds_to_run=cmds_to_run,
            max_parallel=cfg.max_parallel_domains,
            autometrics_log_file=self.log_file,
            cwd=str(self.planbench_root),
            env=proc_env,
        ).run_all()

        if not any(self._result_path(d).is_file() for d in cfg.domains):
            Logger.error("planbench: every domain failed, skipping metric collection")
            return

        self._collect_metrics()

    def _collect_metrics(self) -> None:
        cfg = self.autometric_runs_config
        metrics: dict[str, float] = {}
        for domain in cfg.domains:
            path = self._result_path(domain)
            if not path.is_file():
                Logger.warning(f"planbench: no results file for {domain} at {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            instances = [i for i in data.get("instances", []) if "llm_correct" in i]
            if not instances:
                Logger.warning(f"planbench: no scored instances in {path}")
                continue
            accuracy = sum(1 for i in instances if i["llm_correct"]) / len(instances)
            metrics[f"planbench/{domain}/{cfg.task}/accuracy"] = accuracy

        self.all_metrics = metrics
        if metrics:
            out_json = self.metrics_dir / "planbench_metrics.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    def get_metrics(self) -> dict[str, float]:
        return dict(self.all_metrics)

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        self._send_metrics_to_jd(metrics)
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        jd_metrics = {
            key.upper() if "/" not in key else key.replace("/", "_").upper(): value
            for key, value in processed_metrics.items()
        }
        scores = ScalarCacheScoresBuilder(self.metrics_dir, jd_metrics)
        cfg = self.autometric_runs_config
        for domain in cfg.domains:
            key = f"PLANBENCH_{domain}_{cfg.task}_ACCURACY".upper()
            try:
                data = json.loads(self._result_path(domain).read_text())
                instances = data["instances"]
                if not all(
                    isinstance(item.get("llm_correct"), bool) for item in instances
                ):
                    Logger.warning(
                        f"planbench CI unavailable for {domain}: unscored instances"
                    )
                    continue
                samples = [
                    (str(item["instance_id"]), float(item["llm_correct"]))
                    for item in instances
                ]
            except (OSError, ValueError, KeyError, TypeError) as error:
                Logger.warning(f"planbench CI unavailable for {domain}: {error}")
                continue
            scores.add_metric(key, samples, {"domain": domain, "task": cfg.task})
        JD.send_evaluator_metrics_with_scores(
            scores=scores,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            shooter_name="planbench",
            shooter_version=JD.get_shooter_version(
                str(self.planbench_root / self.VERSION_FILE)
            ),
            model_info=self.model_info,
        )
