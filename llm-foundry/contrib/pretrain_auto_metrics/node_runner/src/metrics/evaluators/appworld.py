from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...config_model import AppWorldEvaluatorConfig, ModelInfoConfig
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import MetricsEvaluator
from ...subprocess_manager import SubprocessManager


APPWORLD_VERSION_FILE = "/home/user/appworld_version.txt"


class AppWorldEvaluator(MetricsEvaluator):
    LATENCY = True

    def setup(
        self,
        autometric_runs_config: AppWorldEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(self.artifacts_path, "appworld")
        repo_root = Path(__file__).resolve().parents[4]
        self.appworld_root = self._resolve(
            autometric_runs_config.appworld_root, repo_root
        )
        self.metrics_path = self.metrics_dir / "appworld_metrics.json"
        self.fingerprint_path = self.metrics_dir / "appworld_setup.json"

    @staticmethod
    def _resolve(raw: str, repo_root: Path) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else repo_root / path

    @property
    def _experiment_name(self) -> str:
        return (
            "simplified_react_code_agent/openai/"
            f"gpt-4o-mini-2024-07-18/{self.autometric_runs_config.dataset}"
        )

    def _appworld_bin(self) -> str:
        return str(Path(self.autometric_runs_config.venv_python).with_name("appworld"))

    def _eval_json_path(self) -> Path:
        cfg = self.autometric_runs_config
        return (
            self.appworld_root
            / "experiments"
            / "outputs"
            / self._experiment_name
            / "evaluations"
            / f"{cfg.dataset}.json"
        )

    def _fingerprint(self) -> dict[str, object]:
        return {
            "experiment": self._experiment_name,
            "override": json.loads(self.autometric_runs_config.to_override_json()),
            "shooter_version": JD.get_shooter_version(APPWORLD_VERSION_FILE),
        }

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "empty"
        env["APPWORLD_ROOT"] = str(self.appworld_root)
        return env

    def _run_cmd(self, args: list[str], run_name: str) -> None:
        manager = SubprocessManager(
            args,
            self.log_file,
            cwd=str(self.appworld_root),
            run_name=run_name,
            env=self._build_env(),
        )
        manager.start()
        manager.wait_for_completion()
        manager.log_error()

    def _is_already_calculated(self) -> bool:
        if not self.metrics_path.exists() or not self.fingerprint_path.exists():
            return False
        try:
            return self._load_json_file(self.fingerprint_path) == self._fingerprint()
        except (OSError, ValueError, TypeError):
            return False

    def run(self) -> None:
        if self._is_already_calculated():
            Logger.info(
                f"AppWorld already calculated at {self.metrics_path}, skipping run"
            )
            return

        self.metrics_path.unlink(missing_ok=True)
        self.fingerprint_path.unlink(missing_ok=True)
        cfg = self.autometric_runs_config
        Logger.info(f"AppWorld run: experiment={self._experiment_name}")
        self._run_cmd(
            [
                self._appworld_bin(),
                "run",
                self._experiment_name,
                "--override",
                cfg.to_override_json(),
                "--num-processes",
                str(cfg.num_processes),
                "--root",
                str(self.appworld_root),
                "--clear-first",
                "--with-evaluation",
                "--without-setup",
            ],
            "appworld_run",
        )
        self._collect_metrics()
        self.fingerprint_path.write_text(
            json.dumps(self._fingerprint(), indent=2), encoding="utf-8"
        )

    def _collect_metrics(self) -> None:
        cfg = self.autometric_runs_config
        eval_json = self._eval_json_path()
        if not eval_json.exists():
            raise RuntimeError(
                f"AppWorld evaluation file not found after run at {eval_json}"
            )
        try:
            aggregate = self._load_json_file(eval_json)["aggregate"]
            metrics = {
                f"appworld/{cfg.dataset}/task_goal_completion": float(
                    aggregate["task_goal_completion"]
                )
                / 100,
                f"appworld/{cfg.dataset}/scenario_goal_completion": float(
                    aggregate["scenario_goal_completion"]
                )
                / 100,
            }
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"AppWorld evaluation file at {eval_json} is malformed: {error}"
            ) from error
        self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        shutil.copyfile(eval_json, self.metrics_dir / "evaluation.json")

    def get_metrics(self) -> dict[str, float]:
        if not self.metrics_path.exists():
            return {}
        return {
            name: float(value)
            for name, value in self._load_json_file(self.metrics_path).items()
        }

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

    @staticmethod
    def _normalize_metric_name_for_jd(metric_name: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", metric_name).strip("_").upper()

    def _convert_metrics_to_jd_keys(
        self, processed_metrics: dict[str, float]
    ) -> dict[str, float]:
        setup = self._normalize_metric_name_for_jd(
            f"scaffold__simplified_react_code_agent__max_steps__{self.autometric_runs_config.max_steps}"
        )
        return {
            f"{self._normalize_metric_name_for_jd(name)}__{setup}": value
            for name, value in processed_metrics.items()
        }

    def _cache_scores(self, jd_metrics: dict[str, float]) -> ScalarCacheScoresBuilder:
        scores = ScalarCacheScoresBuilder(self.metrics_dir, jd_metrics)
        cfg = self.autometric_runs_config
        try:
            individual = self._load_json_file(self.metrics_dir / "evaluation.json")[
                "individual"
            ]
            roster = (
                self.appworld_root / "data" / "datasets" / f"{cfg.dataset}.txt"
            ).read_text()
            expected = {
                line.strip().split(":")[0]
                for line in roster.splitlines()
                if line.strip()
            }
            scenarios: dict[str, list[float]] = defaultdict(list)
            for uid, record in individual.items():
                value = record.get("success", record.get("passes_fully"))
                if not isinstance(value, bool):
                    Logger.warning(f"AppWorld CI unavailable: unscored task {uid}")
                    return scores
                scenarios[uid.split("_")[0]].append(float(value))
            if set(individual) != expected:
                Logger.warning(
                    "AppWorld CI unavailable: incomplete scenario/task population"
                )
                return scores
        except (OSError, ValueError, KeyError, TypeError) as error:
            Logger.warning(f"AppWorld CI unavailable: {error}")
        else:
            identity = {
                "dataset": cfg.dataset,
                "tasks": sorted(expected),
                "max_steps": cfg.max_steps,
            }
            for key in jd_metrics:
                if "_SCENARIO_GOAL_COMPLETION__" in key:
                    values = [
                        (uid, 100.0 * min(items)) for uid, items in scenarios.items()
                    ]
                elif "_TASK_GOAL_COMPLETION__" in key:
                    values = [
                        (uid, 100.0 * len(scenarios) * sum(items) / len(expected))
                        for uid, items in scenarios.items()
                    ]
                else:
                    continue
                scores.add_metric(key, values, identity, tolerance=0.05000001)
        return scores

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        jd_metrics = self._convert_metrics_to_jd_keys(processed_metrics)
        JD.send_evaluator_metrics_with_scores(
            scores=self._cache_scores(jd_metrics),
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            shooter_name="appworld",
            shooter_version=JD.get_shooter_version(APPWORLD_VERSION_FILE),
            model_info=self.model_info,
        )
