import os
import re
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from omegaconf import OmegaConf
from pydantic import TypeAdapter

from ...config_model import (
    LmmsEvalBenchmarksConfig,
    LmmsEvalEvaluatorConfig,
    LmmsEvalRunConfig,
    LmmsEvalRuntimeTarget,
    ModelInfoConfig,
)
from ...job_dispatcher import JD
from ...job_dispatcher.measurement import MeasurementUpload
from ...job_dispatcher.submetrics import JDMetricEntry
from ...logger import Logger
from ...metrics.evaluators.interface import MetricsEvaluator
from ...subprocess_manager import SubprocessManager, SubprocessManagerError

_PRESERVED_GIGAVISION_ENV = {"GIGAVISION_JOB_MODE", "GIGAVISION_JOB_RUN_ID"}
_LATEST_MERGED_SHOOTER_VERSION = "latest-merged"


class LmmsEvalEvaluator(MetricsEvaluator):
    def setup(
        self,
        autometric_runs_config: LmmsEvalEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(artifacts_path, "lmms_eval")
        self.run_env = self._build_subprocess_env()
        self.lmms_eval_venv_python = autometric_runs_config.lmms_eval_venv_python
        self.lmms_eval_version_file = autometric_runs_config.lmms_eval_version_file
        self.run_exports: dict[str, dict[str, Path]] = {}
        self._jd_metrics: dict[str, JDMetricEntry] = {}
        self.cmds_to_run = self._convert_raw_config_to_bash(
            autometric_runs_config,
            self.metrics_dir,
        )
        Logger.info(
            f"LmmsEvalEvaluator setup complete, {len(self.cmds_to_run)} runs queued"
        )

    def run(self) -> None:
        Logger.info("Starting lmms-eval runs through gigavision-eval")
        self.autometrics_log_file = self.log_file

        for run_name, command in self.cmds_to_run.items():
            self._run_command(command, run_name)

    def get_metrics(self) -> dict[str, float]:
        if self.metrics_dir is None:
            Logger.warning("metrics_dir is not set for LmmsEvalEvaluator")
            return {}

        self._jd_metrics = {}
        for run_name, export_info in self.run_exports.items():
            run_metrics = self._export_run_metrics(run_name, export_info)
            for metric_name, value in run_metrics.items():
                if metric_name in self._jd_metrics:
                    Logger.warning(
                        f"Duplicate lmms-eval metric {metric_name}; "
                        f"overwriting previous entry with the result from {run_name}"
                    )
                self._jd_metrics[metric_name] = value

        metrics: dict[str, float] = {}
        for metric_name, entry in self._jd_metrics.items():
            if isinstance(entry, dict):
                metrics[metric_name] = entry["value"]
                metrics.update(entry["submetrics"])
            else:
                metrics[metric_name] = entry
        Logger.info(f"Collected {len(metrics)} lmms-eval metrics")
        return metrics

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

    def _convert_raw_config_to_bash(
        self,
        lmms_eval_runs_config: LmmsEvalEvaluatorConfig,
        metrics_dir: Path,
    ) -> dict[str, list[str]]:
        before_skip = self._gigavision_eval_runs_to_bash_list(
            lmms_eval_runs_config,
            metrics_dir,
        )
        return {
            k: v
            for k, v in before_skip.items()
            if not self._is_already_calculated(k, metrics_dir)
        }

    @staticmethod
    def _build_subprocess_env(
        base_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        env = dict(os.environ if base_env is None else base_env)
        for key in list(env):
            if key.startswith("GIGAVISION_") and key not in _PRESERVED_GIGAVISION_ENV:
                env.pop(key, None)
        return env

    def _gigavision_eval_runs_to_bash_list(
        self,
        lmms_eval_runs_config: LmmsEvalEvaluatorConfig,
        metrics_dir: Path,
    ) -> dict[str, list[str]]:
        runtime_target = lmms_eval_runs_config.require_runtime_target()
        if self.hf_checkpoint_path is None:
            raise ValueError("hf_checkpoint_path is not set for LmmsEvalEvaluator")
        if self.model_info is None:
            raise ValueError("model_info is not set for LmmsEvalEvaluator")

        _, model_name, _, step = JD.parse_checkpoint_path(self.hf_checkpoint_path)
        configs_dir = metrics_dir / LmmsEvalEvaluatorConfig.CONFIGS_DIRNAME
        configs_dir.mkdir(parents=True, exist_ok=True)

        result: dict[str, list[str]] = {}
        safe_run_names: dict[str, str] = {}
        for raw_run_name, run_config in lmms_eval_runs_config.resolved_runs.items():
            run_name = self._safe_run_name(raw_run_name)
            if run_name in safe_run_names:
                raise ValueError(
                    "lmms_eval run names "
                    f"'{safe_run_names[run_name]}' and '{raw_run_name}' "
                    f"resolve to the same output name '{run_name}'"
                )
            safe_run_names[run_name] = raw_run_name

            output_dir = metrics_dir / run_name
            logs_dir = (self.artifacts_path / "logs" / "lmms_eval" / run_name).resolve()
            config_path = configs_dir / f"{run_name}.yaml"
            config = self._build_gigavision_config(
                lmms_eval_config=lmms_eval_runs_config,
                run_config=run_config,
                model_info=self.model_info,
                model_name=model_name,
                step=step,
                runtime_target=runtime_target,
                output_dir=output_dir,
                logs_dir=logs_dir,
            )
            self._save_gigavision_config(config, config_path)
            self.run_exports[run_name] = {
                "config_path": config_path,
                "output_dir": output_dir,
                "metrics_path": metrics_dir / f"{run_name}_metrics.json",
            }
            result[run_name] = [
                lmms_eval_runs_config.lmms_eval_venv_python,
                "-m",
                "gigavision_eval.cli.main",
                "eval",
                "--config",
                str(config_path),
            ]

        return result

    @classmethod
    def _build_gigavision_config(
        cls,
        *,
        lmms_eval_config: LmmsEvalEvaluatorConfig,
        run_config: LmmsEvalRunConfig,
        model_info: ModelInfoConfig,
        model_name: str,
        step: int,
        runtime_target: LmmsEvalRuntimeTarget,
        output_dir: Path,
        logs_dir: Path,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "username": model_info.username,
            "model": {
                "name": model_name,
                "step": step,
            },
            "results": {
                "output_dir": str(output_dir),
                "export": {
                    # GigaVision's current metrics exporter is JD-backed.  Keep
                    # the exporter disabled for PAM runs; the node runner reads
                    # the local result files instead of publishing them.
                    "jd": {
                        "model_class": model_info.model_class,
                        "size": model_info.size,
                        "username": model_info.username,
                        "enabled": False,
                    },
                },
            },
            "evaluation": {
                "logging": {
                    "logs_dir": str(logs_dir),
                },
                "execution": {
                    "cache": {
                        "cache_dir": str(output_dir.parent / "cache"),
                    },
                },
            },
        }
        runtime_target.apply(config)

        benchmarks = run_config.benchmarks or lmms_eval_config.benchmarks
        benchmarks_dict = cls._benchmarks_to_dict(benchmarks)
        if benchmarks_dict:
            config["evaluation"]["benchmarks"] = benchmarks_dict

        config = cls._deep_merge(config, lmms_eval_config.gigavision)
        config = cls._deep_merge(config, run_config.gigavision)
        config["results"]["export"]["jd"]["enabled"] = False
        return config

    @staticmethod
    def _benchmarks_to_dict(
        benchmarks: LmmsEvalBenchmarksConfig | None,
    ) -> dict[str, Any]:
        if benchmarks is None:
            return {}
        return benchmarks.to_dict()

    @staticmethod
    def _save_gigavision_config(config: dict[str, Any], config_path: Path) -> None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=OmegaConf.create(config), f=config_path)
        Logger.info(f"Saved gigavision-eval config to {config_path}")

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = LmmsEvalEvaluator._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _safe_run_name(run_name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_name)).strip("._")
        return safe or "default"

    @staticmethod
    def _is_already_calculated(run_name: str, metrics_dir: Path) -> bool:
        run_output_dir = metrics_dir / run_name
        return run_output_dir.exists() and any(
            LmmsEvalEvaluator._is_results_file(path)
            for path in LmmsEvalEvaluator._iter_result_files(run_output_dir)
        )

    def _run_command(self, command: list[str], run_name: str) -> None:
        sm = SubprocessManager(
            command,
            self.autometrics_log_file,
            None,
            run_name,
            env=self.run_env,
        )
        sm.start()
        sm.wait_for_completion()
        sm.raise_for_exit_code()

    def _export_run_metrics(
        self,
        run_name: str,
        export_info: dict[str, Path],
    ) -> dict[str, JDMetricEntry]:
        results_dir = self._resolve_results_dir(export_info["output_dir"])
        metrics_path = export_info["metrics_path"]
        command = [
            self.lmms_eval_venv_python,
            "-m",
            "gigavision_eval.cli.main",
            "jd",
            "metrics",
            "--config",
            str(export_info["config_path"]),
            "--results-dir",
            str(results_dir),
            "--output",
            str(metrics_path),
        ]

        try:
            self._run_command(command, f"Export lmms-eval metrics {run_name}")
        except SubprocessManagerError as e:
            Logger.error(f"Failed to export lmms-eval metrics for {run_name}: {e}")
            return {}

        try:
            metrics: dict[str, JDMetricEntry] = TypeAdapter(
                dict[str, JDMetricEntry]
            ).validate_python(self._load_json_file(metrics_path), strict=True)
        except (OSError, ValueError) as e:
            Logger.error(
                f"Failed to read exported lmms-eval metrics {metrics_path}: {e}"
            )
            return {}
        if not metrics:
            Logger.error(f"No JD metrics exported for lmms-eval run {run_name}")
        return metrics

    @staticmethod
    def _is_results_file(path: Path) -> bool:
        return path.name == "results.json" or path.name.endswith("_results.json")

    @staticmethod
    def _is_submission_artifact(path: Path) -> bool:
        return "submissions" in path.parts

    @classmethod
    def _iter_result_files(cls, output_dir: Path) -> list[Path]:
        if not output_dir.exists():
            return []
        return [
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and cls._is_results_file(path)
            and not cls._is_submission_artifact(path)
        ]

    @classmethod
    def _resolve_results_dir(cls, output_dir: Path) -> Path:
        result_files = cls._iter_result_files(output_dir)
        if not result_files:
            return output_dir
        return max(result_files, key=lambda path: path.stat().st_mtime).parent

    def _get_first_run_config_path(self) -> Path | None:
        for export_info in self.run_exports.values():
            return export_info["config_path"]
        return None

    def _sync_jd_dataset_mapping(self, config_path: Path | None) -> None:
        if config_path is None:
            Logger.warning(
                "No lmms-eval run config found, skipping JD metric mapping sync"
            )
            return

        command = [
            self.lmms_eval_venv_python,
            "-m",
            "gigavision_eval.cli.main",
            "jd",
            "mapping",
            "sync",
            "--config",
            str(config_path),
        ]

        try:
            self._run_command(command, "Sync lmms-eval JD metric mapping")
        except SubprocessManagerError as e:
            Logger.warning(
                f"Failed to sync lmms-eval JD metric mapping, metrics send will continue: {e}"
            )

    def _upload_jd_cache(
        self, *, model_name: str, step: int, shooter_version: str
    ) -> str | None:
        cache_s3_paths: set[str] = set()
        clear_cache = False
        latest_merged = shooter_version == _LATEST_MERGED_SHOOTER_VERSION
        publication_id = uuid4().hex
        for run_name, export_info in self.run_exports.items():
            results_dir = self._resolve_results_dir(export_info["output_dir"])
            delivery_name = f"{run_name}_latest_merged" if latest_merged else run_name
            delivery_path = self.metrics_dir / f"{delivery_name}_cache_delivery.json"
            command = [
                self.lmms_eval_venv_python,
                "-m",
                "gigavision_eval.cli.main",
                "jd",
                "cache",
                "upload",
                "--config",
                str(export_info["config_path"]),
                "--results-dir",
                str(results_dir),
                "--model-name",
                model_name,
                "--step",
                str(step),
                "--shooter-version",
                shooter_version,
                "--publication-id",
                publication_id,
                "--output",
                str(delivery_path),
            ]
            try:
                self._run_command(command, f"Upload lmms-eval JD cache {run_name}")
                delivery = self._load_json_file(delivery_path)
            except (SubprocessManagerError, OSError, ValueError) as e:
                Logger.warning(
                    f"Failed to upload lmms-eval JD cache for {run_name}; "
                    f"clearing its JD path: {e}"
                )
                clear_cache = True
                continue

            action = delivery.get("action") if isinstance(delivery, dict) else None
            cache_s3_path = (
                delivery.get("cache_s3_path") if isinstance(delivery, dict) else None
            )
            if action == "publish" and isinstance(cache_s3_path, str) and cache_s3_path:
                cache_s3_paths.add(cache_s3_path)
            elif action == "clear":
                clear_cache = True
                if delivery.get("error"):
                    Logger.warning(
                        f"Failed to upload lmms-eval JD cache for {run_name}; "
                        f"clearing its JD path: {delivery['error']}"
                    )
            elif action == "preserve" and latest_merged:
                clear_cache = True
            elif action != "preserve":
                Logger.warning(
                    f"Invalid lmms-eval JD cache delivery for {run_name}; clearing its JD path"
                )
                clear_cache = True

        if clear_cache:
            return ""
        if len(cache_s3_paths) > 1:
            Logger.warning(
                "Lmms-eval runs uploaded JD caches to different paths; clearing the JD cache path"
            )
            return ""
        return next(iter(cache_s3_paths), None)

    def _commit_jd_cache(
        self, *, model_name: str, step: int, shooter_version: str, cache_s3_path: str
    ) -> bool:
        config_path = self._get_first_run_config_path()
        if config_path is None:
            Logger.error("Cannot mark JD cache as published: no lmms-eval config")
            return False
        command = [
            self.lmms_eval_venv_python,
            "-m",
            "gigavision_eval.cli.main",
            "jd",
            "cache",
            "commit",
            "--config",
            str(config_path),
            "--model-name",
            model_name,
            "--step",
            str(step),
            "--shooter-version",
            shooter_version,
            "--cache-s3-path",
            cache_s3_path,
        ]
        try:
            self._run_command(command, "Mark lmms-eval JD cache as published")
        except SubprocessManagerError as e:
            Logger.error(
                "JD accepted metrics but cache publication marker failed; "
                f"future partial runs need manual recovery: {e}"
            )
            return False
        return True

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        shooter_version = JD.get_shooter_version(self.lmms_eval_version_file)
        body, step = JD.build_metrics_body(
            hf_checkpoint_path=self.hf_checkpoint_path,
            jd_metrics=self._jd_metrics,
            shooter_name="lmms_eval",
            shooter_version=shooter_version,
            model_info=self.model_info,
        )
        if not body["metrics"].get(step):
            Logger.warning("No valid lmms-eval primary metrics to send to JD")
            return
        cache_s3_path = self._upload_jd_cache(
            model_name=body["name"],
            step=step,
            shooter_version=shooter_version,
        )
        if cache_s3_path is not None:
            body["cache_s3_path"] = cache_s3_path

        JD.save_metrics_json(body, step, self.metrics_dir)
        self._sync_jd_dataset_mapping(self._get_first_run_config_path())
        if not JD.send_metrics(body, self.hf_checkpoint_path):
            Logger.warning(
                "Skipping lmms-eval latest-merged publication: versioned JD publication failed"
            )
            return
        Logger.info("lmms-eval metrics successfully sent to JD")
        if cache_s3_path and not self._commit_jd_cache(
            model_name=body["name"],
            step=step,
            shooter_version=shooter_version,
            cache_s3_path=cache_s3_path,
        ):
            return

        latest_merged_cache_s3_path = self._upload_jd_cache(
            model_name=body["name"],
            step=step,
            shooter_version=_LATEST_MERGED_SHOOTER_VERSION,
        )
        if not latest_merged_cache_s3_path:
            Logger.warning(
                "Skipping lmms-eval latest-merged publication: CI cache is unavailable or incomplete"
            )
            return

        latest_sent = JD.send_metrics(
            {
                **body,
                "shooter_version": _LATEST_MERGED_SHOOTER_VERSION,
                "cache_s3_path": latest_merged_cache_s3_path,
            },
            self.hf_checkpoint_path,
        )
        if latest_sent:
            self._commit_jd_cache(
                model_name=body["name"],
                step=step,
                shooter_version=_LATEST_MERGED_SHOOTER_VERSION,
                cache_s3_path=latest_merged_cache_s3_path,
            )
