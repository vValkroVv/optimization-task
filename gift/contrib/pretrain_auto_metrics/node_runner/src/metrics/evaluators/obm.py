import gzip
import json
import math
import os
import shutil
from base64 import b64decode
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from ...job_dispatcher import JD
from ...anonymizer import Anonymizer
from ...config_model import (
    ModelInfoConfig,
    ObmAllConfig,
    ObmEvaluatorConfig,
    ObmJudgedEvaluatorConfig,
)
from ...job_dispatcher.cache_scores import (
    OBM_ENCODED_PREFIX,
    ObmCacheScoresBuilder,
    encoded_path,
    obm_cache_files,
)
from ...job_dispatcher.measurement import MeasurementUpload
from ...job_dispatcher.submetrics import MetricValue
from ...logger import Logger
from ...metrics.evaluators.interface import (
    MetricsEvaluator,
    ParallelInferenceEvaluator,
)
from ...subprocess_manager import SubprocessManager


class ObmEvaluator(MetricsEvaluator):
    OBM_METRICS_FILENAME = "all_metrics.jsonl"
    OBM_PATH_ENV = "PAM_OBM_PATH"
    OBM_ARCHIVE_PATH = "/home/user/obm.tar.zst"
    OBM_DIR_NAME = "one-button-metrics"
    OBM_VENV_PYTHON = "/home/user/obm_env/bin/python"
    SHOOTER_VERSION_FILE = "/home/user/obm_version.txt"
    METRICS_DIR_NAME = "obm"

    @classmethod
    def _unzip_obm(cls) -> Path:
        target_dir = cls._get_obm_path()
        if os.getenv(cls.OBM_PATH_ENV):
            Logger.warning(
                f"Using OBM from {target_dir} ({cls.OBM_PATH_ENV}), skipping unzip"
            )
            return target_dir

        archive = cls.OBM_ARCHIVE_PATH
        unzip_log_file = Path("/home/user/unzip_obm.log")
        Logger.info("Unzipping OBM")

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        if unzip_log_file.exists():
            unzip_log_file.unlink()

        unzip_cmd = [
            "tar",
            "--zstd",
            "-xf",
            archive,
            "-C",
            target_dir.parent.absolute().as_posix(),
        ]
        sm = SubprocessManager(unzip_cmd, unzip_log_file, None, "Unzip OBM")
        sm.start()
        sm.wait_for_completion()
        sm.raise_for_exit_code()
        unzip_log_file.unlink()
        Logger.info("Successfully unzipped OBM")
        return target_dir

    @classmethod
    def _get_obm_path(cls) -> Path:
        override = os.getenv(cls.OBM_PATH_ENV)
        return Path(override) if override else Path("/home/user") / cls.OBM_DIR_NAME

    @classmethod
    def _get_venv_python(cls) -> str:
        return cls.OBM_VENV_PYTHON

    @staticmethod
    def _read_obm_encoded_file(path: str | Path) -> str:
        path = Path(path)
        with gzip.open(path, mode="rt", encoding="utf-8") as fp:
            return fp.read()

    @staticmethod
    def _write_obm_encoded_file(path: str | Path, content: str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, mode="wt", encoding="utf-8") as fp:
            fp.write(content)
        return path

    def setup(
        self,
        autometric_runs_config: ObmEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.original_artifacts_path = artifacts_path
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = artifacts_path
        self.hf_checkpoint_path = hf_checkpoint_path
        self.obm_repo_path = self._unzip_obm()
        self.obm_config = autometric_runs_config
        self.model_info = model_info
        self._obm_metrics_encrypted = False
        self.cache_s3_paths: dict[str, str] = {}
        self._create_obm_configs(self.artifacts_path, self.obm_config)
        self._create_metrics_dir(self.artifacts_path, self.METRICS_DIR_NAME)
        self._create_measurement_files(hf_checkpoint_path, self.METRICS_DIR_NAME)
        autometric_runs_config = self._replace_papaya_group_with_b3r_group(
            autometric_runs_config
        )
        self.cmds_to_run = self._convert_raw_config_to_bash(autometric_runs_config)
        self._assert_setup()

    def _obm_run_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if not JD.internet_is_available():
            env["OBM_ENCRYPT"] = "1"
        return env

    def run(self) -> None:
        Logger.info("Starting OBM runs")
        for run_name, cmd_list in self.cmds_to_run.items():
            Logger.info(f"OBM run {run_name}: {cmd_list}")
            subprocess_manager = SubprocessManager(
                cmd_list,
                self.log_file,
                self.obm_repo_path,
                env=self._obm_run_env(),
            )
            subprocess_manager.start()
            subprocess_manager.wait_for_completion()
            subprocess_manager.log_error()
        Logger.info("All OBM runs completed")

    def get_metrics(self) -> dict[str, MetricValue]:
        if not hasattr(self, "metrics_dir") or self.metrics_dir is None:
            Logger.warning("metrics_dir is not set for ObmEvaluator")
            return {}

        metrics_dir = Path(self.metrics_dir)
        path_to_metrics = self._find_obm_metrics_file(metrics_dir)
        if path_to_metrics is None:
            Logger.warning(f"No OBM metrics files found in {metrics_dir}")
            return {}

        Logger.info(f"Found metrics file: {path_to_metrics}")
        self._obm_metrics_encrypted = path_to_metrics.name.startswith(
            OBM_ENCODED_PREFIX
        )
        processed_metrics = self._parse_metrics(str(path_to_metrics))
        Logger.info(f"Collected {len(processed_metrics)} OBM metrics")
        return processed_metrics

    def measurement_metrics(
        self, metrics: dict[str, MetricValue]
    ) -> dict[str, MetricValue]:
        metrics_file = self._find_obm_metrics_file(Path(self.metrics_dir))
        if metrics_file is None:
            return {}
        measured = self._parse_metrics(str(metrics_file), self._measured_datasets())
        return {name: metrics[name] for name in measured if name in metrics}

    def _measured_datasets(self) -> set[str]:
        run_files = self.measurement_files.paths()
        cache_files = Path(self.metrics_dir).glob(
            f"{ObmCacheScoresBuilder.cache_subdir}/*"
        )
        return set(obm_cache_files(path for path in cache_files if path in run_files))

    def _find_obm_metrics_file(self, metrics_dir: Path) -> Optional[Path]:
        encoded = encoded_path(metrics_dir / self.OBM_METRICS_FILENAME)
        jsonl_candidates = sorted(metrics_dir.glob("*.jsonl"))

        if not JD.internet_is_available() and encoded.exists():
            return encoded
        if jsonl_candidates:
            return jsonl_candidates[0]
        if encoded.exists():
            return encoded
        return None

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        self._copy_obm_log_to_metrics_dir()
        self.cache_s3_paths = JD.upload_metrics_and_cache_scores(
            make_builder=self.make_cache_scores_builder,
            metrics_dir=self.metrics_dir,
            hf_checkpoint_path=self.hf_checkpoint_path,
            shooter_name=self.METRICS_DIR_NAME,
            version_file=self.SHOOTER_VERSION_FILE,
            uploaded=uploaded,
        )
        self._send_metrics_to_jd(metrics)
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.original_artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )

    def make_cache_scores_builder(
        self, run_files: Optional[set[Path]] = None
    ) -> ObmCacheScoresBuilder:
        return ObmCacheScoresBuilder(Path(self.metrics_dir), run_files)

    def _copy_obm_log_to_metrics_dir(self) -> None:
        obm_log_path = self._get_obm_path() / "obm.log"
        if not obm_log_path.exists():
            Logger.warning(f"OBM log file not found: {obm_log_path}")
            return

        save_path = Path(self.metrics_dir) / "obm.log"
        if not JD.internet_is_available():
            Anonymizer.anonymize_file(obm_log_path, save_path)
        else:
            shutil.copy2(obm_log_path, save_path)
        Logger.info(f"Saved OBM log to {save_path}")

    def _create_obm_configs(self, artifacts_path: Path, obm_config: ObmEvaluatorConfig):
        obm_config_path = artifacts_path / "obm_config.yaml"
        OmegaConf.save(config=obm_config.runner_config, f=obm_config_path)
        Logger.info(f"Saved obm_config to {obm_config_path}")

    def _replace_papaya_group_with_b3r_group(
        self, autometric_runs_config: ObmEvaluatorConfig
    ) -> ObmEvaluatorConfig:
        groups = autometric_runs_config.all.groups
        if not groups:
            return autometric_runs_config

        if not JD.internet_is_available():
            sber_name = b64decode("U0JFUg==").decode("utf-8").lower()
            lower_groups = [g.lower() for g in groups]
            if sber_name in lower_groups:
                raise ValueError("Replace $B3R group with PAPAYA group in pam config")
            autometric_runs_config.all.groups = [
                sber_name if g == "papaya" else g for g in lower_groups
            ]

        return autometric_runs_config

    def _assert_setup(self):
        assert self.autometric_runs_config is not None
        assert self.cmds_to_run is not None
        assert self.metrics_dir is not None
        Logger.info(f"OBM evaluator setup OK, {len(self.cmds_to_run)} runs ready")

    def _convert_raw_config_to_bash(
        self, obm_runs_config: ObmEvaluatorConfig
    ) -> dict[str, list[str]]:
        cmds = self._obm_runs_to_bash_list(obm_runs_config, self.metrics_dir)
        Logger.info(f"OBM bash cmds: {cmds}")
        return cmds

    def _generate_obm_context(
        self,
        run_name: str,
        run_cfg: ObmAllConfig,
    ) -> Optional[str]:
        disable_few_shots = run_cfg.disable_few_shots
        disable_suggested_params = run_cfg.disable_suggested_params

        if not disable_few_shots and not disable_suggested_params:
            return None

        context: dict[str, bool] = {}
        if disable_few_shots:
            context["disable_few_shots"] = True
        if disable_suggested_params:
            context["disable_suggested_params"] = True

        ctx_path = (
            self.artifacts_path / f"{self.METRICS_DIR_NAME}_context_{run_name}.yaml"
        )
        OmegaConf.save(config=OmegaConf.create(context), f=ctx_path)
        Logger.info(f"Generated OBM context file: {ctx_path}")
        return str(ctx_path)

    def _obm_runs_to_bash_list(
        self, obm_runs_config: ObmEvaluatorConfig, metrics_dir: Path
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        venv_python = self._get_venv_python()

        cmd = [
            venv_python,
            "-m",
            "obm.text",
            "run",
            "-cfg",
            str(self.artifacts_path / "obm_config.yaml"),
            "-d",
            str(metrics_dir),
        ]

        if groups := obm_runs_config.all.groups:
            cmd += ["--groups"] + [g.lower() for g in groups]

        if tasks := obm_runs_config.all.tasks:
            cmd += ["--tasks"] + list(tasks)

        if ignore_tasks := obm_runs_config.all.ignore_tasks:
            cmd += ["--ignore-tasks"] + list(ignore_tasks)

        if n_workers := obm_runs_config.all.n_workers:
            cmd += ["-w", str(n_workers)]

        if obm_runs_config.all.restart:
            cmd += ["-r"]

        if context := obm_runs_config.all.context:
            cmd += ["-ctx", str(context)]
        else:
            generated_ctx = self._generate_obm_context("all", obm_runs_config.all)
            if generated_ctx:
                cmd += ["-ctx", generated_ctx]

        result["all"] = cmd
        return result

    def _parse_metrics(
        self, path_to_metrics: str, datasets: Optional[set[str]] = None
    ) -> dict[str, MetricValue]:
        Logger.info(f"Parsing OBM metrics from {path_to_metrics}")

        path = Path(path_to_metrics)
        try:
            if path.name.startswith(OBM_ENCODED_PREFIX):
                content = self._read_obm_encoded_file(path)
            else:
                content = path.read_text(encoding="utf-8")
            data = [json.loads(line) for line in content.splitlines() if line.strip()]
        except Exception as e:
            Logger.error(f"Parse error in OBM metrics: {e}")
            return {}

        processed = {}
        for entry in data:
            name = entry.get("name")
            if not name:
                continue
            if datasets is not None and name not in datasets:
                continue
            failed = bool(entry.get("error") or entry.get("cancelled"))
            failed_value = (
                entry.get("statistics", {})
                .get("failed", {})
                .get("request", {})
                .get("value")
            )
            if failed_value is not None:
                try:
                    failed_share = float(failed_value)
                    failed |= not math.isfinite(failed_share) or failed_share > 0
                except (TypeError, ValueError):
                    failed = True
            if failed:
                Logger.warning(
                    f"OBM dataset '{name}' failed, reporting failed: "
                    f"error={entry.get('error')}, cancelled={entry.get('cancelled')}, "
                    f"failed runs={failed_value}"
                )

            metric = entry.get("metric")
            metrics = entry.get("metrics") or []
            candidates = ([metric] if metric else []) + [
                m for m in metrics if isinstance(m, dict)
            ]
            if not candidates:
                if failed:
                    processed[name] = "failed"
                    continue
                Logger.warning(
                    f"OBM dataset '{name}' has no metric or metrics to register, skipping"
                )
                continue

            for metric in candidates:
                metric_name = metric.get("name")
                metric_value = metric.get("value")
                if metric_name is not None and failed:
                    processed[f"{name}/{metric_name}"] = "failed"
                elif metric_name is not None and metric_value is not None:
                    try:
                        processed[f"{name}/{metric_name}"] = float(metric_value)
                    except (TypeError, ValueError):
                        pass

        return processed

    def _should_save_encoded_metrics(self) -> bool:
        return not JD.internet_is_available() or self._obm_metrics_encrypted

    def _save_metrics_json(self, body, step):
        try:
            if self._should_save_encoded_metrics():
                content = json.dumps(
                    body, ensure_ascii=False, indent=4, allow_nan=False
                )
                filename = f"metrics_step_{step}.json.enc"
                save_path = Path(self.metrics_dir) / filename
                self._write_obm_encoded_file(save_path, content)
            else:
                filename = f"metrics_step_{step}.json"
                save_path = Path(self.metrics_dir) / filename
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(body, f, ensure_ascii=False, indent=4, allow_nan=False)
            Logger.info(f"Saved metrics to {save_path}")
        except Exception as e:
            Logger.error(f"Error while saving JSON metrics file: {e}")

    def _send_metrics_to_jd(self, processed_metrics):
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return
        body, step = JD.build_metrics_body(
            hf_checkpoint_path=self.hf_checkpoint_path,
            jd_metrics=processed_metrics,
            shooter_name="obm",
            shooter_version=JD.get_shooter_version(self.SHOOTER_VERSION_FILE),
            model_info=self.model_info,
            extra_body=self.cache_s3_paths,
        )

        self._save_metrics_json(body=body, step=step)
        JD.send_metrics(body, self.hf_checkpoint_path)


class ObmJudgedEvaluator(ObmEvaluator, ParallelInferenceEvaluator):
    METRICS_DIR_NAME = "obm_judged"

    def setup(
        self,
        autometric_runs_config: ObmJudgedEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        config_json = autometric_runs_config.model_dump_json(
            indent=2, exclude={"runner_config"}
        )
        Logger.info(f"OBM judged config: {config_json}")
        self.original_artifacts_path = artifacts_path
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = artifacts_path
        self.hf_checkpoint_path = hf_checkpoint_path
        self.obm_repo_path = self._unzip_obm()
        self.obm_config = autometric_runs_config
        self.model_info = model_info
        self._obm_metrics_encrypted = False
        self.cache_s3_paths: dict[str, str] = {}
        self._create_metrics_dir(self.artifacts_path, self.METRICS_DIR_NAME)
        self._create_measurement_files(hf_checkpoint_path, self.METRICS_DIR_NAME)
        self.autometric_runs_config: ObmJudgedEvaluatorConfig = (
            self._replace_papaya_group_with_b3r_group(autometric_runs_config)
        )
        Logger.info(
            f"OBM judged setup OK, {len(self.autometric_runs_config.judges)} judges"
        )

    def run_on_model_name(
        self,
        model_name: str,
    ) -> None:
        runner_yaml = self._write_runner_yaml(model_name)
        judge_tasks = self.autometric_runs_config.judges[model_name]
        cmd = self._obm_judged_run_cmd(runner_yaml) + ["--tasks", *judge_tasks]
        Logger.info(f"OBM judged run ({model_name}): {cmd}")
        sm = SubprocessManager(
            cmd,
            self.log_file,
            self.obm_repo_path,
            env=self._obm_run_env(),
        )
        sm.start()
        sm.wait_for_completion()
        sm.log_error()

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        self._send_metrics_to_jd(metrics)
        self._copy_obm_log_to_metrics_dir()
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.original_artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )

    def _write_runner_yaml(self, judge_model_name: str) -> Path:
        runner_cfg = self.autometric_runs_config.judged_runner_config(judge_model_name)
        judge_file_stem = judge_model_name.replace("/", "_")
        yaml_path = (
            self.artifacts_path
            / f"{self.METRICS_DIR_NAME}_config_{judge_file_stem}.yaml"
        )
        OmegaConf.save(config=runner_cfg, f=yaml_path)
        return yaml_path

    def _obm_judged_run_cmd(self, runner_yaml: Path) -> list[str]:
        cmd = [
            self._get_venv_python(),
            "-m",
            "obm.text",
            "run",
            "-cfg",
            str(runner_yaml),
            "-d",
            str(self.metrics_dir),
        ]

        if n_workers := self.autometric_runs_config.all.n_workers:
            cmd += ["-w", str(n_workers)]
        if self.autometric_runs_config.all.restart:
            cmd += ["-r"]
        if context := self.autometric_runs_config.all.context:
            cmd += ["-ctx", str(context)]
        else:
            generated_ctx = self._generate_obm_context(
                "all", self.autometric_runs_config.all
            )
            if generated_ctx:
                cmd += ["-ctx", generated_ctx]
        return cmd


class NewObmEvaluator(ObmEvaluator):
    """OBM from the new_obm repo: same CLI and data (shared inside obm.tar.zst
    via hardlinks), only the repo dir, venv and version file differ."""

    OBM_PATH_ENV = "PAM_NEW_OBM_PATH"
    OBM_DIR_NAME = "new_obm"
    OBM_VENV_PYTHON = "/home/user/new_obm_env/bin/python"
    SHOOTER_VERSION_FILE = "/home/user/new_obm_version.txt"
    METRICS_DIR_NAME = "new_obm"


class NewObmJudgedEvaluator(NewObmEvaluator, ObmJudgedEvaluator):
    METRICS_DIR_NAME = "new_obm_judged"
