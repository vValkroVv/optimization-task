import copy
import fcntl
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from ...job_dispatcher import JD
from ...anonymizer import Anonymizer
from ...config_model import (
    GigaMetricEvaluatorConfig,
    GigaMetricJudgedEvaluatorConfig,
    ModelInfoConfig,
)
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import (
    MetricsEvaluator,
    ParallelInferenceEvaluator,
)
from ...subprocess_manager import SubprocessManager


class GigaMetricEvaluator(MetricsEvaluator):
    SHOOTER_NAME = "gigametric"
    # Benchmarks not listed here keep all their exported metrics.
    _METRICS_ALLOWLIST: dict[str, set[str]] = {
        "bank_agents__ai_mentor_factology": {"mean_factology_pass"},
        "thinkbench": {
            "mean_accuracy",
            "mean_precision",
            "mean_recall",
            "mean_reflection_quality",
        },
        "reflectionbench": {
            "mean_score",
            "mean_nback_score",
            "mean_prlt_score",
            "mean_wpt_score",
            "mean_dcigt_score",
        },
    }
    _METRICS_SCALE_DOWN: set[str] = {
        "bbac/macro_average",
        "bbac/mean_Accuracy_insensitive_multihop_choice",
        "bbac/mean_Accuracy_insensitive_multistep_choice",
        "bbac/mean_Accuracy_insensitive_single_call",
        "bbac/mean_Accuracy_insensitive_single_call_robustness",
        "reflectionbench/mean_score",
        "reflectionbench/mean_nback_score",
        "reflectionbench/mean_prlt_score",
        "reflectionbench/mean_wpt_score",
        "reflectionbench/mean_dcigt_score",
    }
    ARCHIVE_PATH = Path("/home/user/papayametric.tar.zst")
    RUNTIME_ROOT = Path("/home/user/papayametric")
    LOG_ROOT = Path("/home/user/papayametric_logs")
    UNZIP_LOG_PATH = Path("/home/user/unzip_papayametric.log")
    UV_BIN_PATH = Path("/home/user/papayametric_uv/bin")
    CONFIG_ROOT = Path("/home/user/gigametric_configs")
    PAM_BENCHMARK_CATALOG = (
        Path(__file__).resolve().parents[4]
        / "configs/gigametric/gigametric_default.yaml"
    )

    def setup(
        self,
        autometric_runs_config: GigaMetricEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._failed_validation_ids: set[str] = set()

        self._create_metrics_dir(self.artifacts_path, self.SHOOTER_NAME)

        configured_base_dir = autometric_runs_config.base_dir
        self.base_dir = (
            Path(configured_base_dir).expanduser().resolve()
            if configured_base_dir is not None
            else Path(self.metrics_dir) / "runs"
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.validation_id = f"pam_{uuid.uuid4().hex}"
        self.config_dir = self.CONFIG_ROOT / uuid.uuid4().hex
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.papayametric_root = self._unzip_papayametric()
        benchmark_catalog = (
            self.papayametric_root / "validation" / "configs" / "pam.yaml"
            if autometric_runs_config.benchmark_catalog == "gigametric"
            else self.PAM_BENCHMARK_CATALOG
        )
        self.config_path = autometric_runs_config.materialise_validation_config(
            target=self.config_dir / "_pam_inline_config.yaml",
            example_full=benchmark_catalog,
            model_name=self.hf_checkpoint_path.name,
        )
        Logger.info(
            f"GigaMetric setup: validation_id={self.validation_id}, "
            f"base_dir={self.base_dir}, config={self.config_path}"
        )

    def run(self) -> None:
        self._validate_runtime()
        cmd = self._build_cmd()
        env = self._build_env()
        Logger.info(f"Starting GigaMetric run: {cmd}")
        self._run_validation(cmd, env, self.validation_id)

    def _run_validation(
        self, cmd: list[str], env: dict[str, str], validation_id: str
    ) -> None:
        self._failed_validation_ids.add(validation_id)
        with (self.base_dir / f"{validation_id}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            scoring_base = self._scoring_base_path(validation_id)
            scoring_base.parent.mkdir(parents=True, exist_ok=True)
            archives_dir = self.base_dir / "scoring"
            archives_dir.mkdir(exist_ok=True)
            try:
                successful = self._run_validation_process(cmd, env, validation_id)
            finally:
                for suffix in (".tar.gz", ".tar.gz.enc"):
                    archive = scoring_base.with_name(f"{validation_id}{suffix}")
                    if archive.exists():
                        target = archives_dir / archive.name
                        temporary = target.with_name(target.name + ".tmp")
                        shutil.copyfile(archive, temporary)
                        temporary.replace(target)
                        alternate_suffix = (
                            ".tar.gz" if suffix == ".tar.gz.enc" else ".tar.gz.enc"
                        )
                        alternate_name = f"{validation_id}{alternate_suffix}"
                        if not scoring_base.with_name(alternate_name).exists():
                            (archives_dir / alternate_name).unlink(missing_ok=True)
            if successful:
                self._failed_validation_ids.discard(validation_id)

    def _run_validation_process(
        self, cmd: list[str], env: dict[str, str], validation_id: str
    ) -> bool:
        raw_log_path = self.LOG_ROOT / f"{validation_id}.log"
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        raw_log_path.unlink(missing_ok=True)
        sm = SubprocessManager(
            cmd,
            raw_log_path,
            cwd=str(self._papayametric_root()),
            env=env,
            run_name=f"GigaMetric {validation_id}",
        )
        try:
            sm.start()
            sm.wait_for_completion()
            sm.log_error()
            return sm.is_successful()
        finally:
            if raw_log_path.exists():
                Anonymizer.anonymize_file(raw_log_path, self.log_file, append=True)
                raw_log_path.unlink()

    def _validate_runtime(self) -> None:
        """Проверяет предусловия subprocess ДО запуска."""
        runtime_python = Path(self.autometric_runs_config.runtime_python)
        if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
            raise FileNotFoundError(
                f"GigaMetric runtime_python не найден/не исполняемый: {runtime_python}. "
                "См. GigaMetricEvaluatorConfig.runtime_python "
                "(default /home/user/papayametric_venv/bin/python)."
            )

        run_validation = self._papayametric_root() / "run_validation.py"
        if not run_validation.is_file():
            raise FileNotFoundError(
                f"GigaMetric run_validation.py не найден: {run_validation} "
                f"(_papayametric_root={self._papayametric_root()}). См. Docker archive "
                "/home/user/papayametric.tar.zst and runtime unpack to "
                "/home/user/papayametric."
            )

        config_path = Path(self.config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"GigaMetric config не найден: {config_path}.")

    def get_metrics(self) -> dict[str, float]:
        raw = self._fetch_raw_metrics()

        local_model_id = self.autometric_runs_config.local_model_id
        if local_model_id is None:
            return raw

        reshaped = self._reshape_metrics(raw, local_model_id)
        return reshaped

    def _fetch_raw_metrics(
        self, validation_id: Optional[str] = None
    ) -> dict[str, float]:
        """Изолированно импортирует подмодуль и собирает сырые метрики."""
        validation_id = validation_id or self.validation_id
        if validation_id in self._failed_validation_ids:
            metrics_path = self.base_dir / f"{validation_id}.pam_metrics.json"
            if not metrics_path.is_file():
                return {}
        papayametric_root = str(self._papayametric_root())
        inserted = False

        if papayametric_root not in sys.path:
            sys.path.insert(0, papayametric_root)
            inserted = True

        try:
            from validation.pam_export import get_pam_metrics

            raw = get_pam_metrics(validation_id, self.base_dir)
        finally:
            if inserted:
                try:
                    sys.path.remove(papayametric_root)
                except ValueError:
                    pass

        if not raw:
            Logger.warning(
                f"GigaMetric returned 0 metrics for validation_id={validation_id}; "
                "JD payload will be empty. Check state.json for failed tasks."
            )
        else:
            Logger.info(
                f"GigaMetric returned {len(raw)} raw metrics for validation_id={validation_id}"
            )

        metrics = {}
        for key, value in raw.items():
            _, benchmark, metric = key.rsplit("/", 2)
            allowed = self._METRICS_ALLOWLIST.get(benchmark)
            if allowed is not None and metric not in allowed:
                continue
            if f"{benchmark}/{metric}" in self._METRICS_SCALE_DOWN:
                value /= 100.0
            if benchmark == "toolathlon":
                timeout = OmegaConf.select(
                    OmegaConf.load(self.config_path),
                    "benchmarks.toolathlon.parameters.timeout_seconds",
                )
                if timeout is not None:
                    key = f"{key.rsplit('_TIMEOUT_', 1)[0]}_TIMEOUT_{timeout}"
            metrics[f"{GigaMetricEvaluator.SHOOTER_NAME}/{key.partition('/')[2]}"] = (
                value
            )
        return metrics

    def _reshape_metrics(
        self, raw: dict[str, float], local_model_id: str
    ) -> dict[str, float]:
        """Оставляет метрики выбранной модели, сохраняя namespace и имя метрики.

        Универсально: на пару (bench, metric_name) — отдельный ключ
        ``gigametric/<bench_id>/<metric_name>``. Поддерживает и одну, и
        несколько метрик на бенч — ничего не схлопывается и не теряется.
        Сегодня pam_export отдаёт одну primary на бенч, но при расширении
        контракта (несколько метрик) адаптер пробросит все без правок.

        Снимаем только сегмент ``<model_id>`` (фильтр по local_model_id),
        сохраняя префикс ``gigametric/`` — чтобы ключи не коллидировали с
        другими шутерами при ``metrics_results.update()`` в metrics_manager.
        Дубль-гард ниже срабатывает только на настоящий дубль (bench+metric).
        """
        model_prefix = f"gigametric/{local_model_id}/"
        reshaped: dict[str, float] = {}

        for key, value in raw.items():
            if not key.startswith(model_prefix):
                continue
            bench_id, _, metric_name = key[len(model_prefix) :].partition("/")
            if not bench_id or not metric_name:
                continue

            out_key = f"gigametric/{bench_id}/{metric_name}"
            if bench_id == "toolathlon":
                out_key += "_no_api"
            if out_key in reshaped:  # pragma: no cover - defensive: raw-ключи уникальны
                Logger.warning(
                    f"GigaMetric: duplicate metric key {out_key} after reshape; keeping first"
                )
                continue
            reshaped[out_key] = value

        if raw and not reshaped:
            self._raise_model_id_error(raw, local_model_id)

        Logger.info(
            f"GigaMetric reshaped {len(reshaped)} metrics for local_model_id={local_model_id}"
        )
        return reshaped

    def _raise_model_id_error(self, raw: dict[str, float], local_model_id: str) -> None:
        """Формирует понятную ошибку в случае опечатки в local_model_id."""
        available_models = sorted(
            {k.split("/")[1] for k in raw if k.startswith("gigametric/")}
        )
        raise ValueError(
            f"GigaMetric: local_model_id={local_model_id!r} matched 0 keys "
            f"(available models in state.json: {available_models})."
        )

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
    def _unzip_papayametric() -> Path:
        archive = GigaMetricEvaluator.ARCHIVE_PATH
        target_dir = GigaMetricEvaluator._get_runtime_path()
        unzip_log_file = GigaMetricEvaluator.UNZIP_LOG_PATH

        if not archive.exists():
            Logger.warning(
                f"GigaMetric archive not found: {archive}. "
                "Falling back to existing source path."
            )
            return GigaMetricEvaluator._papayametric_root()

        Logger.info("Unzipping GigaMetric")
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        if unzip_log_file.exists():
            unzip_log_file.unlink()

        unzip_cmd = [
            "tar",
            "--zstd",
            "-xf",
            archive.as_posix(),
            "-C",
            target_dir.parent.absolute().as_posix(),
        ]
        sm = SubprocessManager(unzip_cmd, unzip_log_file, None, "Unzip GigaMetric")
        sm.start()
        sm.wait_for_completion()
        if unzip_log_file.exists():
            unzip_log_file.unlink()
        sm.raise_for_exit_code()
        Logger.info("Successfully unzipped GigaMetric")
        return target_dir

    @staticmethod
    def _get_runtime_path() -> Path:
        return GigaMetricEvaluator.RUNTIME_ROOT

    @staticmethod
    def _papayametric_root() -> Path:
        runtime_root = GigaMetricEvaluator._get_runtime_path()
        if (runtime_root / "run_validation.py").exists():
            return runtime_root
        return Path(__file__).resolve().parents[4] / "contrib" / "papayametric"

    def _build_cmd(
        self,
        *,
        config_path: Optional[Path] = None,
        validation_id: Optional[str] = None,
    ) -> list[str]:
        runtime_python = self.autometric_runs_config.runtime_python
        cmd = [
            str(runtime_python),
            str(self._papayametric_root() / "run_validation.py"),
            "--config",
            str(config_path or self.config_path),
            "--base-dir",
            str(self.base_dir),
            "--validation-id",
            validation_id or self.validation_id,
            "--storage-base-path",
            str(self._scoring_base_path(validation_id)),
            "storage.offline=true",
        ]
        return cmd

    def _scoring_base_path(self, validation_id: Optional[str] = None) -> Path:
        # Raw model answers must stay on pod-local storage, not NFS.
        return (
            self.RUNTIME_ROOT.parent
            / "papayametric_scoring"
            / (validation_id or self.validation_id)
        )

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.UV_BIN_PATH.exists():
            env["PATH"] = f"{env.get('PATH', '')}:{self.UV_BIN_PATH}"
        env.update(self.autometric_runs_config.env or {})
        python_paths = ["/home/user/rli_sdk", "/home/user/gigarli/src"]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env.update(
            {
                "TRANSFORMERS_OFFLINE": "1",
                "UV_OFFLINE": "1",
                "GZIP_ARTIFACTS": "1",
                "ARTIFACTS_ARCHIVE_ENABLED": "1",
            }
        )
        env.setdefault("RLI_API_BASE", self.get_rli_url())
        return env

    def _gigametric_version(self) -> str:
        per_run = Path(self.base_dir) / self.validation_id / "gigametric_version.txt"
        if per_run.exists():
            try:
                return per_run.read_text(encoding="utf-8").strip().replace("\n", " | ")
            except (OSError, UnicodeDecodeError) as e:
                Logger.error(f"Failed to read {per_run}: {e}")
        return self.autometric_runs_config.shooter_version

    def _upload_runs_to_s3(self, shooter_version: str) -> Optional[str]:
        key_prefix = JD.ci_key_prefix(
            self.hf_checkpoint_path, GigaMetricEvaluator.SHOOTER_NAME, shooter_version
        )
        if key_prefix is None or not JD.upload_artifacts(self.base_dir, key_prefix):
            return None
        return JD.gigaeye_dir_url(key_prefix)

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        shooter_version = self._gigametric_version()
        cache_dir_s3_path = self._upload_runs_to_s3(shooter_version)
        JD.send_evaluator_metrics(
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            jd_metrics=processed_metrics,
            shooter_name=GigaMetricEvaluator.SHOOTER_NAME,
            shooter_version=shooter_version,
            model_info=self.model_info,
            extra_body=(
                {"cache_dir_s3_path": cache_dir_s3_path}
                if cache_dir_s3_path is not None
                else None
            ),
        )


class GigaMetricJudgedEvaluator(GigaMetricEvaluator, ParallelInferenceEvaluator):
    SHOOTER_NAME = "gigametric_judged"

    def setup(
        self,
        autometric_runs_config: GigaMetricJudgedEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        super().setup(
            autometric_runs_config=autometric_runs_config,
            artifacts_path=artifacts_path,
            hf_checkpoint_path=hf_checkpoint_path,
            model_info=model_info,
        )
        self._base_validation_config = self._load_validation_config()
        ignored = autometric_runs_config.unjudged_benchmarks(
            self._base_validation_config
        )
        if ignored:
            Logger.warning(
                f"GigaMetric judged: бенчи без судьи в judged-ране игнорируются: "
                f"{ignored}. Несудейские бенчи замеряет эвалуатор gigametric."
            )
        self._judge_tasks = autometric_runs_config.judges
        Logger.info(f"GigaMetric judged setup OK, {len(self._judge_tasks)} judges")

    def run_on_model_name(self, model_name: str) -> None:
        validation_id = self._judge_validation_id(model_name)
        self._failed_validation_ids.add(validation_id)
        self._validate_runtime()
        config_path = self._write_judged_config(
            model_name, self._judge_tasks[model_name]
        )
        cmd = self._build_cmd(config_path=config_path, validation_id=validation_id)
        env = self._build_env()
        Logger.info(f"GigaMetric judged run ({model_name}): {cmd}")
        self._run_validation(cmd, env, validation_id)

    def get_metrics(self) -> dict[str, float]:
        # Каждый судья пишет свой validation_id; бенчи между судьями не
        # пересекаются, поэтому merge без коллизий.
        raw: dict[str, float] = {}
        for judge_model_name in self._judge_tasks:
            validation_id = self._judge_validation_id(judge_model_name)
            raw.update(self._fetch_raw_metrics(validation_id=validation_id))

        local_model_id = self.autometric_runs_config.local_model_id
        if local_model_id is None:
            return raw
        return self._reshape_metrics(raw, local_model_id)

    def _load_validation_config(self) -> dict[str, Any]:
        cfg = self.autometric_runs_config
        if cfg.validation_config is not None:
            return copy.deepcopy(cfg.validation_config)
        return OmegaConf.to_container(OmegaConf.load(self.config_path), resolve=False)

    def _judge_validation_id(self, judge_model_name: str) -> str:
        return f"{self.validation_id}_{self._model_slug(judge_model_name)}"

    def _write_judged_config(
        self, judge_model_name: str, task_subset: list[str]
    ) -> Path:
        target = self.config_dir / (
            f"_pam_judged_config_{self._model_slug(judge_model_name)}.yaml"
        )
        self.autometric_runs_config.materialise_judged_validation_config(
            target=target,
            base_config=self._base_validation_config,
            judge_model_name=judge_model_name,
            task_subset=task_subset,
        )
        Logger.info(f"GigaMetric judged config materialised to {target}")
        return target

    def _gigametric_version(self) -> str:
        for judge_model_name in self._judge_tasks:
            validation_id = self._judge_validation_id(judge_model_name)
            per_run = Path(self.base_dir) / validation_id / "gigametric_version.txt"
            if not per_run.exists():
                continue
            try:
                return per_run.read_text(encoding="utf-8").strip().replace("\n", " | ")
            except (OSError, UnicodeDecodeError) as e:
                Logger.error(f"Failed to read {per_run}: {e}")
        return self.autometric_runs_config.shooter_version
