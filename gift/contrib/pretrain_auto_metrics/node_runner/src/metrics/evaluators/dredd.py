import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...config_model import (
    DreddEvaluatorConfig,
    ModelInfoConfig,
)
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import ScoringMetricsEvaluator
from ...subprocess_manager import SubprocessManager

# Ключи метрик, записываемых dredd в judgment_summary.jsonl (см. contrib/dredd/src/utils/judgment_summary.py)
_SUMMARY_NUMERIC_KEYS = (
    "total",
    "candidate_wins",
    "baseline_wins",
    "ties",
    "failed_judgments",
    "win_rate",
    "loss_rate",
    "tie_rate",
    "score_0_100",
)
_PERCENT_METRIC_KEYS = ("win_rate", "loss_rate", "tie_rate")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


def get_dredd_submodule_path() -> Path:
    pretrain_auto_metrics_root = Path(__file__).resolve().parents[4]
    return pretrain_auto_metrics_root / "contrib" / "dredd"


def _load_summary_from_jsonl(summary_path: Path) -> Optional[dict[str, object]]:
    if not summary_path.exists():
        Logger.warning(f"File not found: {summary_path}")
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            line = next((line for line in f if line.strip()), None)
        if not line:
            return None
        return json.loads(line)
    except (json.JSONDecodeError, OSError) as e:
        Logger.warning(f"Failed to load {summary_path}: {e}")
        return None


class DreddEvaluator(ScoringMetricsEvaluator):
    DREDD_PYTHON = "/home/user/dredd_env/bin/python"
    SCORING_PHASE = "judgement"
    autometric_runs_config: DreddEvaluatorConfig
    _run_failed = False

    def setup(
        self,
        autometric_runs_config: DreddEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(self.artifacts_path, "dredd")
        self._dredd_phase = "generation"
        self._run_failed = False
        self._assert_setup()
        Logger.info(
            "DreddEvaluator setup successfully, ready for inference/scoring phases"
        )

    def _assert_setup(self) -> None:
        assert self.autometric_runs_config is not None
        assert self.metrics_dir is not None
        assert self.artifacts_path is not None
        assert self.hf_checkpoint_path is not None
        assert self.model_info is not None

    @staticmethod
    def _force_full_dataset_evaluation(payload: dict[str, Any]) -> None:
        """PAM always evaluates full datasets — override max_questions everywhere."""
        experiment_overrides = dict(payload.get("experiment", {}))
        if experiment_overrides.get("max_questions") is not None:
            Logger.info(
                "PAM DreddEvaluator forces full dataset evaluation; "
                f"ignoring experiment.max_questions={experiment_overrides.get('max_questions')}"
            )
        experiment_overrides["max_questions"] = None
        payload["experiment"] = experiment_overrides

        for dataset_overrides in payload.get("datasets", []):
            dataset_experiment = dict(dataset_overrides.get("experiment", {}))
            if dataset_experiment.get("max_questions") is not None:
                Logger.info(
                    "PAM DreddEvaluator forces full dataset evaluation; "
                    f"ignoring datasets[].experiment.max_questions for {dataset_overrides.get('name', '<unknown>')}"
                )
            dataset_experiment["max_questions"] = None
            dataset_overrides["experiment"] = dataset_experiment

    def _apply_model_overrides(self, payload: dict[str, Any]) -> None:
        """Set model.path, model.artifacts_dir, and model.name from the checkpoint."""
        model_overrides = dict(payload.get("model", {}))
        model_overrides["path"] = str(self.metrics_dir)
        model_overrides["artifacts_dir"] = ""  # Write arenas directly into metrics_dir
        _, checkpoint_model_name, _, _ = JD.parse_checkpoint_path(
            self.hf_checkpoint_path
        )
        if checkpoint_model_name:
            model_overrides["name"] = checkpoint_model_name
        payload["model"] = model_overrides

    def _convert_raw_config_to_bash(
        self,
        dredd_overrides: DreddEvaluatorConfig,
        phase: Optional[str] = None,
    ) -> dict[str, list[str]]:
        dredd_root = get_dredd_submodule_path()
        payload = copy.deepcopy(dredd_overrides.to_dict())
        self._force_full_dataset_evaluation(payload)
        self._apply_model_overrides(payload)
        payload["merged_config_path"] = str(
            self.metrics_dir / "dredd_merged_config.yaml"
        )

        config_path = str(dredd_root / "configs" / "default.yaml")
        overrides_json = json.dumps(payload, default=str)
        cmd = [
            self.DREDD_PYTHON,
            str(dredd_root / "dredd.py"),
            "--config",
            config_path,
            "--overrides-json",
            overrides_json,
        ]
        if phase is not None:
            cmd.extend(["--phase", phase])
        return {"dredd": cmd}

    def run_phase(self, phase: str) -> None:
        self._dredd_phase = phase
        Logger.info(f"Starting Dredd runs {phase=}")
        dredd_root = get_dredd_submodule_path()
        if not dredd_root.exists():
            self._run_failed = True
            raise RuntimeError(f"Dredd root not found: {dredd_root}")
        cmds_to_run = self._convert_raw_config_to_bash(
            self.autometric_runs_config, phase=phase
        )
        failed_runs = 0
        for run_name, cmd_list in cmds_to_run.items():
            Logger.info(f"Dredd run {run_name}: {cmd_list}")
            sm = SubprocessManager(
                cmd_list,
                self.log_file,
                dredd_root,
                f"Dredd {run_name} {phase}",
            )
            sm.start()
            sm.wait_for_completion()
            sm.log_error()
            if not sm.is_successful():
                self._run_failed = True
                failed_runs += 1
                Logger.error(
                    f"Dredd run {run_name} phase={phase} failed with exit code {sm.exit_code}; "
                    "metrics will not be published"
                )
        if phase == "generation" and failed_runs == len(cmds_to_run):
            raise RuntimeError("Dredd: every generation run failed")
        Logger.info("Dredd runs completed")

    def _get_configured_baselines(self) -> dict[str, str]:
        common_baseline = self.autometric_runs_config.baseline
        mapping: dict[str, str] = {}
        for dataset in self.autometric_runs_config.datasets or []:
            baseline = dataset.baseline or common_baseline
            if baseline is not None and baseline.name:
                mapping[dataset.name] = baseline.name
        return mapping

    @staticmethod
    def _scale_metric(key: str, val: float) -> tuple[str, float]:
        if key in _PERCENT_METRIC_KEYS:
            return key, val / 100.0
        if key == "score_0_100":
            return "score_0_1", val / 100.0
        return key, val

    def _collect_baseline_metrics(
        self,
        arena_name: str,
        baseline_dir: Path,
    ) -> tuple[dict[str, float], int]:
        summary_path = baseline_dir / "judgment_summary.jsonl"
        summary = _load_summary_from_jsonl(summary_path)
        if not summary:
            Logger.error(
                f"Not publishing Dredd metrics: missing or empty {summary_path}"
            )
            return {}, 0
        data_root = Path(
            os.environ.get("DREDD_DATA_ROOT", get_dredd_submodule_path() / "data")
        )
        if (
            not (data_root / "arenas").is_dir()
            and (data_root / "data" / "arenas").is_dir()
        ):
            data_root = data_root / "data"
        questions_path = data_root / "arenas" / f"{arena_name}.jsonl"
        try:
            with questions_path.open(encoding="utf-8") as stream:
                expected = sum(1 for line in stream if line.strip())
        except OSError as exc:
            Logger.error(
                f"Not publishing Dredd metrics: cannot verify coverage against {questions_path}: {exc}"
            )
            return {}, 0
        total = summary.get("total")
        if not isinstance(total, int) or not 0 < total <= expected:
            Logger.error(
                f"Not publishing Dredd metrics from {summary_path}: "
                f"judged={summary.get('total')}, expected={expected}"
            )
            return {}, 0
        prefix = f"dredd/{arena_name}/{baseline_dir.name}"
        metrics: dict[str, float] = {}
        for key in _SUMMARY_NUMERIC_KEYS:
            if key not in summary:
                continue
            val = summary[key]
            if isinstance(val, (int, float)):
                metric_name, metric_val = self._scale_metric(key, float(val))
                metrics[f"{prefix}/{metric_name}"] = metric_val
        return metrics, max(int(summary.get("failed_judgments", 0)), expected - total)

    def get_metrics(self) -> dict[str, float]:
        if self._run_failed:
            Logger.error(
                "Not publishing Dredd metrics: subprocess failed; summaries are diagnostic only"
            )
            return {}
        if self._dredd_phase == "generation":
            return {}
        if self.metrics_dir is None:
            Logger.warning("metrics_dir is not set for DreddEvaluator")
            return {}
        artifacts_arenas = Path(self.metrics_dir) / "arenas"
        if not artifacts_arenas.exists():
            Logger.warning(f"No Dredd artifacts at {artifacts_arenas}")
            return {}
        configured_baselines = self._get_configured_baselines()
        result: dict[str, float] = {}
        failed_judgments = 0
        max_failed_judgments = 80
        default_format = (self.autometric_runs_config.judge.model_extra or {}).get(
            "input_format"
        )
        input_formats = {
            dataset.name: (dataset.model_extra or {})
            .get("judge", {})
            .get("input_format", default_format)
            for dataset in self.autometric_runs_config.datasets or []
        }
        for arena_dir in sorted(artifacts_arenas.iterdir()):
            if not arena_dir.is_dir():
                continue
            arena_name = arena_dir.name
            judgments_dir = arena_dir / "judgments"
            if not judgments_dir.exists():
                continue
            for baseline_dir in sorted(judgments_dir.iterdir()):
                if not baseline_dir.is_dir():
                    continue
                if configured_baselines and arena_name in configured_baselines:
                    if baseline_dir.name != configured_baselines[arena_name]:
                        Logger.info(
                            f"Skipping stale baseline {baseline_dir.name} for {arena_name} "
                            f"(configured: {configured_baselines[arena_name]})"
                        )
                        continue
                metrics, failed = self._collect_baseline_metrics(
                    arena_name, baseline_dir
                )
                if not metrics:
                    return {}
                failed_judgments += failed
                limit = (
                    0
                    if input_formats.get(arena_name, default_format)
                    == "structured_sbs_json"
                    else max_failed_judgments
                )
                if failed > limit or failed_judgments > max_failed_judgments:
                    Logger.error(
                        f"Not publishing Dredd metrics: {arena_name}/{baseline_dir.name} "
                        f"failed={failed} (limit={limit}); "
                        f"run failed={failed_judgments} (limit={max_failed_judgments})"
                    )
                    return {}
                result.update(metrics)
        Logger.info(f"Collected {len(result)} Dredd metrics")
        return result

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

    def _cache_scores(self, metrics: dict[str, float]) -> ScalarCacheScoresBuilder:
        scores = ScalarCacheScoresBuilder(self.metrics_dir, metrics)
        if not (self.metrics_dir / "dredd_merged_config.yaml").is_file():
            Logger.warning("DREDD CI unavailable: merged scoring config is missing")
            return scores
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(get_dredd_submodule_path()), env.get("PYTHONPATH")])
        )
        script = get_dredd_submodule_path() / "export_ci.py"
        manager = SubprocessManager(
            [self.DREDD_PYTHON, str(script), str(self.metrics_dir)],
            self.log_file,
            get_dredd_submodule_path(),
            "DREDD CI export",
            env=env,
        )
        try:
            manager.start()
            manager.wait_for_completion()
        except OSError as error:
            Logger.warning(f"DREDD CI export failed: {error}")
            return scores
        manager.log_error()
        if not manager.is_successful():
            return scores
        judge_name = self.autometric_runs_config.judge.model_name
        for path in self.metrics_dir.glob("arenas/*/judgments/*/ci_quality.json"):
            arena, baseline = path.parents[2].name, path.parent.name
            key = self._normalize_metric_name_for_jd(
                f"{arena}_VS_{baseline}_JUDGE_{judge_name}"
            )
            try:
                summary = _load_summary_from_jsonl(
                    path.parent / "judgment_summary.jsonl"
                )
                if not summary or summary.get("failed_judgments"):
                    continue
                data = json.loads(path.read_text())
                if len(data["samples"]) == summary["total"]:
                    scores.add_metric(
                        key, data["samples"], data["identity"], tolerance=0.0005000001
                    )
            except (OSError, ValueError, KeyError, TypeError) as error:
                Logger.warning(f"DREDD CI unavailable for {key}: {error}")
        return scores

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        metrics_for_jd = self._convert_metrics_to_jd_keys(processed_metrics)
        JD.send_evaluator_metrics_with_scores(
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            scores=self._cache_scores(metrics_for_jd),
            shooter_name="dredd",
            shooter_version=JD.get_shooter_version("/home/user/dredd_version.txt"),
            model_info=self.model_info,
        )

    @staticmethod
    def _normalize_metric_name_for_jd(metric_name: str) -> str:
        normalized = _NON_ALNUM_RE.sub("_", metric_name).strip("_")
        return normalized.upper()

    def _convert_metrics_to_jd_keys(
        self, processed_metrics: dict[str, float]
    ) -> dict[str, float]:
        """
        JD reporting format for Dredd:
        - keep only SCORE_0_1-like metrics
        - key by benchmark + baseline + judge
          (e.g. ARENA_HARD_RU_VS_GPT_4_0613_JUDGE_MINIMAX_M2_7)
        """
        judge_name = self.autometric_runs_config.judge.model_name
        converted: dict[str, float] = {}
        for metric_name, metric_value in processed_metrics.items():
            parts = metric_name.split("/")
            if len(parts) < 4 or parts[0] != "dredd":
                continue
            benchmark_name = parts[1]
            baseline_name = parts[2]
            summary_key = parts[-1]

            # Keep only score metric for JD.
            if summary_key == "score_0_1":
                score_0_1 = float(metric_value)
            else:
                continue

            jd_metric_name = self._normalize_metric_name_for_jd(
                f"{benchmark_name}_VS_{baseline_name}_JUDGE_{judge_name}"
            )
            if not jd_metric_name:
                Logger.warning(f"Skipping metric with empty JD key: {metric_name!r}")
                continue
            if jd_metric_name in converted and converted[jd_metric_name] != score_0_1:
                Logger.warning(
                    f"Duplicate Dredd benchmark metric for JD key {jd_metric_name=}: "
                    f"old={converted[jd_metric_name]=} new={score_0_1=} (metric={metric_name=})",
                )
            converted[jd_metric_name] = score_0_1
        return converted
