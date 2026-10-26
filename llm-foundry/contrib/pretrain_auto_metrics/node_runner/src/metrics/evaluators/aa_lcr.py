import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

from ...config_model import AaLcrEvaluatorConfig, ModelInfoConfig
from ...job_dispatcher import JD, SubmetricGroup
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...subprocess_manager import SubprocessManager
from .interface import ScoringMetricsEvaluator

_JD_METRIC_RE = re.compile(r"(?:^|/)accuracy(?:_repeat_\d+)?$")


class AaLcrEvaluator(ScoringMetricsEvaluator):
    SHOOTER_NAME = "aa_lcr"
    _JUDGE_OK_STATUSES = ("ok", "skipped_empty_answer")

    autometric_runs_config: AaLcrEvaluatorConfig

    @property
    def checkpoint_model_name(self) -> str:
        _, checkpoint_model_name, _, _ = JD.parse_checkpoint_path(
            self.hf_checkpoint_path
        )
        return checkpoint_model_name

    def setup(
        self,
        autometric_runs_config: AaLcrEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(self.artifacts_path, self.SHOOTER_NAME)
        self._runner_root = Path(autometric_runs_config.runner_root)
        self._setup_path = Path(self.metrics_dir) / "aa_lcr_setup.json"
        self._setup_identity = self._effective_setup()
        self._reuse_generation = self._can_reuse_generation()
        if not self._reuse_generation:
            try:
                self._setup_path.unlink(missing_ok=True)
            except OSError as e:
                Logger.error(f"aa_lcr: failed to remove stale setup record: {e}")

        Logger.info(
            "AaLcrEvaluator setup successfully, ready for inference/scoring phases"
        )

    def _can_reuse_generation(self) -> bool:
        if not self._setup_path.exists() or self._setup_identity is None:
            return False
        setup_record = self._load_runner_json(self._setup_path)
        if (
            not isinstance(setup_record, dict)
            or setup_record.get("semantic_identity") != self._setup_identity
        ):
            return False
        return self._verify_generation_outputs()

    def _write_setup_record(self) -> bool:
        if self._setup_identity is None:
            Logger.error("aa_lcr: setup identity is unavailable")
            return False
        setup_record = {
            "semantic_identity": self._setup_identity,
            "data_locations": {
                "dataset_csv": self.autometric_runs_config.dataset_csv,
                "text_dir": self.autometric_runs_config.text_dir,
            },
        }
        try:
            self._setup_path.write_text(
                json.dumps(setup_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            Logger.error(f"aa_lcr: failed to write {self._setup_path}: {e}")
            return False
        return True

    def run_phase(self, phase: str) -> None:
        if phase == "generation" and self._reuse_generation:
            Logger.info("aa_lcr: reusing compatible generation outputs")
            return
        runner_script = self._runner_root / "run_aa_lcr.py"
        if not runner_script.exists():
            raise RuntimeError(
                f"aa_lcr: runner not found in the image: {runner_script}"
            )
        overrides = self.autometric_runs_config.build_overrides(
            out_dir=Path(self.metrics_dir),
            checkpoint_model_name=self.checkpoint_model_name,
            phase=phase,
        )
        cmd = [
            self.autometric_runs_config.runner_python,
            str(runner_script),
            "--phase",
            phase,
            "--overrides-json",
            json.dumps(overrides, default=str),
        ]
        Logger.info(f"AaLcr run phase={phase}")
        sm = SubprocessManager(
            cmd, Logger.autometrics_log_file, str(self._runner_root), f"AaLcr {phase}"
        )
        sm.start()
        sm.wait_for_completion()
        sm.log_error()
        if not sm.is_successful():
            raise RuntimeError(
                f"aa_lcr: {phase} runner failed with exit code {sm.exit_code}"
            )
        if phase == "generation":
            if not self._verify_generation_outputs():
                raise RuntimeError("aa_lcr: generation outputs are invalid")
            if not self._write_setup_record():
                raise RuntimeError("aa_lcr: failed to save generation setup")
        elif not self._verify_evaluation_outputs():
            raise RuntimeError("aa_lcr: evaluation outputs are invalid")
        Logger.info(f"AaLcr phase={phase} completed")

    @staticmethod
    def _load_runner_json(path: Path) -> Any:
        if not path.exists():
            Logger.error(f"aa_lcr: runner output not found at {path}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeError) as e:
            Logger.error(f"aa_lcr: unreadable runner output at {path}: {e}")
            return None

    def _dataset_question_count(self) -> int | None:
        config = self.autometric_runs_config
        dataset_csv = Path(config.dataset_csv)
        if not dataset_csv.exists():
            Logger.error(f"aa_lcr: dataset not found at {dataset_csv}")
            return None
        try:
            with dataset_csv.open(encoding="utf-8", newline="") as handle:
                count = sum(
                    1
                    for row in csv.DictReader(handle)
                    if (row.get("question_id") or "").strip()
                )
        except (csv.Error, OSError, UnicodeError) as e:
            Logger.error(f"aa_lcr: failed to read dataset at {dataset_csv}: {e}")
            return None
        if config.limit is not None:
            count = min(count, config.limit)
        if not count:
            Logger.error(f"aa_lcr: no questions found in {dataset_csv}")
            return None
        return count

    def _expected_task_count(self) -> int | None:
        config = self.autometric_runs_config
        question_count = self._dataset_question_count()
        if question_count is None:
            return None
        if config.use_gold_as_candidate:
            return question_count
        return question_count * config.n_repeats

    def _verify_generation_outputs(self) -> bool:
        manifest_path = Path(self.metrics_dir) / "outputs" / "manifest.json"
        if not manifest_path.exists():
            Logger.error(
                "aa_lcr: the runner reported success but left no manifest "
                f"at {manifest_path}"
            )
            return False
        manifest = self._load_runner_json(manifest_path)
        if not isinstance(manifest, dict):
            Logger.error(f"aa_lcr: expected a manifest object at {manifest_path}")
            return False
        expected = self._expected_task_count()
        if expected is None:
            return False
        if len(manifest) != expected:
            Logger.error(
                f"aa_lcr: generation produced {len(manifest)} answers instead "
                f"of {expected}: {manifest_path}"
            )
            return False
        failed = sorted(
            task_id
            for task_id, record in manifest.items()
            if not isinstance(record, dict) or record.get("status") != "ok"
        )
        if failed:
            Logger.error(
                f"aa_lcr: generation failed for {len(failed)} of {expected} "
                f"tasks after retries, first: {failed[:5]}"
            )
            return False
        return True

    def _verify_evaluation_outputs(self) -> bool:
        results_path = Path(self.metrics_dir) / "aa_lcr_results.json"
        if not results_path.exists():
            Logger.error(
                "aa_lcr: the runner reported success but left no results "
                f"at {results_path}"
            )
            return False
        rows = self._load_runner_json(results_path)
        if not isinstance(rows, list):
            Logger.error(f"aa_lcr: expected a result list at {results_path}")
            return False
        expected = self._expected_task_count()
        if expected is None:
            return False
        if len(rows) != expected:
            Logger.error(
                f"aa_lcr: the judge scored {len(rows)} answers instead of "
                f"{expected}: {results_path}"
            )
            return False
        failed = sorted(
            (
                row.get("task_id", "") if isinstance(row, dict) else "",
                row.get("judge_status", "") if isinstance(row, dict) else "",
            )
            for row in rows
            if not isinstance(row, dict)
            or row.get("judge_status") not in self._JUDGE_OK_STATUSES
        )
        if failed:
            Logger.error(
                f"aa_lcr: the judge failed for {len(failed)} of {len(rows)} "
                f"answers, first: {failed[:5]}"
            )
            return False
        return True

    def get_metrics(self) -> dict[str, float]:
        if self.metrics_dir is None:
            Logger.warning("metrics_dir is not set for AaLcrEvaluator")
            return {}
        try:
            return self._collect_metrics()
        except Exception as e:
            Logger.error(f"aa_lcr: metric collection failed: {e}")
            return {}

    def _collect_metrics(self) -> dict[str, float]:
        metrics_path = Path(self.metrics_dir) / "aa_lcr_metrics.json"
        if not metrics_path.exists():
            Logger.error(
                "aa_lcr: the runner reported success but left no metrics "
                f"at {metrics_path}"
            )
            return {}
        raw = self._load_runner_json(metrics_path)
        result: dict[str, float] = {}
        for name, value in raw.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                Logger.error(f"aa_lcr: metric {name!r} is not numeric: {value!r}")
                return {}
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                Logger.error(
                    f"aa_lcr: metric {name!r} is not finite: {numeric_value!r}"
                )
                return {}
            result[name] = numeric_value
        headline = result.get("aa_lcr/accuracy")
        if headline is None:
            Logger.error(
                f"aa_lcr: required metric 'aa_lcr/accuracy' is missing in {metrics_path}"
            )
            return {}
        invalid_accuracies = {
            name: value
            for name, value in result.items()
            if _JD_METRIC_RE.search(name) and not 0.0 <= value <= 1.0
        }
        if invalid_accuracies:
            Logger.error(
                f"aa_lcr: accuracy metrics must be in [0, 1]: {invalid_accuracies}"
            )
            return {}
        Logger.info(f"Collected {len(result)} AaLcr metrics")
        return result

    def _effective_setup(self) -> dict[str, Any] | None:
        config = self.autometric_runs_config
        shooter_version = self._runner_version()
        data_fingerprint = self._data_fingerprint()
        if data_fingerprint is None:
            return None
        return {
            "shooter_version": shooter_version,
            "data_fingerprint": data_fingerprint,
            **config.result_identity(),
        }

    def _data_fingerprint(self) -> str | None:
        config = self.autometric_runs_config
        dataset_path = Path(config.dataset_csv)
        text_dir = Path(config.text_dir)
        if not dataset_path.exists():
            Logger.error(f"aa_lcr: dataset not found at {dataset_path}")
            return None
        if not text_dir.is_dir():
            Logger.error(f"aa_lcr: text directory not found at {text_dir}")
            return None
        digest = hashlib.sha256()
        try:
            digest.update(b"dataset\0")
            digest.update(dataset_path.read_bytes())
            files = sorted(
                candidate for candidate in text_dir.rglob("*") if candidate.is_file()
            )
            for path in files:
                digest.update(b"\0document\0")
                digest.update(path.relative_to(text_dir).as_posix().encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
        except OSError as e:
            Logger.error(f"aa_lcr: failed to fingerprint benchmark data: {e}")
            return None
        return digest.hexdigest()

    def _runner_version(self) -> str:
        version_path = self._runner_root.parent / "aa_lcr_version.txt"
        return JD.get_shooter_version(str(version_path))

    def _shooter_version(self) -> str:
        return f"vendored-{self._runner_version()[:12]}"

    def _jd_metric_name(self, metric_tail: str) -> str:
        config = self.autometric_runs_config
        mode = "gold" if config.use_gold_as_candidate else "model"
        repeats = 1 if config.use_gold_as_candidate else config.n_repeats
        name = "_".join(
            (
                self.SHOOTER_NAME,
                metric_tail,
                "judge",
                config.judge.model_name,
                "mode",
                mode,
                "repeats",
                str(repeats),
            )
        )
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_").upper()

    def _convert_metrics_to_jd_keys(
        self, processed_metrics: dict[str, float]
    ) -> dict[str, float]:
        return {
            self._jd_metric_name(name.rsplit("/", 1)[-1]): value
            for name, value in processed_metrics.items()
            if _JD_METRIC_RE.search(name)
        }

    def _cache_scores(
        self, metrics_for_jd: dict[str, float], accuracy: str
    ) -> ScalarCacheScoresBuilder:
        scores = ScalarCacheScoresBuilder(self.metrics_dir, metrics_for_jd)
        if self._verify_evaluation_outputs():
            try:
                rows = self._load_runner_json(
                    Path(self.metrics_dir) / "aa_lcr_results.json"
                )
                repeats = (
                    1
                    if self.autometric_runs_config.use_gold_as_candidate
                    else self.autometric_runs_config.n_repeats
                )
                questions: dict[str, dict[int, float]] = {}
                for row in rows:
                    values = questions.setdefault(row["question_id"], {})
                    if row["repeat"] in values or row["verdict"] not in {
                        "CORRECT",
                        "INCORRECT",
                    }:
                        Logger.warning(
                            "AA-LCR CI unavailable: duplicate repeat or invalid verdict"
                        )
                        return scores
                    values[row["repeat"]] = float(row["verdict"] == "CORRECT")
                if any(
                    set(values) != set(range(1, repeats + 1))
                    for values in questions.values()
                ):
                    Logger.warning(
                        "AA-LCR CI unavailable: incomplete question/repeat population"
                    )
                    return scores
                judge_name = self.autometric_runs_config.judge.model_name
                identity = dict(self._setup_identity or {})
                identity["judge_model"] = judge_name
                scores.add_metric(
                    accuracy,
                    [
                        (uid, sum(values.values()) / repeats)
                        for uid, values in questions.items()
                    ],
                    identity,
                )
            except (ValueError, KeyError, TypeError) as error:
                Logger.warning(f"AA-LCR CI unavailable: {error}")
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
        accuracy = self._jd_metric_name("accuracy")
        JD.send_evaluator_metrics_with_scores(
            scores=self._cache_scores(metrics_for_jd, accuracy),
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            shooter_name=self.SHOOTER_NAME,
            shooter_version=self._shooter_version(),
            model_info=self.model_info,
            submetric_groups=[
                SubmetricGroup(
                    key=accuracy,
                    members=[key for key in metrics_for_jd if key != accuracy],
                    value=accuracy,
                )
            ],
        )

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        try:
            self._send_metrics_to_jd(metrics)
            tar_path = JD.tar_metrics_dir(
                self.metrics_dir, self.artifacts_path, type(self).__name__
            )
            JD.symlink_metrics_for_transfer(
                tar_path, self.hf_checkpoint_path, type(self).__name__
            )
        except Exception as e:
            Logger.error(f"aa_lcr: post-processing failed: {e}")
